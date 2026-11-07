"""LLM planner: prompts the client, strictly parses/validates JSON, retries malformed output, records every call."""
from __future__ import annotations
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from src.models.schemas import Mission, FlightPlan
from src.planner.llm import LLMClient, LLMError
from src.planner.prompts import SYSTEM_PROMPT, PROMPT_VERSION, PROMPT_MODES, initial_prompt


def _call_cost(client, r):
    """USD estimate for one call from the model registry; None when prices are unknown (never reported as 0)."""
    spec = getattr(client, "spec", None)
    if spec is None:
        return None
    from src.planner.model_registry import cost_usd
    return cost_usd(spec, r.prompt_tokens, r.completion_tokens)


@dataclass
class PlannerOutput:
    plan: FlightPlan | None
    calls: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def tokens(self):
        return sum((c["prompt_tokens"] or 0) + (c["completion_tokens"] or 0) for c in self.calls)


def extract_json(text: str):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()   # reasoning models (Qwen3, DeepSeek-R1 ...) emit a think block first
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    return json.loads(t)  # strict: the whole payload must be one JSON document


def normalize_plan_dict(d, enabled: bool = True):
    """Repair ONLY benign format slips (never geometry): drop START/DEST entries from 'waypoints' (start is implicit, destination has its own
    field) and ignore unknown extra fields. Returns (dict, notes). Applied identically to every condition and recorded per call
    (call['normalized']), so a strict-vs-normalized sensitivity analysis is possible. Set normalize_format: false for strict parsing."""
    notes: list[str] = []
    if not enabled or not isinstance(d, dict) or not isinstance(d.get("waypoints"), list):
        return d, notes
    d = dict(d)
    keep = ("id", "x", "y", "altitude")
    wps = []
    for w in d["waypoints"]:
        if isinstance(w, dict):
            wid = str(w.get("id", "")).strip().upper()
            if wid == "START" or wid in ("DEST", "DESTINATION"):
                notes.append(f"dropped {wid} entry from waypoints")
                continue
            extra = sorted(k for k in w if k not in keep)
            if extra:
                notes.append(f"ignored unknown waypoint fields {extra}")
                w = {k: w[k] for k in keep if k in w}
        wps.append(w)
    d["waypoints"] = wps
    dst = d.get("destination")
    if isinstance(dst, dict):
        extra = sorted(k for k in dst if k not in ("x", "y", "altitude", "id"))
        if extra:
            notes.append(f"ignored unknown destination fields {extra}")
            d["destination"] = {k: dst[k] for k in ("x", "y", "altitude", "id") if k in dst}
    return d, notes


def derive_seed(seed: int, role: str, idx: int) -> int:
    return (seed * 1009 + idx * 9176 + (0 if role == "initial" else 1)) % (2 ** 31)


class LLMPlanner:
    def __init__(self, client: LLMClient, temperature: float = 0.0, max_tokens: int = 2000, max_parse_retries: int = 2, log_prompts: bool = True, normalize_format: bool = True, prompt_mode: str = "spec"):
        if prompt_mode not in PROMPT_MODES:
            raise ValueError(f"prompt_mode must be one of {PROMPT_MODES}")
        self.prompt_mode = prompt_mode
        self.client, self.temperature, self.max_tokens = client, temperature, max_tokens
        self.normalize_format = normalize_format
        self.max_parse_retries, self.log_prompts = max_parse_retries, log_prompts

    def config(self) -> dict:
        return {**self.client.config(), "temperature": self.temperature, "max_tokens": self.max_tokens,
                "max_parse_retries": self.max_parse_retries, "prompt_version": PROMPT_VERSION,
                "normalize_format": self.normalize_format, "prompt_mode": self.prompt_mode}

    def _call(self, role: str, mission: Mission, user: str, seed: int, temperature: float | None = None) -> PlannerOutput:
        temp = self.temperature if temperature is None else temperature
        out = PlannerOutput(None)
        err = None
        for attempt in range(self.max_parse_retries + 1):
            u = user if attempt == 0 else (user + f"\n\n[PARSE_ERROR] Your previous reply was rejected: {err}\nReply with ONLY the JSON object in the required schema.")
            call = {"role": role, "attempt": attempt, "provider": self.client.name, "model": self.client.model, "is_mock": self.client.is_mock,
                    "prompt_version": PROMPT_VERSION, "prompt_mode": self.prompt_mode, "temperature": temp,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "prompt_sha256": hashlib.sha256(u.encode()).hexdigest()[:16],
                    "parse_error": None, "prompt_tokens": None, "completion_tokens": None, "latency_s": None, "raw_response": None}
            if self.log_prompts:
                call["prompt"] = u
            try:
                r = self.client.generate(SYSTEM_PROMPT, u, temp, self.max_tokens, seed * 100 + attempt)
            except LLMError:
                raise  # provider/transport failure: never turned into an experimental outcome; the runner stops and saves
            call.update(prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens, latency_s=r.latency_s,
                        raw_response=r.text, tokens_estimated=r.tokens_estimated,
                        cost_usd=_call_cost(self.client, r), rate_limit_wait_s=(r.raw or {}).get("rate_limit_wait_s"), request_latency_s=(r.raw or {}).get("request_latency_s", r.latency_s),
                        params_adjusted=(r.raw or {}).get("params_adjusted"),
                        request_temperature=(r.raw or {}).get("request_temperature", temp), response_meta=r.raw)
            try:
                d, notes = normalize_plan_dict(extract_json(r.text), self.normalize_format)
                call["normalized"] = notes
                out.plan = FlightPlan.from_llm_json(d, mission.mission_id)
                out.calls.append(call)
                out.error = None
                return out
            except (ValueError, TypeError, OverflowError, RecursionError) as e:   # JSONDecodeError and SchemaError are ValueErrors
                err = f"{type(e).__name__}: {e}"
                call["parse_error"] = err
                out.calls.append(call)
                out.error = err
        return out

    def initial_plan(self, mission: Mission, seed: int, attempt: int = 0, temperature: float | None = None, nonce: bool = False) -> PlannerOutput:
        return self._call("initial", mission, initial_prompt(mission, self.prompt_mode, attempt if nonce else None), derive_seed(seed, "initial", attempt), temperature)

    def repair_plan(self, mission: Mission, prompt: str, seed: int, iteration: int) -> PlannerOutput:
        return self._call("repair", mission, prompt, derive_seed(seed, "repair", iteration))

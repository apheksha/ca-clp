"""Model capability metadata, cost/latency accounting and availability handling for multi-model studies.

* `get_spec(provider, model)`    -> ModelSpec merged from config/models.yaml (provider defaults < model entry); unknown models are flagged.
* `adapt_request(spec, ...)`     -> provider-specific parameter handling (omit unsupported temperature/seed, clamp max_tokens, enforce min_temperature); every
                                    adjustment is returned so it can be RECORDED in the call log (no silent parameter changes).
* `cost_usd(spec, ptok, ctok)`   -> cost estimate or None when prices are unknown (never 0 for unknown).
* `check_available(spec)`        -> (ok, reason): API key present (and optionally a reachability probe for local servers).
* `resolve_available(...)`       -> first available model among a model and its fallbacks, with the reasons for skipping the others."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import yaml

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "models.yaml")


@dataclass
class ModelSpec:
    provider: str
    model: str
    supports_temperature: bool = True
    supports_seed: bool = False
    min_temperature: float | None = None
    min_max_tokens: int | None = None
    max_output_tokens: int | None = None
    context_window: int | None = None
    usd_per_mtok_in: float | None = None
    usd_per_mtok_out: float | None = None
    rpm: float = 0
    key_env: str | None = None
    reasoning_model: bool = False
    notes: str = ""
    fallbacks: list = field(default_factory=list)
    capabilities_unknown: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def load_registry(path: str | None = None) -> dict:
    with open(path or _PATH) as fh:
        return yaml.safe_load(fh)


def get_spec(provider: str, model: str, path: str | None = None) -> ModelSpec:
    reg = load_registry(path)
    base = dict(reg.get("providers", {}).get(provider, {}))
    entry = next((m for m in reg.get("models", []) if m.get("provider") == provider and m.get("model") == model), None)
    unknown = entry is None
    d = {**base, **{k: v for k, v in (entry or {}).items() if k not in ("provider", "model")}}
    fields = set(ModelSpec.__dataclass_fields__)
    spec = ModelSpec(provider, model, **{k: v for k, v in d.items() if k in fields})
    spec.capabilities_unknown = unknown and provider not in ("mock", "follower", "none")
    return spec


def adapt_request(spec: ModelSpec, temperature: float, max_tokens: int, seed: int | None):
    """Return (temperature|None, max_tokens, seed|None, adjustments). None = omit the parameter from the request."""
    adj = {}
    t, mt, sd = temperature, max_tokens, seed
    if not spec.supports_temperature:
        if t is not None:
            adj["temperature_omitted"] = {"requested": temperature, "reason": "model does not accept a temperature parameter"}
        t = None
    elif spec.min_temperature is not None and t is not None and t < spec.min_temperature:
        adj["temperature_raised"] = {"requested": temperature, "sent": spec.min_temperature}
        t = spec.min_temperature
    if spec.min_max_tokens is not None and mt < spec.min_max_tokens:
        adj["max_tokens_raised"] = {"requested": max_tokens, "sent": spec.min_max_tokens}
        mt = spec.min_max_tokens
    if spec.max_output_tokens is not None and mt > spec.max_output_tokens:
        adj["max_tokens_clamped"] = {"requested": max_tokens, "sent": spec.max_output_tokens}
        mt = spec.max_output_tokens
    if sd is not None and not spec.supports_seed:
        adj["seed_not_sent"] = {"reason": "provider ignores/does not support a seed"}
        sd = None
    return t, mt, sd, adj


def cost_usd(spec: ModelSpec, prompt_tokens, completion_tokens) -> float | None:
    if spec.usd_per_mtok_in is None or spec.usd_per_mtok_out is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (prompt_tokens * spec.usd_per_mtok_in + completion_tokens * spec.usd_per_mtok_out) / 1e6


def check_available(spec: ModelSpec, probe: bool = False) -> tuple[bool, str]:
    if spec.provider in ("mock", "follower", "none"):
        return True, "built-in"
    if spec.key_env and not os.environ.get(spec.key_env):
        return False, f"environment variable {spec.key_env} is not set"
    if probe and spec.provider == "ollama":
        import urllib.request
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        try:
            urllib.request.urlopen(base + "/models", timeout=2)
        except Exception as e:
            return False, f"ollama server not reachable at {base}: {e}"
    return True, "ok"


def resolve_available(provider: str, model: str, probe: bool = False, path: str | None = None):
    """Try `provider:model`, then its configured fallbacks ("provider:model" strings). Returns (spec|None, skipped=[{model, reason}])."""
    skipped, queue, seen = [], [f"{provider}:{model}"], set()
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        p, m = cur.split(":", 1)
        spec = get_spec(p, m, path)
        ok, why = check_available(spec, probe)
        if ok:
            return spec, skipped
        skipped.append({"model": cur, "reason": why})
        queue.extend(spec.fallbacks or [])
    return None, skipped

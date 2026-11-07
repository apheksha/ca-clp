"""Empirical check that the provider really honours `temperature` (reviewer concern: some Gemini endpoints may ignore custom temperatures).

Method: send the SAME initial-planning prompt N times at temperature 0 and N times at temperature 0.7 (fresh calls, seeds varied where the provider accepts a
seed) and count distinct replies. Honoured => diversity(0.7) clearly exceeds diversity(0). Ignored => both look the same (typically both identical or both diverse).
The verdict is a heuristic about ONE prompt: it can show that a parameter has an effect, never that it has none. Results are saved to outputs/tempcheck/.
Without this evidence the resample baseline (D_resample, T = 0.7) must not be described as 'sampling at 0.7'."""
from __future__ import annotations
import hashlib
import json
import os
import time
import yaml
from src.models.schemas import Mission
from src.planner.llm import make_client, LLMError
from src.planner.prompts import SYSTEM_PROMPT, initial_prompt


def _distinct(texts: list[str]) -> int:
    return len({hashlib.sha256(t.strip().encode()).hexdigest() for t in texts})


def analyse(res0: list[dict], res7: list[dict]) -> dict:
    d0, d7 = _distinct([r["text"] for r in res0]), _distinct([r["text"] for r in res7])
    sent0 = sorted({r["request_temperature"] for r in res0})
    sent7 = sorted({r["request_temperature"] for r in res7})
    n = max(len(res0), len(res7))
    if d7 > d0 and d7 >= 2:
        verdict = "HONOURED (reply diversity rises with temperature)"
    elif d0 == d7 == 1:
        verdict = "INCONCLUSIVE-LIKELY-IGNORED (identical replies at both temperatures; do not attribute resample differences to temperature)"
    elif d0 >= 2 and d0 >= d7:
        verdict = "NOT-HONOURED-AT-0 (replies already vary at T=0 and are no more varied at 0.7: temperature has no visible effect)"
    else:
        verdict = "INCONCLUSIVE"
    return {"n_per_temperature": n, "distinct_T0": d0, "distinct_T0.7": d7, "temperature_sent_T0": sent0, "temperature_sent_T0.7": sent7, "verdict": verdict}


def run_tempcheck(provider: str, model: str, n: int = 6, say=print, scenario_path: str = "benchmark/generated/hard_000.json", out_root: str = "outputs/tempcheck", client=None,
                 raise_errors: bool = False):
    cfg = yaml.safe_load(open("config/default.yaml"))
    cfg.update(provider=provider, model=model)
    client = client or make_client(cfg)
    m = Mission.from_dict(json.load(open(scenario_path))["mission"])
    prompt = initial_prompt(m)
    out = {"T0": [], "T0.7": []}
    try:
        for label, temp in (("T0", 0.0), ("T0.7", 0.7)):
            for k in range(n):
                r = client.generate(SYSTEM_PROMPT, prompt, temp, cfg.get("max_tokens", 2000), seed=1000 + k)
                out[label].append({"text": r.text, "request_temperature": (r.raw or {}).get("request_temperature", temp), "response_model": (r.raw or {}).get("response_model"),
                                   "system_fingerprint": (r.raw or {}).get("system_fingerprint")})
    except LLMError as e:
        if raise_errors:          # run_free.py decides whether to wait (quota) or stop
            raise
        say(f"[tempcheck] provider error: {e}")
        return False
    rep = analyse(out["T0"], out["T0.7"])
    rep.update(provider=provider, model=model, prompt_scenario=scenario_path, when=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               response_models=sorted({x["response_model"] for v in out.values() for x in v if x["response_model"]}))
    os.makedirs(out_root, exist_ok=True)
    path = os.path.join(out_root, f"tempcheck_{provider}_{model.replace('/', '_')}_{int(time.time())}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"report": rep, "replies": out}, fh, indent=1)
    say(json.dumps(rep, indent=1))
    say(f"[tempcheck] saved {path}")
    return True

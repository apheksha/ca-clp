"""Experiment runner: scenarios x conditions x seeds -> append-only raw JSONL (+manifest), never overwriting.
Provider failures (HTTP 429, quota, network) are NEVER recorded as experimental outcomes: the run stops cleanly, everything finished is
saved, and `--resume` continues later. Progress is printed continuously."""
from __future__ import annotations
import json
import logging
import os
import time
import yaml
from src.models.schemas import Mission, FlightPlan
from src.benchmark.generator import load_scenarios
from src.benchmark.feasibility import assess
from src.planner.llm import make_client, LLMError, stop_info
from src.planner.planner import LLMPlanner, PlannerOutput
from src.planner.prompts import PROMPT_VERSION
from src.repair.loop import LoopConfig, run_condition, RESAMPLE_TEMPERATURE
from src.evaluation.metrics import run_metrics
from src.utils.io import new_run_id, create_unique_dir, env_manifest, append_jsonl, sha256_file, read_jsonl

log = logging.getLogger("runner")


def load_yaml(p):
    with open(p) as f:
        return yaml.safe_load(f)


def temperature_audit(runs_path: str) -> dict:
    """What was actually SENT per call (not what the config says), so a provider that silently ignores/clamps `temperature` can be spotted.
    Compare with `python run.py tempcheck`, which tests empirically whether sampling diversity changes with temperature."""
    from collections import Counter
    c: Counter = Counter()
    models: Counter = Counter()
    if not os.path.exists(runs_path):
        return {}
    for r in read_jsonl(runs_path):
        for it in r["iterations"]:
            for call in it.get("calls", []):
                c[str(call.get("request_temperature", call.get("temperature")))] += 1
                rm = (call.get("response_meta") or {}).get("response_model")
                if rm:
                    models[rm] += 1
    return {"request_temperature_counts": dict(c), "response_model_counts": dict(models),
            "note": "temperatures as sent by the client; a provider may still ignore them server-side (verify with `run.py tempcheck`)"}


def max_calls_per_scenario_seed(conds) -> int:
    n = 1  # shared initial plan
    for c in conds:
        if c.kind in ("closed_loop", "resample"):
            n += c.max_iterations
    return n


def _say(msg):
    print(msg, flush=True)


def run_experiment(default_cfg_path: str, exp_cfg_path: str, provider: str | None = None, model: str | None = None,
                   raw_root: str = "outputs/raw", scenario_dir: str | list = "benchmark/generated", seeds: list[int] | None = None,
                   limit: int | None = None, difficulties: list[str] | None = None, conditions: list[str] | None = None,
                   max_calls: int | None = None, resume_dir: str | None = None, temperature: float | None = None,
                   progress=_say, estimate_only: bool = False, prompt_mode: str | None = None) -> str | None:
    cfg = load_yaml(default_cfg_path)
    exp = load_yaml(exp_cfg_path)
    if provider:
        cfg["provider"] = provider
    if model:
        cfg["model"] = model
    cfg.update(exp.get("planner", {}))
    if temperature is not None:
        cfg["temperature"] = temperature
    if prompt_mode:
        cfg["prompt_mode"] = prompt_mode
    seeds = seeds if seeds is not None else exp["seeds"]
    dirs = [scenario_dir] if isinstance(scenario_dir, str) else list(scenario_dir)
    scen, scen_files = [], {}
    for d in dirs:
        for s in load_scenarios(d):
            scen.append(s)
            scen_files[s["scenario_id"]] = os.path.join(d, s["scenario_id"] + ".json")
    if not scen:
        raise SystemExit(f"no scenarios found in {dirs} (folder missing or empty); nothing to run. Generate them with `python run.py generate` or restore benchmark/real/.")
    if difficulties:
        scen = [s for s in scen if s["difficulty"] in difficulties]
    if not scen:
        raise SystemExit(f"no scenarios left after filtering difficulties={difficulties}")
    if limit:
        scen = scen[:limit]
    conds = [LoopConfig(name=n, **c) for n, c in exp["conditions"].items() if not conditions or n in conditions]
    per = max_calls_per_scenario_seed(conds)
    llm_conds = [c for c in conds if c.kind not in ("classical", "astar")]
    if cfg["provider"] in ("mock",) and cfg.get("prompt_mode", "spec") != "spec":
        raise ValueError("the mock heuristic reads the structured MISSION_SPEC; prompt_mode nl_only/spec_only needs a real LLM (or the 'follower' dry-run client for spec_only)")
    est = per * len(scen) * len(seeds) if llm_conds else 0
    interval = max(cfg.get("request_interval_s", 0) or 0, 60.0 / cfg["requests_per_minute"] if cfg.get("requests_per_minute") else 0)
    progress(f"[plan] {len(scen)} scenarios x {len(conds)} conditions x {len(seeds)} seeds = {len(scen)*len(conds)*len(seeds)} runs; "
             f"worst case ~{est} LLM calls (typically ~half)." + (f" At >= {interval:.1f}s/call that is up to ~{est*max(interval,1)/60:.0f} min." if cfg["provider"] != "mock" else ""))
    if estimate_only:
        return None
    client = make_client(cfg)
    client.on_wait = lambda w, k, why: progress(f"   ...provider said '{why}', waiting {w:.0f}s (retry {k})")
    planner = LLMPlanner(client, cfg.get("temperature", 0.0), cfg.get("max_tokens", 2000), cfg.get("max_parse_retries", 2), cfg.get("log_prompts", True), cfg.get("normalize_format", True), cfg.get("prompt_mode", "spec"))
    for c_ in conds:
        eff = c_.temperature if c_.temperature is not None else (RESAMPLE_TEMPERATURE if c_.kind == "resample" else None)
        if c_.kind == "resample" and not client.is_mock and eff == 0:
            raise ValueError(f"condition {c_.name}: resample at temperature 0 returns (nearly) identical plans - set temperature > 0")
    if resume_dir:
        out_dir = resume_dir
        old = json.load(open(os.path.join(out_dir, "manifest.json")))
        run_id = old["experiment_run_id"]
        if client.name == "none":
            bad = [c.name for c in conds if c.kind in ("llm_only", "llm_verify", "resample", "closed_loop")]
            if bad:
                raise SystemExit(f"provider 'none' can only add model-free conditions to an existing run; these need a model: {bad}")
            progress(f"[info] model-free session: adding {[c.name for c in conds]} to {old['provider']}/{old['model']} run, re-using its cached first plans (0 API calls)")
        elif old["model"] != client.model or old["provider"] != client.name:
            progress(f"[warn] resuming with provider={client.name} model={client.model} (was {old['provider']}/{old['model']}) — runs are mixed-provider; disclose this in any write-up")
    else:
        run_id = new_run_id("mock" if client.is_mock else "real")
        out_dir = create_unique_dir(raw_root, run_id)
        old = {}
    manifest = {"experiment_run_id": run_id, "is_mock": client.is_mock, "provider": client.name, "model": client.model,
                "planner_config": planner.config(), "prompt_version": PROMPT_VERSION, "prompt_mode": planner.prompt_mode,
                "model_spec": (getattr(client, "spec").to_dict() if getattr(client, "spec", None) is not None else None), "seeds": seeds,
                "conditions": {c.name: c.to_dict() for c in conds}, "default_config": cfg, "experiment_config": exp,
                "n_scenarios": len(scen), "scenario_ids": [s["scenario_id"] for s in scen],
                "scenario_files_sha256": {s["scenario_id"]: sha256_file(scen_files[s["scenario_id"]]) for s in scen},
                "data_sources": sorted({s["metadata"].get("source", "synthetic") for s in scen}),
                "pairs": exp.get("pairs", []), "pricing": cfg.get("pricing"), "started": old.get("started", time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                "resumed": old.get("resumed", 0) + (1 if resume_dir else 0), "env": env_manifest(),
                "WARNING": "MOCK RUN: pipeline test only, NOT evidence about real LLMs" if client.is_mock else None}
    if resume_dir and old:
        # Provenance on resume: keep the environment / code fingerprint of the session that STARTED the run (it produced the earlier records) and log every
        # later session separately; keep every condition already in the file (a resume restricted with --conditions must not hide the others from reports).
        manifest["env"] = old.get("env", manifest["env"])
        manifest["env_resumes"] = old.get("env_resumes", []) + [env_manifest()]
        manifest["conditions"] = {**old.get("conditions", {}), **manifest["conditions"]}
        manifest["sessions"] = old.get("sessions", [{"started": old.get("started"), "conditions": list(old.get("conditions", {})), "provider": old.get("provider"), "model": old.get("model")}]) + \
            [{"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "conditions": [c.name for c in conds], "provider": client.name, "model": client.model}]
        if client.name == "none":
            # a model-free session (e.g. adding G_direct_* controls) re-uses the cached first plans of the original model: the run still belongs to that model
            for k in ("provider", "model", "is_mock", "planner_config", "model_spec", "prompt_mode"):
                if k in old:
                    manifest[k] = old[k]
    if not client.is_mock and (cfg.get("temperature", 0) == 0 and len(seeds) > 1):
        progress("[warn] temperature 0 with several seeds: repeated seeds may be near-identical (not independent). Use temperature>0 for multi-seed designs.")
    out_path = os.path.join(out_dir, "runs.jsonl")
    cache: dict = {}
    done: set = set()
    n = 0
    if resume_dir and os.path.exists(out_path):
        for r in read_jsonl(out_path):
            done.add((r["scenario_id"], r["condition"], r["seed"]))
            n += 1
            if r["condition_config"]["kind"] not in ("classical", "astar"):
                i0 = r["iterations"][0]
                cache.setdefault((r["scenario_id"], r["seed"]), PlannerOutput(FlightPlan.from_dict(i0["plan"]) if i0["plan"] else None, i0["calls"], i0.get("error")))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    progress(f"checking feasibility of {len(scen)} missions with the oracle (no model calls; terrain missions take up to a minute each) ...")
    from src.benchmark.oracle_cache import assess_cached       # v12: verdicts cached on disk (re-verified on load)
    feas_cache = {s["scenario_id"]: assess_cached(Mission.from_dict(s["mission"])) for s in scen}
    calls_used, stop_reason, t_start = 0, None, time.time()
    stop_struct = None
    total_units = len(scen) * len(seeds)
    unit = 0
    try:
        for seed in seeds:
            for s in scen:
                unit += 1
                if all((s["scenario_id"], c.name, seed) in done for c in conds):
                    continue
                if max_calls is not None and calls_used >= max_calls:   # stop between scenarios so conditions stay paired
                    stop_reason = f"call budget reached ({calls_used}/{max_calls})"
                    break
                m = Mission.from_dict(s["mission"])
                key = (s["scenario_id"], seed)
                for c in conds:
                    if (s["scenario_id"], c.name, seed) in done:
                        continue
                    fresh_initial = c.kind not in ("classical", "astar") and key not in cache
                    o = run_condition(m, planner, c, seed, initial=None if c.kind in ("classical", "astar") else cache.get(key), feasibility=feas_cache[s["scenario_id"]])
                    if c.kind not in ("classical", "astar"):
                        cache.setdefault(key, o.initial)  # shared initial plan => paired comparison across conditions
                    rec = {"run_id": f"{run_id}-{n:06d}", "experiment_id": run_id, "scenario_id": s["scenario_id"], "difficulty": s["difficulty"],
                           "scenario_meta": s["metadata"], "condition": c.name, "condition_config": c.to_dict(), "model": client.model,
                           "provider": client.name, "is_mock": client.is_mock, "seed": seed, "prompt_mode": planner.prompt_mode,
                           "initial_plan": o.initial.plan.to_dict() if o.initial.plan else None,
                           "iterations": [it.__dict__ for it in o.iterations],
                           "final_plan": o.final_plan.to_dict() if o.final_plan else None,
                           "final_verification": o.final_verification.to_dict() if o.final_verification else None,
                           "final_status": o.status, "metrics": {"released": o.released, "oracle_inconsistent": o.oracle_inconsistent},
                           "feasibility": o.feasibility, "timing": {"wall_time_s": o.wall_time_s}}
                    rec["metrics"].update(run_metrics(rec, cfg.get("pricing")))
                    append_jsonl(out_path, rec)
                    calls_used += (len(o.initial.calls) if fresh_initial else 0) + sum(len(it.calls) for it in o.iterations[1:])
                    n += 1
                el = time.time() - t_start
                frac = unit / total_units
                progress(f"[{unit}/{total_units}] {s['scenario_id']:<28} runs={n:<5} api_calls={calls_used:<5} elapsed={el/60:5.1f}min "
                         f"eta~{(el/frac - el)/60 if frac else 0:5.1f}min")
            if stop_reason:
                break
    except LLMError as e:
        stop_reason = f"provider error: {e}"
        stop_struct = stop_info(e)
    except KeyboardInterrupt:
        stop_reason = "interrupted by user (Ctrl-C)"
    manifest["temperature_audit"] = temperature_audit(out_path)
    manifest["stop"] = stop_struct or stop_info(None, stop_reason)     # v12: structured (kind = none | budget | interrupted | daily_quota | rate_limit | auth | bad_request | provider_error)
    manifest.update(stopped_reason=stop_reason, stopped_early_by_call_budget=bool(stop_reason and "budget" in stop_reason),
                    api_calls_this_session=calls_used, n_runs=n, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if stop_reason:
        progress(f"\n[stopped] {stop_reason}\n   {n} runs are saved in {out_dir}\n   continue later with:  python run.py experiment --provider {client.name} --model {client.model} --resume {out_dir}")
    return out_dir

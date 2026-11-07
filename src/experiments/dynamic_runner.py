"""Objective 4 runner: controlled mid-flight changes -> repair of the REMAINING route (or from-scratch re-plan baseline).
Same safety rules as the static runner: provider errors are never outcomes; progress is printed; --resume works."""
from __future__ import annotations
import json
import os
import random
import time
from src.models.schemas import Mission
from src.benchmark.generator import load_scenarios
from src.benchmark.feasibility import assess
from src.dynamic.changes import CHANGES, comfortable_plan, execution_state, build_dynamic_mission, build_delayed_dynamic_mission, classify_failure
from src.planner.llm import make_client, LLMError, stop_info
from src.planner.planner import LLMPlanner, PlannerOutput
from src.planner.prompts import PROMPT_VERSION
from src.repair.loop import LoopConfig, run_condition, RESAMPLE_TEMPERATURE
from src.verifier.verifier import verify, resources
from src.evaluation.metrics import run_metrics
from src.experiments.runner import load_yaml, _say
from src.utils.io import new_run_id, create_unique_dir, env_manifest, append_jsonl, read_jsonl


def selection_order(scen: list[dict], dcfg: dict) -> list[dict]:
    """Order in which scenarios are considered for the dynamic study.
    selection: "seeded_random" (default) = per-difficulty shuffle with `selection_seed`, so the n_per_difficulty subset is NOT simply the first files
    (avoids an ordering/selection bias); "first" = file order (legacy behaviour, kept for reproducing older runs)."""
    mode = dcfg.get("selection", "seeded_random")
    if mode == "first":
        return list(scen)
    rng = random.Random(f"dynamic-selection|{dcfg.get('selection_seed', 2024)}")
    by_d: dict[str, list[dict]] = {}
    for s in scen:
        by_d.setdefault(s["difficulty"], []).append(s)
    out = []
    for d in sorted(by_d):
        g = sorted(by_d[d], key=lambda x: x["scenario_id"])
        rng.shuffle(g)
        out.extend(g)
    return out


def build_cases(scen: list[dict], dcfg: dict, info: dict | None = None) -> list[dict]:
    """One case = (scenario, change type, progress at change, detection delay). `info` (optional dict) is filled with selection bookkeeping
    (eligible / selected / skipped scenario ids) so the manifest can document exactly which scenarios entered the study and why."""
    cases, per = [], {}
    delays = dcfg.get("detection_delays", [0.0]) or [0.0]
    elig, selected, no_trigger = [], [], []
    for s in selection_order(scen, dcfg):
        d = s["difficulty"]
        if s["metadata"]["feasibility"] != "FEASIBLE":
            continue
        elig.append(s["scenario_id"])
        if per.get(d, 0) >= dcfg.get("n_per_difficulty", 5):
            continue
        m = Mission.from_dict(s["mission"])
        plan = comfortable_plan(m)
        if plan is None:
            no_trigger.append(s["scenario_id"])
            continue
        added = 0
        for ch in dcfg.get("changes", CHANGES):
            for pr in dcfg.get("progress", [0.35, 0.65]):
                for dl in delays:
                    rng = random.Random(f"{s['scenario_id']}|{ch}|{pr}")
                    st = execution_state(m, plan, pr)
                    if dl and dl > 0:
                        st_d = execution_state(m, plan, min(pr + dl, 0.98))
                        r = build_delayed_dynamic_mission(m, plan, st, st_d, ch, rng)
                        st_used = st_d
                    else:
                        r = build_dynamic_mission(m, plan, st, ch, rng)
                        st_used = st
                    if r is None:
                        continue
                    new, rem, inf = r
                    if dcfg.get("require_triggered", True) and verify(new, rem).success:
                        continue                      # the change did not invalidate the stale plan: tests nothing, so it is not a case
                    feas = assess(new)
                    tag = "" if not dl else f"__d{int(round(dl * 100))}"
                    cases.append({"case_id": f"{s['scenario_id']}__{ch}__{int(pr*100)}{tag}", "scenario_id": s["scenario_id"], "difficulty": d, "change": ch,
                                  "progress": pr, "detection_delay": float(dl or 0.0), "mission": new, "remaining_plan": rem, "state": st_used, "info": inf, "feas": feas,
                                  "orig_mission": m, "orig_plan": plan, "source": s["metadata"].get("source", "synthetic")})
                    added += 1
        per[d] = per.get(d, 0) + (1 if added else 0)
        (selected if added else no_trigger).append(s["scenario_id"])
    if info is not None:
        info.update(mode=dcfg.get("selection", "seeded_random"), seed=dcfg.get("selection_seed", 2024), n_eligible=len(elig), eligible=elig, selected=selected,
                    considered_but_no_triggered_case=no_trigger, n_per_difficulty=dcfg.get("n_per_difficulty", 5), detection_delays=list(delays),
                    n_cases=len(cases))
    return cases


def run_dynamic_experiment(default_cfg_path: str, exp_cfg_path: str, provider: str | None = None, model: str | None = None, raw_root: str = "outputs/raw",
                           scenario_dir=("benchmark/generated",), seeds: list[int] | None = None, conditions: list[str] | None = None, max_calls: int | None = None,
                           resume_dir: str | None = None, temperature: float | None = None, progress=_say, limit_cases: int | None = None,
                           prompt_mode: str | None = None) -> str:
    cfg, exp = load_yaml(default_cfg_path), load_yaml(exp_cfg_path)
    if provider:
        cfg["provider"] = provider
    if model:
        cfg["model"] = model
    cfg.update(exp.get("planner", {}))
    if temperature is not None:
        cfg["temperature"] = temperature
    if prompt_mode:
        cfg["prompt_mode"] = prompt_mode
    dcfg = exp.get("dynamic", {})
    seeds = seeds if seeds is not None else dcfg.get("seeds", [0])
    scen = [s for d in ([scenario_dir] if isinstance(scenario_dir, str) else scenario_dir) for s in load_scenarios(d)]
    if not scen:
        raise SystemExit(f"no scenarios found in {scenario_dir} (folder missing or empty); nothing to run.")
    sel_info: dict = {}
    cases = build_cases(scen, dcfg, sel_info)
    if limit_cases:
        cases = cases[:limit_cases]
    names = conditions or dcfg.get("conditions") or list(exp["conditions"])
    conds = [LoopConfig(name=n, **exp["conditions"][n]) for n in names]
    progress(f"[plan] {len(cases)} dynamic cases x {len(conds)} conditions x {len(seeds)} seeds = {len(cases)*len(conds)*len(seeds)} runs")
    client = make_client(cfg)
    client.on_wait = lambda w, k, why: progress(f"   ...provider said '{why}', waiting {w:.0f}s (retry {k})")
    planner = LLMPlanner(client, cfg.get("temperature", 0.0), cfg.get("max_tokens", 2000), cfg.get("max_parse_retries", 2), cfg.get("log_prompts", True), cfg.get("normalize_format", True), cfg.get("prompt_mode", "spec"))
    for c_ in conds:
        eff = c_.temperature if c_.temperature is not None else (RESAMPLE_TEMPERATURE if c_.kind == "resample" else None)
        if c_.kind == "resample" and not client.is_mock and eff == 0:
            raise ValueError(f"condition {c_.name}: resample at temperature 0 returns (nearly) identical plans - set temperature > 0")
    if resume_dir:
        out_dir = resume_dir
        run_id = json.load(open(os.path.join(out_dir, "manifest.json")))["experiment_run_id"]
    else:
        run_id = new_run_id("dyn-mock" if client.is_mock else "dyn-real")
        out_dir = create_unique_dir(raw_root, run_id)
    manifest = {"experiment_run_id": run_id, "kind": "dynamic", "is_mock": client.is_mock, "provider": client.name, "model": client.model,
                "planner_config": planner.config(), "prompt_version": PROMPT_VERSION, "prompt_mode": planner.prompt_mode, "seeds": seeds, "conditions": {c.name: c.to_dict() for c in conds},
                "default_config": cfg, "experiment_config": exp, "n_cases": len(cases), "case_selection": sel_info, "pairs": exp.get("dynamic_pairs", []), "pricing": cfg.get("pricing"),
                "started": (json.load(open(os.path.join(out_dir, "manifest.json"))).get("started") if resume_dir and os.path.exists(os.path.join(out_dir, "manifest.json")) else time.strftime("%Y-%m-%dT%H:%M:%S%z")), "env": env_manifest(),
                "WARNING": "MOCK RUN: pipeline test only, NOT evidence about real LLMs" if client.is_mock else None}
    if resume_dir and os.path.exists(os.path.join(out_dir, "manifest.json")):
        old = json.load(open(os.path.join(out_dir, "manifest.json")))     # provenance: keep the starting session's env, log later sessions (see runner.py)
        manifest["env"] = old.get("env", manifest["env"])
        manifest["env_resumes"] = old.get("env_resumes", []) + [env_manifest()]
        manifest["conditions"] = {**old.get("conditions", {}), **manifest["conditions"]}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    path = os.path.join(out_dir, "runs.jsonl")
    done = {(r["scenario_id"], r["condition"], r["seed"]) for r in read_jsonl(path)} if (resume_dir and os.path.exists(path)) else set()
    n, calls_used, stop, t0 = len(done), 0, None, time.time()
    stop_struct = None
    try:
        for seed in seeds:
            for ci, c in enumerate(cases):
                if all((c["case_id"], k.name, seed) in done for k in conds):
                    continue
                if max_calls is not None and calls_used >= max_calls:
                    stop = f"call budget reached ({calls_used}/{max_calls})"
                    break
                m, rem = c["mission"], c["remaining_plan"]
                init = PlannerOutput(rem, [], None)
                v0 = verify(m, rem)
                meta = {"feasibility": c["feas"].status, "witness_distance": c["feas"].witness_distance,
                        "witness_energy": resources(c["feas"].witness_plan, m)["energy"] if c["feas"].witness_plan else None,
                        "source": c["source"], "change": c["change"], "orig_scenario": c["scenario_id"]}
                for k in conds:
                    if (c["case_id"], k.name, seed) in done:
                        continue
                    o = run_condition(m, planner, k, seed, initial=None if k.kind in ("classical", "astar") else init, feasibility=c["feas"])
                    rec = {"run_id": f"{run_id}-{n:06d}", "experiment_id": run_id, "scenario_id": c["case_id"], "difficulty": c["difficulty"], "scenario_meta": meta,
                           "condition": k.name, "condition_config": k.to_dict(), "model": client.model, "provider": client.name, "is_mock": client.is_mock, "seed": seed,
                           "initial_plan": rem.to_dict(), "iterations": [it.__dict__ for it in o.iterations], "final_plan": o.final_plan.to_dict() if o.final_plan else None,
                           "final_verification": o.final_verification.to_dict() if o.final_verification else None, "final_status": o.status,
                           "metrics": {"released": o.released, "oracle_inconsistent": o.oracle_inconsistent}, "feasibility": o.feasibility,
                           "timing": {"wall_time_s": o.wall_time_s},
                           "dynamic": {"change": c["change"], "progress": c["progress"], "detection_delay": c.get("detection_delay", 0.0), "exposure": c["info"].get("exposure"), "orig_scenario": c["scenario_id"], "info": c["info"], "state": c["state"].to_dict(),
                                       "affected": not v0.success, "remaining_waypoints": len(c["state"].remaining_waypoints),
                                       "remaining_required": len(m.required_waypoints), "description": m.description, "mission": m.to_dict(),
                                       "original_mission": c["orig_mission"].to_dict(), "original_plan": c["orig_plan"].to_dict()}}
                    rec["dynamic"]["failure_cause"] = classify_failure(rec)
                    rec["metrics"].update(run_metrics(rec, cfg.get("pricing")))
                    append_jsonl(path, rec)
                    calls_used += sum(len(it.calls) for it in o.iterations[1:])
                    n += 1
                el = time.time() - t0
                progress(f"[{ci+1}/{len(cases)}] {c['case_id']:<44} runs={n:<5} api_calls={calls_used:<5} elapsed={el/60:5.1f}min")
            if stop:
                break
    except LLMError as e:
        stop = f"provider error: {e}"
        stop_struct = stop_info(e)
    except KeyboardInterrupt:
        stop = "interrupted by user (Ctrl-C)"
    expected = {(c["case_id"], k.name, sd) for c in cases for k in conds for sd in seeds}
    got = {(r["scenario_id"], r["condition"], r["seed"]) for r in read_jsonl(path)} if os.path.exists(path) else set()
    missing = sorted(expected - got)
    manifest["stop"] = stop_struct or stop_info(None, stop)
    manifest.update(stopped_reason=stop, api_calls_this_session=calls_used, n_runs=n, finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    completeness={"expected_runs": len(expected), "completed_runs": len(expected & got), "missing_runs": len(missing),
                                  "complete": not missing, "termination_reason": stop or ("completed" if not missing else "ended with missing runs (see missing_sample)"),
                                  "completed_per_condition": {k.name: sum(1 for x in got if x[1] == k.name) for k in conds},
                                  "missing_sample": [list(x) for x in missing[:25]]})
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    if stop:
        progress(f"\n[stopped] {stop}\n   {n} runs saved in {out_dir}\n   continue with:  python run.py dynamic --provider {client.name} --model {client.model} --resume {out_dir}")
    return out_dir

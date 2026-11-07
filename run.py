#!/usr/bin/env python
"""CLI. Examples:
  python run.py generate
  python run.py demo
  python run.py experiment --provider mock
  python run.py evaluate outputs/raw/<run_id>
  python run.py reproduce            # generate + experiment + evaluate (mock unless --provider given)
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yaml


from src.utils.dotenv import load_dotenv   # v12: shared, BOM/quote tolerant


load_dotenv()
from src.benchmark.generator import generate_benchmark
from src.benchmark.demos import run_demos
from src.experiments.runner import run_experiment
from src.evaluation.report import evaluate_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
D, E = "config/default.yaml", "config/experiments.yaml"


def detect_provider(model: str, default_provider: str = "mock") -> str:
    """Best-effort provider guess from a model id; pass --provider explicitly whenever the guess could be wrong."""
    if not model:
        return default_provider
    m = model.lower()
    if m.startswith("mock"):
        return "mock"
    if "gemini" in m:
        return "gemini"
    if "gpt" in m or m.startswith(("o1", "o3", "o4")):
        return "openai"
    if "claude" in m:
        return "anthropic"
    if any(k in m for k in ("llama", "mixtral", "gemma")):
        return "groq"
    if "command" in m:
        return "cohere"
    if "grok" in m:
        return "grok"
    return default_provider

def run_folder(model: str, dataset: str, *, reserve: bool = False, resume: str | None = None) -> str:
    """
    Return a path like:
        outputs/run/<model>_<dataset>_<timestamp>
    If `reserve` is True we add a timestamp; if we are resuming we reuse the
    folder that already exists.
    """
    from datetime import datetime
    import os

    # base name without the timestamp – useful for reproducibility
    base = f"run_{model}_{dataset}".replace("-", "_").replace(".", "_")

    if reserve or resume:
        # use the current time so each fresh run gets a unique folder
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        name = f"{base}_{ts}"
    else:
        # when the user just asks for “help” we give a deterministic name
        name = base

    # final path: outputs/run/<name>
    out_dir = os.path.join("outputs", "run", name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def gen():
    with open(D, encoding="utf-8") as f:
        b = yaml.safe_load(f)["benchmark"]
    s = generate_benchmark("benchmark/generated", "benchmark/metadata", b["per_difficulty"], b["n_infeasible"], b["base_seed"], b.get("n_corridor", 10), b.get("n_per_combo", 3), "benchmark/maps", b.get("n_vertical", 0))
    print(f"generated {len(s)} scenarios -> benchmark/generated")

def find_run_root(path):
    """Walk up from a raw/processed path to the actual run folder (the one holding state.json)."""
    if not path:
        return None
    p = os.path.abspath(path)
    cur = p
    while True:
        if os.path.isfile(os.path.join(cur, "state.json")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    parts = p.split(os.sep)
    if "raw" in parts:
        return os.sep.join(parts[:parts.index("raw")])
    return p

def evaluation_targets(run_dir):
    """Where `run.py evaluate <run_dir>` writes. Returns (processed_root, figures_root, subfolder).
    * run inside a `final` run folder (an ancestor holds state.json): <that folder>/processed and /figures, no per-run subfolder (unchanged behaviour);
    * standard layout <root>/raw/<run_id>: <root>/processed/<run_id> and <root>/figures/<run_id>.
    Previously the second case also wrote straight into <root>/processed/ (no subfolder), so evaluating a second run silently overwrote the CSVs of the first."""
    p = os.path.abspath(run_dir)
    cur = p
    while True:
        if os.path.isfile(os.path.join(cur, "state.json")):
            return os.path.join(cur, "processed"), os.path.join(cur, "figures"), False
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    parent = os.path.dirname(p)
    root = os.path.dirname(parent) if os.path.basename(parent) == "raw" else parent
    return os.path.join(root, "processed"), os.path.join(root, "figures"), True


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("generate")
    sp.add_parser("demo")
    for n in ("experiment", "reproduce"):
        p = sp.add_parser(n)
        p.add_argument("--provider")
        p.add_argument("--model")
        p.add_argument("--seeds", type=int, nargs="*")
        p.add_argument("--limit", type=int)
        p.add_argument("--difficulties", nargs="*")
        p.add_argument("--conditions", nargs="*")
        p.add_argument("--max-calls", type=int, help="stop (cleanly, between scenarios) after this many API calls; protects free-tier daily quotas")
        p.add_argument("--resume", help="raw run dir to continue (same provider/model); finished runs are skipped")
        p.add_argument("--exp-config", default=E)
        p.add_argument("--scenario-dir", nargs="*", default=["benchmark/generated"], help="one or more scenario folders (e.g. benchmark/generated benchmark/real)")
        p.add_argument("--temperature", type=float)
        p.add_argument("--estimate", action="store_true", help="only print the planned number of runs/API calls")
        p.add_argument("--prompt-mode", choices=["spec", "nl_only", "spec_only"], help="spec (paper default) | nl_only (free-text mission only: tests constraint extraction) | spec_only")
        p.add_argument("--models", nargs="*", help="multi-model study: provider:model pairs (e.g. gemini:gemini-3.5-flash-lite groq:llama-3.3-70b-versatile); one raw run dir per model")
    dy = sp.add_parser("dynamic", help="Objective 4: mid-flight changes + replanning (mock unless --provider given)")
    for a_, kw in (("--provider", {}), ("--model", {}), ("--max-calls", {"type": int}), ("--resume", {}), ("--temperature", {"type": float}), ("--limit-cases", {"type": int})):
        dy.add_argument(a_, **kw)
    dy.add_argument("--prompt-mode", choices=["spec", "nl_only", "spec_only"])
    dy.add_argument("--seeds", type=int, nargs="*")
    dy.add_argument("--conditions", nargs="*")
    dy.add_argument("--exp-config", default=E)
    dy.add_argument("--scenario-dir", nargs="*", default=["benchmark/generated"])
    lv = sp.add_parser("llm-value", help="paired analysis: LLM repair vs LLM-free direct execution of the verifier's detours/route")
    lv.add_argument("run_dir")
    lv.add_argument("--out")
    tc = sp.add_parser("tempcheck", help="empirically test whether the provider honours `temperature` (diversity at T=0 vs T=0.7)")
    tc.add_argument("--provider", required=True)
    tc.add_argument("--model", required=True)
    tc.add_argument("--n", type=int, default=6)
    cm = sp.add_parser("compare-models", help="tabulate several raw run dirs (different models / prompt modes) side by side")
    cm.add_argument("run_dirs", nargs="+")
    cm.add_argument("--out", default="outputs/processed/model_comparison")
    dm = sp.add_parser("dynamics", help="post-hoc flight-dynamics feasibility check (turn radius, climb rate, endurance, fillet-vs-zone) of released plans in a raw run dir")
    dm.add_argument("run_dir")
    dm.add_argument("--unit-m", type=float, default=None, help="metres per map unit (default: 1 for synthetic, 1 for real [real maps are in metres])")
    dm.add_argument("--min-turn-radius", type=float, default=30.0)
    dm.add_argument("--max-climb-rate", type=float, default=3.0)
    dm.add_argument("--cruise-speed", type=float, default=12.0)
    dm.add_argument("--max-flight-time-s", type=float, default=None)
    gs = sp.add_parser("geoscenario", help="build a scenario from geodata (airspace GeoJSON, OSM buildings, SRTM terrain) per a YAML config")
    gs.add_argument("--config", required=True)
    gs.add_argument("--out", default="benchmark/geo")
    sm = sp.add_parser("simulate", help="fly the released plans of a raw run in the built-in simulator and evaluate the executed trajectories")
    sm.add_argument("run_dir")
    sm.add_argument("--vehicle", default="generic_quad_large")
    sm.add_argument("--unit-m", type=float, default=None, help="metres per map unit (default: 1 for real scenarios, 20 for synthetic)")
    sm.add_argument("--wind", type=float, nargs=2, default=[0.0, 0.0])
    sm.add_argument("--gust", type=float, default=1.0)
    sm.add_argument("--seeds", type=int, nargs="*", default=[0])
    sm.add_argument("--conditions", nargs="*")
    sm.add_argument("--limit", type=int)
    sm.add_argument("--run-seeds", type=int, nargs="*", help="fly only the plans of these experiment seeds (default: all)")
    sm.add_argument("--out", help="output folder (default: <run_dir>/simulation)")
    sm.add_argument("--altitude-margin", type=float, default=0.0, help="fly each plan with waypoint altitudes lifted into [min+d, max-d] (re-verified first; default 0 = as released)")
    px = sp.add_parser("px4", help="PX4 SITL bridge: export world + mission (dry-run) or fly a plan in a running PX4/Gazebo (needs pymavlink; NOT tested live here)")
    px.add_argument("--scenario", required=True)
    px.add_argument("--plan", required=True, help="JSON file with a FlightPlan dict")
    px.add_argument("--origin", type=float, nargs=2, required=True, metavar=("LAT", "LON"))
    px.add_argument("--unit-m", type=float, default=1.0)
    px.add_argument("--connection", default="udp:127.0.0.1:14540")
    px.add_argument("--dry-run", action="store_true")
    px.add_argument("--out", default="outputs/px4")
    on = sp.add_parser("online", help="event-detection + online re-planning experiment in the simulator (events sensed, not injected)")
    on.add_argument("--scenario-dir", default="benchmark/generated")
    on.add_argument("--n", type=int, default=10)
    on.add_argument("--vehicle", default="generic_quad_large")
    on.add_argument("--unit-m", type=float, default=20.0)
    on.add_argument("--replanners", nargs="*", default=["none", "classical"], help="none | classical | llm (llm needs --provider/--model and a key)")
    on.add_argument("--provider", default="mock")
    on.add_argument("--model", default="mock-heuristic-v1")
    on.add_argument("--out", default="outputs/online")
    on.add_argument("--clearance-m", type=float, default=None, help="extra clearance around zones for classical re-plans (execution robustness)")
    ed = sp.add_parser("evaluate-dynamic")
    ed.add_argument("run_dir")
    pf = sp.add_parser("preflight", help="check config, estimate calls, make ONE smoke call (real providers) before a long run")
    pf.add_argument("--model", required=True)
    pf.add_argument("--provider")
    pf.add_argument("--profile", default="reduced")
    pf.add_argument("--no-smoke", action="store_true")

    # Command: final (and alias: run)
    for cmd_name in ("final", "run"):
        fn = sp.add_parser(cmd_name, help="preflight + static + dynamic + evaluation + merged report in dedicated run folder; supports resume & new run")
        fn.add_argument("--model", required=True, help="model id (e.g. gemini-2.5-flash or mock-heuristic-v1)")
        fn.add_argument("--provider", help="provider (gemini, groq, openai, mock, etc.). Auto-detected from model if omitted.")
        fn.add_argument("--dataset", "--profile", dest="profile", default="reduced", choices=["reduced", "full", "pilot", "reduced-fresh"],
                        help="dataset profile: 'reduced' (34 scenarios) or 'full' (118 scenarios)")
        fn.add_argument("--resume", nargs="?", const=True, default=False,
                        help="resume execution: if path omitted, resumes latest run folder for this model+dataset; or pass folder path")
        fn.add_argument("--new-run", "--fresh", action="store_true", dest="fresh",
                        help="start a NEW uniquely-named run folder with timestamp, even if a previous run is incomplete")
        fn.add_argument("--run-dir", help="explicit run folder to use (overrides automatic naming)")
        fn.add_argument("--max-calls", type=int, help="API-call budget for THIS session (static+dynamic together); re-run to continue")
        fn.add_argument("--seeds", type=int, nargs="*")
        fn.add_argument("--auto-retries", type=int, default=3)
        fn.add_argument("--retry-wait-min", type=float, default=10.0)
        fn.add_argument("--allow-code-change", action="store_true")
        fn.add_argument("--skip-smoke", action="store_true")
        fn.add_argument("--resume-raw", help="reuse/continue an existing static raw run dir")
        fn.add_argument("--resume-dynamic-raw", help="reuse/continue an existing dynamic raw run dir")

    rp_ = sp.add_parser("report", help="rebuild the merged report from an outputs/runs/<...> or outputs/final/<...> folder")
    rp_.add_argument("state_dir")
    r = sp.add_parser("real", help="download the OurAirports dataset and build airport-anchored scenarios (real coordinates, MODELLED zones; not legal airspace) (needs internet)")
    r.add_argument("--csv", help="use an already downloaded airports.csv")
    e = sp.add_parser("evaluate")
    e.add_argument("run_dir")
    a = ap.parse_args()
    if a.cmd == "generate":
        gen()
    elif a.cmd == "demo":
        print(run_demos("outputs/processed/demo"))
    elif a.cmd in ("experiment", "reproduce"):
        if a.cmd == "reproduce":
            gen()
        targets = [tuple(x.split(":", 1)) for x in a.models] if getattr(a, "models", None) else [(a.provider, a.model)]
        from src.planner.model_registry import resolve_available
        skipped_all = []
        for prov, mod in targets:
            spec, skipped = resolve_available(prov, mod)          # unavailable model (no key, ...) -> configured fallbacks, else skip WITH the reason recorded
            skipped_all += skipped
            if spec is None:
                print(f"[skip] {prov}:{mod} unavailable: {skipped[-1]['reason'] if skipped else '?'}")
                continue
            if (spec.provider, spec.model) != (prov, mod):
                print(f"[fallback] {prov}:{mod} -> {spec.provider}:{spec.model}")
                prov, mod = spec.provider, spec.model
            if len(targets) > 1:
                print(f"\n===== model {prov}:{mod} =====")
            d = run_experiment(D, a.exp_config, prov, mod, seeds=a.seeds, limit=a.limit, difficulties=a.difficulties, conditions=a.conditions, max_calls=a.max_calls, resume_dir=a.resume, scenario_dir=a.scenario_dir, temperature=a.temperature, estimate_only=a.estimate, prompt_mode=a.prompt_mode)
            if d:
                print("raw data:", d)
                if os.path.exists(os.path.join(d, "runs.jsonl")):
                    print("processed:", evaluate_run(d))
    elif a.cmd == "dynamic":
        from src.experiments.dynamic_runner import run_dynamic_experiment
        from src.evaluation.dynamic_report import evaluate_dynamic
        d = run_dynamic_experiment(D, a.exp_config, a.provider, a.model, scenario_dir=a.scenario_dir, seeds=a.seeds, conditions=a.conditions,
                                   max_calls=a.max_calls, resume_dir=a.resume, temperature=a.temperature, limit_cases=a.limit_cases, prompt_mode=a.prompt_mode)
        print("raw data:", d)
        if os.path.exists(os.path.join(d, "runs.jsonl")):
            print("processed:", evaluate_dynamic(d))
    elif a.cmd == "geoscenario":
        from src.geodata.scenario_builder import build_geo_mission, mission_to_scenario
        cfg = yaml.safe_load(open(a.config))
        m, f = build_geo_mission(cfg, os.path.dirname(os.path.abspath(a.config)))
        os.makedirs(a.out, exist_ok=True)
        sc = mission_to_scenario(m, f, cfg.get("mission_id", "geo"))
        path = os.path.join(a.out, sc["scenario_id"] + ".json")
        json.dump(sc, open(path, "w"))
        print(f"{path}: oracle {f.status} ({f.reason}); {len(m.no_fly_zones)} zones; terrain={'yes' if m.terrain else 'no'}")
    elif a.cmd == "simulate":
        from src.simulator.batch import simulate_run
        print("execution report:", simulate_run(a.run_dir, a.vehicle, a.unit_m, tuple(a.wind), a.gust, tuple(a.seeds), a.conditions, a.limit, a.out, run_seeds=a.run_seeds, altitude_margin_m=a.altitude_margin))
    elif a.cmd == "px4":
        from src.models.schemas import Mission, FlightPlan
        from src.simulator.px4_sitl import run_px4_mission, save_log_csv, log_to_trajectory
        from src.simulator.export_px4 import save_qgc_plan
        from src.simulator.gazebo_world import write_world
        from src.simulator.evaluation import evaluate_execution
        d = json.load(open(a.scenario))
        m = Mission.from_dict(d["mission"] if "mission" in d else d)
        plan = FlightPlan.from_dict(json.load(open(a.plan)))
        os.makedirs(a.out, exist_ok=True)
        print("world:", write_world(m, os.path.join(a.out, "mission_world.sdf"), origin_lat=a.origin[0], origin_lon=a.origin[1]))
        print("QGC plan:", save_qgc_plan(os.path.join(a.out, "mission.plan"), m, plan, a.origin[0], a.origin[1], a.unit_m))
        rows = run_px4_mission(m, plan, a.origin[0], a.origin[1], a.unit_m, a.connection, dry_run=a.dry_run)
        if rows:
            print("log:", save_log_csv(rows, os.path.join(a.out, "px4_flight_log.csv")))
            ev = evaluate_execution(m, plan, log_to_trajectory(rows, m, a.origin[0], a.origin[1]))
            print(json.dumps({k: ev[k] for k in ("executed_feasible", "zone_intrusions", "arrived", "tracking")}, default=str))
    elif a.cmd == "online":
        from src.dynamic.online_experiment import run_online_experiment
        from src.dynamic.online import classical_replanner, loop_replanner
        rps = {}
        for r in a.replanners:
            if r == "none":
                rps["none"] = None
            elif r == "classical":
                rps["classical"] = classical_replanner(a.clearance_m)
            elif r == "llm":
                from src.planner.llm import make_client
                from src.planner.planner import LLMPlanner
                from src.repair.loop import LoopConfig
                cfg = yaml.safe_load(open("config/default.yaml"))
                cfg.update(provider=a.provider, model=a.model)
                rps["llm_C_full"] = loop_replanner(LLMPlanner(make_client(cfg), 0.0, cfg.get("max_tokens", 2000)), LoopConfig(name="C_full", kind="closed_loop", feedback_level="evidence", repair_mode="targeted"))
        print("online experiment:", run_online_experiment(a.scenario_dir, a.out, rps, a.n, a.vehicle, a.unit_m))
    elif a.cmd == "llm-value":
        from src.evaluation.llm_value import llm_value_analysis
        r = llm_value_analysis(a.run_dir, a.out)
        print("llm-value:", r or "needs C_* and G_direct_* conditions in the run")
    elif a.cmd == "tempcheck":
        from src.planner.temperature_check import run_tempcheck
        sys.exit(0 if run_tempcheck(a.provider, a.model, a.n) else 1)
    elif a.cmd == "compare-models":
        from src.evaluation.model_compare import compare_runs
        print("comparison:", compare_runs(a.run_dirs, a.out))
    elif a.cmd == "dynamics":
        from src.verifier.dynamics import evaluate_run_dynamics, VehicleLimits
        lim = VehicleLimits(cruise_speed=a.cruise_speed, max_climb_rate=a.max_climb_rate, min_turn_radius=a.min_turn_radius, max_flight_time_s=a.max_flight_time_s)
        print("dynamics report:", evaluate_run_dynamics(a.run_dir, lim, unit_m=a.unit_m or 1.0))
    elif a.cmd == "evaluate-dynamic":
        from src.evaluation.dynamic_report import evaluate_dynamic
        print("processed:", evaluate_dynamic(a.run_dir))
    elif a.cmd == "preflight":
        from src.pipeline.final import preflight
        provider = a.provider or detect_provider(a.model)
        sys.exit(0 if preflight(provider, a.model, a.profile, smoke=not a.no_smoke) else 1)
    elif a.cmd in ("final", "run"):
        from src.pipeline.final import run_final
        provider = a.provider or detect_provider(a.model)

        # ----- resolve the run folder: raw/processed/figures + report all live in it
        run_dir = find_run_root(a.run_dir) if a.run_dir else None

        run_final(provider, a.model, a.profile, a.max_calls, a.seeds,
                a.auto_retries, a.retry_wait_min, a.allow_code_change,
                a.skip_smoke,
                resume_raw=a.resume_raw, resume_dynamic_raw=a.resume_dynamic_raw,
                fresh=a.fresh, resume=a.resume,
                run_dir=run_dir)
            
    elif a.cmd == "report":
        from src.evaluation.final_report import build_final_report
        print("report:", build_final_report(a.state_dir))
    elif a.cmd == "real":
        from src.benchmark.real_airspace import fetch, build_real_scenarios
        path = a.csv or fetch()
        sc = build_real_scenarios(path, "benchmark/real", "benchmark/metadata")
        print(f"built {len(sc)} airport-anchored scenarios (modelled zones) in benchmark/real")
    elif a.cmd == "evaluate":
        processed_root, figures_root, sub = evaluation_targets(a.run_dir)
        print("processed:", evaluate_run(a.run_dir, processed_root=processed_root, figures_root=figures_root, subfolder=sub))
            
if __name__ == "__main__":
    main()

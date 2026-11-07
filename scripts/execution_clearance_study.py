#!/usr/bin/env python
"""Execution-aware clearance study (v12, no model calls).

Shortest-path witnesses hug zone boundaries (clearance 0.5 map units = 10 m at 20 m/unit on the synthetic map), and a flown vehicle cuts corners and drifts. This script
re-plans every feasible mission with the visibility-graph planner at larger clearances, optionally lifts altitudes into the band by a margin, RE-VERIFIES the plan against the
unchanged mission (budgets still apply), and flies it in the built-in simulator with the same settings as the main study.

  python scripts/execution_clearance_study.py --scenario-dir benchmark/generated --unit-m 20 --vehicle generic_quad_large --clearances 0 1 2 --altitude-margin 5
Clearance 0 means the planner default. Output: <out>/clearance_executions.csv and clearance_summary.csv."""
import argparse, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.benchmark.generator import load_scenarios
from src.benchmark.feasibility import assess
from src.models.schemas import Mission
from src.simulator.batch import prepare_mission
from src.simulator.sim import simulate, SimConfig
from src.simulator.study import execution_record, planned_polyline_m, lift_altitudes
from src.models.vehicle import get_vehicle
from src.verifier.verifier import verify
from src.evaluation.report import wilson

ap = argparse.ArgumentParser()
ap.add_argument("--scenario-dir", default="benchmark/generated")
ap.add_argument("--unit-m", type=float, default=20.0)
ap.add_argument("--vehicle", default="generic_quad_large")
ap.add_argument("--clearances", type=float, nargs="*", default=[0.0, 1.0, 2.0], help="edge clearance in map units (0 = the original witness)")
ap.add_argument("--altitude-margin", type=float, default=5.0)
ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
ap.add_argument("--gust", type=float, default=1.0)
ap.add_argument("--out", default="outputs/review_v12/simulation/clearance_study")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
winds = {"calm": (0.0, 0.0), "crosswind": (0.0, 4.0)}
veh = get_vehicle(a.vehicle)
req_tol = max(1.5 * veh.acceptance_radius_m, 5.0)
rows = []
scen = [s for s in load_scenarios(a.scenario_dir) if s["metadata"]["feasibility"] == "FEASIBLE"]
for s in scen:
    m = Mission.from_dict(s["mission"])
    for c in a.clearances:
        f = assess(m, edge_clearance=c)          # c = 0: the original oracle witness; c > 0: every leg keeps c map units from every zone
        base = {"scenario_id": s["scenario_id"], "clearance_units": c, "clearance_m": c * a.unit_m, "altitude_margin_m": a.altitude_margin}
        if f.status != "FEASIBLE":
            rows.append({**base, "admissible": False, "reason": f"planner: {f.status}"})
            continue
        plan = f.witness_plan
        if a.altitude_margin > 0:
            plan = lift_altitudes(m, plan, a.altitude_margin)
        if not verify(m, plan).success:
            rows.append({**base, "admissible": False, "reason": "lifted plan violates a budget"})
            continue
        for wname, w in winds.items():
            mm = prepare_mission(m, a.vehicle, a.unit_m, w)
            L = sum(math.dist(p[:2], q[:2]) for p, q in zip(planned_polyline_m(mm, plan), planned_polyline_m(mm, plan)[1:]))
            dt = max(0.1, L / 12.0 / 20000.0)
            for sd in a.seeds:
                tr = simulate(mm, plan, cfg=SimConfig(dt=dt, gust_sigma=a.gust, seed=sd, record_every=max(1, int(1.0 / dt))))
                er = execution_record(mm, plan, tr, required_tol_m=req_tol)
                rows.append({**base, "admissible": True, "reason": "", "wind": wname, "sim_seed": sd, **er})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "clearance_executions.csv"), index=False)
summ = []
for c, g in df.groupby("clearance_units"):
    fl = g[g.admissible == True]  # noqa: E712
    n_m = g.scenario_id.nunique()
    adm = g.groupby("scenario_id").admissible.max().sum()
    k_l2, n = int(fl.L2_executed_feasible.sum()) if len(fl) else 0, len(fl)
    near = ((fl.intrusion_zones > 0) | fl.collision) if len(fl) else pd.Series(dtype=bool)
    lo, hi = wilson(k_l2, n) if n else (float("nan"), float("nan"))
    summ.append({"clearance_units": c, "clearance_m": c * a.unit_m if c > 0 else "default", "missions": n_m, "admissible_missions": int(adm), "flights": n,
                 "L2_rate": k_l2 / n if n else float("nan"), "L2_lo": lo, "L2_hi": hi, "near_miss_or_collision_rate": float(near.mean()) if n else float("nan"),
                 "mission_completed_rate": float(fl.mission_completed.mean()) if n else float("nan")})
pd.DataFrame(summ).to_csv(os.path.join(a.out, "clearance_summary.csv"), index=False)
print(pd.DataFrame(summ).round(3).to_string(index=False))

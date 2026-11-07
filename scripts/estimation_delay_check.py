#!/usr/bin/env python
"""State-estimation delay at the planning level (v12, no model calls).

When a change is detected the re-planner starts from the ESTIMATED position. If the estimate lags by tau seconds, the vehicle is really v*tau further along its
old route. For every zone change of the model-free mid-flight cases (new or enlarged zone, airport and synthetic missions), the visibility-graph planner plans
from the lagging estimate; the plan is then checked from the TRUE position (the vehicle flies from where it really is to the plan's first waypoint) against the
changed mission. Reported: share of re-plans that are still valid, and of those that cross a zone, for tau = 0, 0.5, 1, 2, 5 s.
Speeds: 22 m/s fixed wing (airport missions, metres), 14 m/s quadrotor at 20 m per map unit (synthetic missions).

  python scripts/estimation_delay_check.py --out outputs/review_v12/estimation_delay
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import yaml
from src.benchmark.generator import load_scenarios
from src.benchmark.feasibility import assess
from src.experiments.dynamic_runner import build_cases
from src.models.schemas import Mission
from src.verifier.verifier import verify

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="outputs/review_v12/estimation_delay")
ap.add_argument("--max-cases", type=int, default=120)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
dcfg = yaml.safe_load(open("config/experiments_dynamic_modelfree.yaml", encoding="utf-8"))["dynamic"]
dcfg = {**dcfg, "detection_delays": [0.0], "changes": ["add_zone", "enlarge_zone"]}
rows = []
for sdir, speed_units, label in (("benchmark/real", 22.0, "airport"), ("benchmark/generated", 14.0 / 20.0, "synthetic")):
    cases = [c for c in build_cases(load_scenarios(sdir), dcfg) if c["feas"].status == "FEASIBLE"][:a.max_cases]
    for c in cases:
        m_true = c["mission"]
        rem = c["remaining_plan"]
        p_true = tuple(m_true.start)
        nxt = rem.ordered()[0] if rem.waypoints else rem.destination
        dx, dy = nxt.x - p_true[0], nxt.y - p_true[1]
        n = math.hypot(dx, dy) or 1.0
        for tau in (0.0, 0.5, 1.0, 2.0, 5.0):
            lag = speed_units * tau
            p_est = [p_true[0] - dx / n * lag, p_true[1] - dy / n * lag]
            d = m_true.to_dict()
            d["start"] = p_est
            d["max_distance"] = m_true.max_distance + lag          # the planner thinks it still has to fly the lag
            m_est = Mission.from_dict(d)
            f = assess(m_est)
            if f.status != "FEASIBLE":
                rows.append({"set": label, "case": c["case_id"], "tau_s": tau, "planned": False})
                continue
            v = verify(m_true, f.witness_plan)                     # executed from where the vehicle really is
            fams = {x.family for x in v.violations}
            rows.append({"set": label, "case": c["case_id"], "change": c["change"], "tau_s": tau, "planned": True, "valid_from_true_position": v.success,
                         "zone_violation": "no_fly_zone" in fams, "lag_m": lag * (1.0 if label == "airport" else 20.0)})
    print(label, len(cases), "cases", flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(a.out, "estimation_delay.csv"), index=False)
pl = df[df.planned == True]  # noqa: E712
summ = pl.groupby(["set", "tau_s"]).agg(cases=("case", "size"), valid=("valid_from_true_position", "mean"), zone=("zone_violation", "mean"), lag_m=("lag_m", "first")).reset_index()
summ.to_csv(os.path.join(a.out, "estimation_delay_summary.csv"), index=False)
print(summ.round(3).to_string(index=False))

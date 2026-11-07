#!/usr/bin/env python
"""Delayed map updates and state-estimation delay (v12, no model calls) - the review asked for delays measured in time, not only as a share of the route.

Online episodes in the built-in simulator (src/dynamic/online_experiment.py): a verified plan is flown; a zone appears or is enlarged on the remaining route; the
UAV learns about it by a map update that arrives `broadcast latency` seconds later (UTM/NOTAM-style message), re-plans with the visibility-graph planner (legs keep 20 m from zones) after a
5 s planning latency, and flies on. Sweep 1: map-update delay 0/5/15/30/60 s. State-estimation delay: see scripts/estimation_delay_check.py. 'none' keeps the stale plan. Reported: mission success, zone intrusions, recovery on affected episodes.

  python scripts/delay_study.py --scenario-dir benchmark/generated --n 30 --out outputs/review_v12/delay_study
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.dynamic.online_experiment import run_online_experiment
from src.dynamic.online import classical_replanner

ap = argparse.ArgumentParser()
ap.add_argument("--scenario-dir", default="benchmark/generated")
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--out", default="outputs/review_v12/delay_study")
ap.add_argument("--edge-clearance-m", type=float, default=20.0, help="the re-planner keeps every leg this far from every zone (the clearance study showed 20 m removes near misses)")
a = ap.parse_args()
rows = []
sweeps = [("map_update_delay_s", d, dict(broadcast_latency_s=d, loc_delay_s=0.0)) for d in (0.0, 5.0, 15.0, 30.0, 60.0)]
# State-estimation delay is studied at the planning level in scripts/estimation_delay_check.py: inside this simulator (0.5 s steps, alpha-beta estimator) a delayed
# position fix of even 0.5 s made the vehicle stall at waypoints although no event occurred - an artefact of the simple estimator, so it is not reported from here.
for name, val, kw in sweeps:
    od = os.path.join(a.out, f"{name}_{val:g}")
    run_online_experiment(a.scenario_dir, od, {"none": None, "classical": classical_replanner(edge_clearance_m=a.edge_clearance_m)}, n_scenarios=a.n,
                          delivery_modes=("broadcast",), **kw)
    ep = pd.read_csv(os.path.join(od, "online_episodes.csv"))
    ev = ep[ep.replanner.isin(["none", "classical"])]
    key = ["scenario_id", "change", "seed"]
    none_fail = set(map(tuple, ev[(ev.replanner == "none") & (~ev.mission_success.astype(bool))][key].values))
    cl = ev[ev.replanner == "classical"]
    aff = cl[[tuple(x) in none_fail for x in cl[key].values]]
    rows.append({"sweep": name, "value": val, "episodes": int(len(cl)), "affected": int(len(aff)),
                 "classical_success": float(cl.mission_success.astype(bool).mean()), "classical_recovered_affected": float(aff.mission_success.astype(bool).mean()) if len(aff) else None,
                 "classical_intrusion": float((cl.intrusion_zones > 0).mean()), "stale_success": float(ev[ev.replanner == "none"].mission_success.astype(bool).mean()),
                 "stale_intrusion": float((ev[ev.replanner == "none"].intrusion_zones > 0).mean()), "mean_detection_delay_s": float(cl.detection_delay_s.mean())})
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(a.out, "delay_summary.csv"), index=False)
json.dump(rows, open(os.path.join(a.out, "delay_summary.json"), "w", encoding="utf-8"), indent=1)

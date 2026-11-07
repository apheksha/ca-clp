"""Re-verify every saved FINAL plan under different energy weights k_v (no API calls).
usage: python tools/energy_sensitivity.py <runs.jsonl> [--factors 0 0.5 1 2 4]
The energy budget is kept fixed in absolute terms; only the climb weight k_v is rescaled, so the table shows how conclusions depend on the energy model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow `python tools/<x>.py` from the repo root
import sys, argparse, collections, dataclasses
from tools._common import runs, mission_map, FlightPlan
from src.verifier.verifier import verify

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--factors", type=float, nargs="*", default=[0, 0.5, 1, 2, 4])
a = ap.parse_args()
rows = runs(a.runs); mm = mission_map(rows)
ok = collections.defaultdict(lambda: collections.Counter()); tot = collections.Counter()
for r in rows:
    if not r.get("final_plan"):
        tot[r["condition"]] += 1; continue
    m = mm.get(r["scenario_id"])
    tot[r["condition"]] += 1
    if m is None:
        continue
    plan = FlightPlan.from_dict(r["final_plan"])
    for f in a.factors:
        m2 = dataclasses.replace(m, energy_k_v=m.energy_k_v * f)
        ok[r["condition"]][f] += int(verify(m2, plan).success)
print("condition," + ",".join(f"kv x{f:g}" for f in a.factors))
for c in tot:
    print(c + "," + ",".join(f"{100*ok[c][f]/tot[c]:.1f}" for f in a.factors))

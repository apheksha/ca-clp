"""Dynamic-study summary with an explicit seed filter and stop-reason / status breakdown (no API calls).
usage: python tools/dynamic_summary.py <dynamic runs.jsonl> [--seed 0]   (--seed -1 keeps all seeds)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow `python tools/<x>.py` from the repo root
import argparse, collections
from tools._common import runs, wilson

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rows = [r for r in runs(a.runs) if a.seed < 0 or r["seed"] == a.seed]
by = collections.defaultdict(list)
for r in rows:
    by[r["condition"]].append(r)
print(f"seed filter: {'all' if a.seed < 0 else a.seed}; runs kept: {len(rows)}; distinct cases: {len({r['scenario_id'] for r in rows})}")
print("condition,n,success%,ci_lo,ci_hi,statuses")
for c, v in by.items():
    f = [r for r in v if r["scenario_meta"]["feasibility"] == "FEASIBLE"]
    k = sum(r["metrics"]["final_success"] for r in f); lo, hi = wilson(k, len(f))
    st = collections.Counter(r["final_status"] for r in f)
    print(f"{c},{len(f)},{100*k/max(1,len(f)):.1f},{100*lo:.1f},{100*hi:.1f},{dict(st)}")
print("\nby change type (success% over feasible cases):")
ch = collections.defaultdict(lambda: [0, 0])
for r in rows:
    if r["scenario_meta"]["feasibility"] == "FEASIBLE":
        ch[(r["condition"], r["dynamic"]["change"])][0] += r["metrics"]["final_success"]; ch[(r["condition"], r["dynamic"]["change"])][1] += 1
for (c, g), (k, n) in sorted(ch.items()):
    print(f"{c},{g},{k}/{n}")

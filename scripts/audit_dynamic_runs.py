#!/usr/bin/env python
"""Re-verify the stale plans of recorded dynamic runs with the corrected mission id and recompute the dynamic results on the truly affected cases (no API calls).
usage: python scripts/audit_dynamic_runs.py <dynamic raw dir> [--static <static raw dir of the same benchmark>] [--out outputs/review_v12/dynamic_audit]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.dynamic_audit import audit_dynamic_run

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--static")
ap.add_argument("--out", default="outputs/review_v12/dynamic_audit")
a = ap.parse_args()
out = os.path.join(a.out, os.path.basename(os.path.normpath(a.run_dir)))
s = audit_dynamic_run(a.run_dir, out, a.static)
print(json.dumps({k: v for k, v in s.items() if k != "selection"}, indent=1))
print("written to", out)

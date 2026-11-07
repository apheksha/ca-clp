#!/usr/bin/env python
"""Prints/saves the geometry self-test characterisation (counts, tolerance, adversarial cases) for the paper. python scripts/geometry_test_report.py [--n 3000]"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.geometry_selftest import run_selftest

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--out", default="outputs/geometry_selftest.json")
a = ap.parse_args()
r = run_selftest(a.n)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
json.dump(r, open(a.out, "w"), indent=1)
print(json.dumps(r, indent=1))
sys.exit(0 if r["ok"] else 1)

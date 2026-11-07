"""Copy a deterministic, stratified subset of benchmark/generated into benchmark/reduced (first k scenarios of every group, sorted by id).
Usage: python scripts/make_reduced_benchmark.py [--per-level 5] [--per-combo 1] [--corridor 3] [--infeasible 2]"""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="benchmark/generated"); ap.add_argument("--dst", default="benchmark/reduced")
ap.add_argument("--per-level", type=int, default=5); ap.add_argument("--per-combo", type=int, default=1)
ap.add_argument("--corridor", type=int, default=3); ap.add_argument("--infeasible", type=int, default=2)
a = ap.parse_args()
quota = {"easy": a.per_level, "medium": a.per_level, "hard": a.per_level, "corridor": a.corridor, "infeasible": a.infeasible}
files = sorted(f for f in os.listdir(a.src) if f.endswith(".json"))
groups = {}
for f in files:
    s = json.load(open(os.path.join(a.src, f)))
    g = s["difficulty"] if s["difficulty"] != "combo" else f.rsplit("_", 1)[0]          # one group per constraint combination
    groups.setdefault(g, []).append(f)
os.makedirs(a.dst, exist_ok=True)
for f in os.listdir(a.dst):
    if f.endswith(".json"):
        os.remove(os.path.join(a.dst, f))
n = 0
for g, fs in sorted(groups.items()):
    k = quota.get(g, a.per_combo)
    for f in fs[:k]:
        shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f)); n += 1
print(f"copied {n} scenarios from {len(groups)} groups -> {a.dst}")

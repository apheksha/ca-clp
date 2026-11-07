"""Rebuild the airport scenario JSON files (benchmark/real/*.json) from a finished airport run's logged prompts (no network, no API).
usage: python tools/rebuild_airport_scenarios.py <airport runs.jsonl> [--out benchmark/real]
Needed because the OurAirports download is not bundled; the logged [MISSION_SPEC] is the exact mission that was flown."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow `python tools/<x>.py` from the repo root
import argparse, json, os, re
from tools._common import runs
ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--out", default="benchmark/real"); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); seen = set()
for r in runs(a.runs):
    if r["scenario_id"] in seen:
        continue
    for it in r["iterations"]:
        for c in it["calls"]:
            p = c.get("prompt", "")
            mt = re.search(r"\[MISSION_SPEC\](.*?)\[/MISSION_SPEC\]", p, re.S); dt = re.search(r"MISSION DESCRIPTION:\n(.*?)\n\nMISSION SPECIFICATION", p, re.S)
            if mt and dt and r["scenario_id"] not in seen:
                spec = json.loads(mt.group(1)); spec["description"] = dt.group(1)
                json.dump({"scenario_id": r["scenario_id"], "difficulty": r["difficulty"], "mission": spec, "metadata": r["scenario_meta"]},
                          open(os.path.join(a.out, r["scenario_id"] + ".json"), "w"), indent=1); seen.add(r["scenario_id"])
print(len(seen), "scenarios written to", a.out)

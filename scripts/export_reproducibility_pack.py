#!/usr/bin/env python
"""Writes a self-contained reproducibility pack (reviewer: 'publish fixed scenario manifests, prompt templates, seeds, model configuration and verification tests').

outputs/reproducibility_pack/
  prompts/system_prompt.txt, initial_<mode>.txt, repair_<level>_<mode>.txt   exact prompt text (on a fixed example mission) + PROMPT_VERSION
  scenario_manifest.csv      scenario_id, difficulty, source, feasibility, sha256 of the scenario file (+ generator base_seed / attempt)
  configs/                   copies of config/*.yaml
  tests_inventory.txt        every test module and test function (unittest discovery names)
  PACK_README.md             versions, seeds, how to regenerate every number
Usage: python scripts/export_reproducibility_pack.py [--out outputs/reproducibility_pack]"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, os, shutil, sys, platform, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.schemas import Mission
from src.planner.prompts import SYSTEM_PROMPT, PROMPT_VERSION, PROMPT_MODES, initial_prompt
from src.feedback.feedback import build_feedback, repair_prompt, LEVELS
from src.verifier.verifier import verify
from src.benchmark.feasibility import assess
from src.models.schemas import FlightPlan, Waypoint


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/reproducibility_pack")
    ap.add_argument("--example", default="benchmark/generated/hard_000.json")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "prompts"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "configs"), exist_ok=True)
    m = Mission.from_dict(json.load(open(a.example))["mission"])
    open(os.path.join(a.out, "prompts", "system_prompt.txt"), "w").write(f"[PROMPT_VERSION={PROMPT_VERSION}]\n{SYSTEM_PROMPT}\n")
    for mode in PROMPT_MODES:
        open(os.path.join(a.out, "prompts", f"initial_{mode}.txt"), "w").write(initial_prompt(m, mode))
    # a deliberately bad plan (straight line through everything) to show the repair prompts
    wps = [Waypoint(r.id, r.x, r.y, (m.min_altitude + m.max_altitude) / 2, i + 1) for i, r in enumerate(m.required_waypoints)]
    plan = FlightPlan(m.mission_id, wps, Waypoint("DEST", m.destination[0], m.destination[1], (m.min_altitude + m.max_altitude) / 2, len(wps) + 1)).validate()
    res = verify(m, plan)
    for lvl in LEVELS:
        for mode in PROMPT_MODES:
            fb = build_feedback(m, plan, res, lvl, "targeted" if lvl in ("evidence", "prescriptive") else "regenerate")
            open(os.path.join(a.out, "prompts", f"repair_{lvl}_{mode}.txt"), "w").write(repair_prompt(m, plan, fb, mode))
    rows = []
    for d in ("benchmark/generated", "benchmark/real", "benchmark/reduced", "benchmark/pilot"):
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            rec = json.load(open(f))
            rows.append({"scenario_id": rec["scenario_id"], "folder": d, "difficulty": rec.get("difficulty"), "source": rec["metadata"].get("source", "synthetic"),
                         "feasibility": rec["metadata"].get("feasibility"), "base_seed": rec.get("base_seed"), "attempt": rec.get("attempt"), "sha256": sha(f)})
    with open(os.path.join(a.out, "scenario_manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for f in glob.glob("config/*.yaml"):
        shutil.copy(f, os.path.join(a.out, "configs"))
    names = []
    def walk(s):
        for t in s:
            walk(t) if isinstance(t, unittest.TestSuite) else names.append(t.id())
    walk(unittest.defaultTestLoader.discover("tests", top_level_dir="."))
    open(os.path.join(a.out, "tests_inventory.txt"), "w").write(f"{len(names)} tests\n" + "\n".join(sorted(names)) + "\n")
    open(os.path.join(a.out, "PACK_README.md"), "w").write(f"""# Reproducibility pack
- prompt version: `{PROMPT_VERSION}`; prompt modes: {', '.join(PROMPT_MODES)}
- python {platform.python_version()} on {platform.platform()}
- scenarios: {len(rows)} files hashed in scenario_manifest.csv (regenerate with `python run.py generate`; benchmark base_seed in config/default.yaml)
- every real run also writes outputs/raw/<run>/manifest.json (model id, temperature audit, prompt version/mode, scenario hashes, code fingerprint)
- verification tests: {len(names)} (tests_inventory.txt); geometry self-test: `python scripts/geometry_test_report.py`; oracle validation: `python scripts/oracle_validation.py`
- all metrics are recomputed from raw runs.jsonl by `python run.py evaluate <raw_dir>`
""")
    print("pack written to", a.out, f"({len(rows)} scenarios, {len(names)} tests)")


if __name__ == "__main__":
    main()

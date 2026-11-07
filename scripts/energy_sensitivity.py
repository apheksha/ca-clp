#!/usr/bin/env python
"""Energy-model sensitivity (reviewer: 'Energy-model sensitivity: test alternative vehicle parameters and energy-cost coefficients').

The verifier's energy is a COST PROXY: E = L + k_v*A + k_wp*n  (units = route-length units; k_v: length-units per metre climb; k_wp: length-units per waypoint).
This script asks: how much do the benchmark conclusions depend on the chosen coefficients?
  Part 1 (no LLM needed): for every stored mission with an energy budget, re-run the oracle for a grid of (k_v, k_wp) with
        (a) the budget kept FIXED in absolute terms, and (b) the budget rescaled so that the SLACK over the witness energy is preserved.
        Output: share of missions still FEASIBLE, mean witness energy.
  Part 2 (optional --run-dir): re-verify the released final plans of a raw run under every grid point (fixed budget) -> how many verified-successful
        plans would have violated the battery under other coefficients (robustness of the energy verdict).
Writes outputs/energy_sensitivity/*.csv"""
from __future__ import annotations
import argparse
import copy
import glob
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.models.schemas import Mission, FlightPlan
from src.benchmark.feasibility import assess
from src.verifier.verifier import verify, resources

KV = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
KWP = [0.0, 0.5, 2.0, 5.0]
KV_FACTORS = [0.0, 0.5, 1.0, 2.0, 4.0]          # --relative: multiples of each mission's own k_v
KWP_FRACTIONS = [0.0, 0.01, 0.02, 0.05]         # --relative: per-waypoint overhead as a fraction of the mission's witness route length


def _coeffs(m, kv, kw, relative: bool, witness_len):
    """(k_v, k_wp) for one grid point. relative: kv is a factor on the mission's own k_v, kw a fraction of its witness route length per waypoint."""
    if not relative:
        return kv, kw
    return kv * m.energy_k_v, kw * (witness_len or 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario-dir", nargs="*", default=["benchmark/generated", "benchmark/real"])
    ap.add_argument("--run-dir")
    ap.add_argument("--out", default="outputs/energy_sensitivity")
    ap.add_argument("--relative", action="store_true", help="scale-free grid (v12): k_v = factor x the mission's own k_v; k_wp = fraction x the mission's witness route length "
                    "per waypoint. Without it the grid values are absolute (original behaviour), which is only meaningful when all missions share one scale.")
    a = ap.parse_args()
    global KV, KWP
    if a.relative:
        KV, KWP = KV_FACTORS, KWP_FRACTIONS
    os.makedirs(a.out, exist_ok=True)
    ms = []
    for d in a.scenario_dir:
        for f in sorted(glob.glob(os.path.join(d, "*.json"))):
            rec = json.load(open(f))
            m = Mission.from_dict(rec["mission"])
            if m.is_active("energy") and m.max_energy is not None and rec["metadata"].get("feasibility") == "FEASIBLE":
                w_e = rec["metadata"].get("witness_energy")
                if not w_e:     # airport scenarios do not store it: witness energy under the mission's own coefficients (needed for the slack-preserving variant)
                    m0 = copy.deepcopy(m)
                    m0.max_energy = 1e12
                    f0 = assess(m0)
                    w_e = resources(f0.witness_plan, m)["energy"] if f0.status == "FEASIBLE" else None
                ms.append((rec["scenario_id"], m, w_e))
    wd = {}
    for sid, m, _ in ms:
        m0 = copy.deepcopy(m)
        m0.max_energy, m0.max_distance = 1e12, 1e12
        f0 = assess(m0)
        wd[sid] = f0.witness_distance if f0.status == "FEASIBLE" else None
    rows = []
    for kv in KV:
        for kw in KWP:
            fixed_ok = resc_ok = 0
            e_sum = 0.0
            for sid, m, w_e in ms:
                m2 = copy.deepcopy(m)
                m2.energy_k_v, m2.energy_k_wp = _coeffs(m, kv, kw, a.relative, wd.get(sid))
                f = assess(m2)
                fixed_ok += f.status == "FEASIBLE"
                m3 = copy.deepcopy(m2)
                m3.max_energy = 1e12
                f3 = assess(m3)
                if f3.status == "FEASIBLE" and w_e:
                    e_new = resources(f3.witness_plan, m3)["energy"]
                    slack = m.max_energy / w_e
                    m3.max_energy = e_new * slack
                    resc_ok += assess(m3).status == "FEASIBLE"
                    e_sum += e_new
            rows.append({"k_v": kv, "k_wp": kw, "n_missions": len(ms), "feasible_fixed_budget": fixed_ok / max(len(ms), 1),
                         "feasible_slack_preserved": resc_ok / max(len(ms), 1), "mean_witness_energy": e_sum / max(len(ms), 1)})
    for r_ in rows:
        r_["grid_mode"] = "relative" if a.relative else "absolute"
    pd.DataFrame(rows).to_csv(os.path.join(a.out, "oracle_energy_sensitivity.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    if a.run_dir:
        from src.utils.io import read_jsonl
        recs = [r for r in read_jsonl(os.path.join(a.run_dir, "runs.jsonl")) if r.get("final_plan") and (r.get("final_verification") or {}).get("success") and r["metrics"].get("released")]
        res = []
        for kv in KV:
            for kw in KWP:
                bad = tot = 0
                for r in recs:
                    mj = (r.get("dynamic") or {}).get("mission")
                    if mj is None:
                        fs = glob.glob(os.path.join("benchmark", "*", r["scenario_id"] + ".json"))
                        if not fs:
                            continue
                        mj = json.load(open(fs[0]))["mission"]
                    m = Mission.from_dict(mj)
                    if not (m.is_active("energy") and m.max_energy is not None):
                        continue
                    if a.relative and r["scenario_id"] not in wd:
                        m0 = copy.deepcopy(m)
                        m0.max_energy, m0.max_distance = 1e12, 1e12
                        f0 = assess(m0)
                        wd[r["scenario_id"]] = f0.witness_distance if f0.status == "FEASIBLE" else None
                    m.energy_k_v, m.energy_k_wp = _coeffs(m, kv, kw, a.relative, wd.get(r["scenario_id"]))
                    tot += 1
                    bad += not verify(m, FlightPlan.from_dict(r["final_plan"])).success
                res.append({"k_v": kv, "k_wp": kw, "grid_mode": "relative" if a.relative else "absolute", "n_successful_plans_with_energy_budget": tot,
                            "fraction_violating_under_new_coefficients": bad / tot if tot else None})
        pd.DataFrame(res).to_csv(os.path.join(a.out, "plan_energy_robustness.csv"), index=False)
        print(pd.DataFrame(res).round(3).to_string(index=False))


if __name__ == "__main__":
    main()

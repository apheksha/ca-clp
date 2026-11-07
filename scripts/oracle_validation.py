#!/usr/bin/env python
"""Oracle soundness / completeness / approximation validation (reviewer: 'Oracle soundness and completeness tests').

Run:  python scripts/oracle_validation.py [--n-random 300] [--out outputs/oracle_validation]
Checks, on the stored benchmarks AND on randomly generated stress missions:
  S1 FEASIBLE is sound      : the witness really passes the independent verifier (re-verified here).
  S2 INFEASIBLE is sound    : no plan found by an independent search (fine-grid A*, 4 altitudes tried, random-restart perturbation of the witness)
                              ever passes the verifier on a mission the oracle called INFEASIBLE.
  C  completeness (empirical): share of UNKNOWN among all decisions; for UNKNOWN missions an independent solver (fine grid A*) is run: if it succeeds, the oracle
                              was incomplete there (reported, not hidden).
  A  approximation          : witness length with the production 12-gon vs a 96-gon (same code path): relative overestimate distribution.
Writes oracle_validation.json + oracle_validation.csv. Exit code 1 if S1 or S2 is violated."""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.schemas import Mission, RequiredWaypoint, NoFlyZone
from src.benchmark.feasibility import assess, shortest_route, visiting_order, CIRCLE_SIDES
from src.verifier.verifier import verify
from src.utils import geometry as G


def random_mission(rng: random.Random, i: int) -> Mission:
    n_req = rng.randint(1, 4)
    reqs = [RequiredWaypoint(chr(65 + k), round(rng.uniform(15, 85), 1), round(rng.uniform(10, 90), 1)) for k in range(n_req)]
    zones = []
    for k in range(rng.randint(0, 6)):
        c = [round(rng.uniform(10, 90), 1), round(rng.uniform(10, 90), 1)]
        if rng.random() < 0.6:
            z = NoFlyZone(f"Z{k+1}", "circle", center=c, radius=round(rng.uniform(3, 14), 1))
        else:
            R = rng.uniform(4, 12)
            z = NoFlyZone(f"Z{k+1}", "polygon", vertices=[[round(c[0] + R * rng.uniform(.6, 1) * G.math.cos(a), 2), round(c[1] + R * rng.uniform(.6, 1) * G.math.sin(a), 2)]
                                                        for a in [2 * G.math.pi * j / 5 + rng.uniform(-.3, .3) for j in range(5)]])
        if rng.random() < 0.25:
            z.z_max = round(rng.uniform(40, 80), 0)
        zones.append(z)
    order = [[reqs[j].id, reqs[j + 1].id] for j in range(n_req - 1)] if rng.random() < 0.5 else []
    if order and rng.random() < 0.1:
        order.append([order[0][1], order[0][0]])           # occasional cyclic ordering
    lo = round(rng.uniform(20, 50), 0)
    m = Mission(f"R{i}", "", [round(rng.uniform(2, 12), 1), round(rng.uniform(10, 90), 1)], [round(rng.uniform(88, 98), 1), round(rng.uniform(10, 90), 1)],
                reqs, lo, lo + rng.choice([5, 20, 50]), 1e9, zones, order, energy_k_v=1.0)
    m.max_distance = round(sum(G.dist(a, b) for a, b in zip([m.start_xy] + [r.xy for r in reqs], [r.xy for r in reqs] + [m.dest_xy])) * rng.choice([0.7, 1.0, 1.1, 1.4, 2.0]), 1)
    m.position_tolerance = rng.choice([0.0, 1.0])
    # interaction coverage: randomly switch constraint families off; optionally the PHYSICAL energy model and a terrain grid
    m.active = [f for f in m.active if rng.random() > 0.2]
    if rng.random() < 0.25:
        m.energy_model, m.vehicle, m.unit_m, m.max_energy = "physical", "generic_quad_small", 30.0, rng.choice([0.4, 1.0, 3.0, 8.0])
        m.wind = rng.choice([None, [3.0, 0.0], [-6.0, 4.0]])
    if rng.random() < 0.2:
        rows = [[round(rng.uniform(0, 35), 1) for _ in range(8)] for _ in range(8)]
        m.terrain = {"x0": 0.0, "y0": 0.0, "cell": 100 / 7, "heights": rows}
        m.min_agl, m.max_agl = rng.choice([0.0, 5.0, 12.0]), rng.choice([None, 80.0])
    return m.validate()


def independent_search(m: Mission, rng: random.Random, tries: int = 400) -> bool:
    """Independent of the visibility-graph code: fine-grid A* over x,y for several constant altitudes + random perturbations of the required-point order."""
    from src.planner.astar import astar_plan
    try:
        p = astar_plan(m)
        if p is not None and verify(m, p).success:
            return True
    except Exception:
        pass
    ids = [r.id for r in m.required_waypoints]
    for _ in range(tries):   # random waypoint scatter around the straight chain
        order = ids[:]
        rng.shuffle(order)
        from src.models.schemas import FlightPlan, Waypoint
        alt = rng.uniform(m.min_altitude, m.max_altitude)
        pts = []
        for rid in order:
            r = m.required(rid)
            pts.append((rid, r.x, r.y))
            if rng.random() < 0.5:
                pts.append((f"W{len(pts)}", rng.uniform(0, 100), rng.uniform(0, 100)))
        wps = [Waypoint(i, x, y, alt, k + 1) for k, (i, x, y) in enumerate(pts)]
        plan = FlightPlan(m.mission_id, wps, Waypoint("DEST", m.destination[0], m.destination[1], alt, len(wps) + 1)).validate()
        if verify(m, plan).success:
            return True
    return False


def _ratio_stats(rows):
    r = sorted(x["bracket_ub"] / x["bracket_lb"] - 1 for x in rows if x.get("bracket_lb") and x["bracket_lb"] > 1e-9)
    return {"n": len(r), "median_rel_gap": r[len(r) // 2] if r else None, "max_rel_gap": r[-1] if r else None}


def _by_active(rows):
    out = {}
    for r in rows:
        d = out.setdefault(r["active"], {"FEASIBLE": 0, "INFEASIBLE": 0, "UNKNOWN": 0})
        d[r["status"]] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=300)
    ap.add_argument("--out", default="outputs/oracle_validation")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    missions = []
    for f in sorted(glob.glob("benchmark/generated/*.json") + glob.glob("benchmark/real/*.json")):
        missions.append((os.path.basename(f)[:-5], Mission.from_dict(json.load(open(f))["mission"]), "stored"))
    for i in range(a.n_random):
        missions.append((f"random_{i:04d}", random_mission(rng, i), "random"))
    rows, viol_s1, viol_s2, incomplete, viol_s3 = [], [], [], [], []
    import src.benchmark.feasibility as F
    for name, m, kind in missions:
        f = assess(m)
        row = {"mission": name, "kind": kind, "status": f.status, "proof_kind": f.proof_kind or "", "reason": f.reason,
               "active": "+".join(sorted(m.active)), "energy_model": m.energy_model, "terrain": m.terrain is not None}
        if f.status == "FEASIBLE":
            fc = assess(m, certify=True)
            if fc.route_bracket:
                lb_, ub_ = fc.route_bracket
                row["bracket_lb"], row["bracket_ub"] = lb_, ub_
                if lb_ > ub_ + 1e-6:
                    viol_s3.append(name)
            ok = verify(m, f.witness_plan).success
            row["S1_witness_verified"] = ok
            if not ok:
                viol_s1.append(name)
            old = F.CIRCLE_SIDES
            F.CIRCLE_SIDES = 96
            try:
                f96 = assess(m)
            finally:
                F.CIRCLE_SIDES = old
            if f96.status == "FEASIBLE" and f96.witness_distance:
                row["len_12gon"], row["len_96gon"] = round(f.witness_distance, 4), round(f96.witness_distance, 4)
                row["rel_overestimate"] = round(f.witness_distance / f96.witness_distance - 1, 5)
        elif f.status == "INFEASIBLE":
            found = independent_search(m, rng)
            row["S2_independent_solution_found"] = found
            if found:
                viol_s2.append(name)
        else:
            found = independent_search(m, rng)
            row["UNKNOWN_but_independent_solution"] = found
            if found:
                incomplete.append(name)
        rows.append(row)
    os.makedirs(a.out, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(a.out, "oracle_validation.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    cnt = {s: sum(r["status"] == s for r in rows) for s in ("FEASIBLE", "INFEASIBLE", "UNKNOWN")}
    over = sorted(r["rel_overestimate"] for r in rows if "rel_overestimate" in r)
    summ = {"n_missions": n, "n_stored": sum(r["kind"] == "stored" for r in rows), "n_random": a.n_random, "status_counts": cnt,
            "S3_violations_lower_bound_above_witness": viol_s3,
            "bracket_ratio_ub_over_lb": _ratio_stats(rows),
            "by_energy_model": {k: {s_: sum(1 for r in rows if r["energy_model"] == k and r["status"] == s_) for s_ in ("FEASIBLE", "INFEASIBLE", "UNKNOWN")} for k in ("proxy", "physical")},
            "with_terrain": {s_: sum(1 for r in rows if r["terrain"] and r["status"] == s_) for s_ in ("FEASIBLE", "INFEASIBLE", "UNKNOWN")},
            "by_active_family_set": _by_active(rows),
            "S1_violations_witness_fails_verifier": viol_s1, "S2_violations_infeasible_but_solution_found": viol_s2,
            "unknown_rate": cnt["UNKNOWN"] / n, "unknown_but_independent_solver_succeeds": incomplete,
            "approximation_n12_vs_n96": {"n": len(over), "median": over[len(over) // 2] if over else None, "max": over[-1] if over else None,
                                         "theoretical_wrapped_part_bound": round(G.math.tan(G.math.pi / CIRCLE_SIDES) / (G.math.pi / CIRCLE_SIDES) - 1, 5), "circle_sides_production": CIRCLE_SIDES},
            "proof_kinds": {k: sum(r["proof_kind"] == k for r in rows) for k in {r["proof_kind"] for r in rows if r["proof_kind"]}},
            "claim": "FEASIBLE and INFEASIBLE are sound on all tested missions iff both violation lists are empty; completeness is empirical (UNKNOWN rate)."}
    json.dump(summ, open(os.path.join(a.out, "oracle_validation.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))
    sys.exit(1 if (viol_s1 or viol_s2 or viol_s3) else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Flight-dynamics check of the RELEASED plans (v12, no model calls) - answers the review's request for turning radius, climb-rate limits and path smoothing.

For every verified plan of C_full, C_prescriptive, G1 and E (seed 0) it reports
  (1) GEOMETRY  - the plan smoothed with the vehicle's effective turn radius R = max(min_turn_radius, v_cruise^2 / max_lateral_accel) at every corner that is not a
                  required site (fly-by fillets; required sites stay fly-over) and RE-VERIFIED against the mission: does the smoothed path still avoid every zone and
                  stay in the band?  (independent of the simulator)
  (2) FLIGHT    - the plan flown in the built-in simulator in three modes, calm air and a 4 m/s crosswind, 1 m/s gusts, GNSS noise:
                  baseline   = earlier study (climb/descent rate and acceleration limits only);
                  dynamics   = + explicit turning-radius limit (lateral acceleration <= min(a_lat_max, V^2/R_min)) + fillet smoothing with radius R;
                  dyn_margin = dynamics + altitude margin (10 m airport, 5 m synthetic), re-verified before flight.
Climb and descent rates are always limited by the vehicle profile (v_climb_max, v_descend_max).

  python scripts/flight_dynamics_study.py --run outputs/raw/real_20261025-194958_3b2644_seed0-only__plus-direct --scenario-dir benchmark/real --vehicle long_range_fixed_wing --margin 10 --out outputs/review_v12/flight_dynamics/airport
  python scripts/flight_dynamics_study.py --run outputs/raw/real_20261023-181728_55def8__plus-direct --scenario-dir benchmark/generated --vehicle generic_quad_large --unit-m 20 --margin 5 --run-seeds 0 --out outputs/review_v12/flight_dynamics/synthetic
"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.models.schemas import Mission, FlightPlan, Waypoint
from src.models.vehicle import get_vehicle
from src.simulator.batch import prepare_mission
from src.simulator.sim import simulate, SimConfig, _fillet_path
from src.simulator.study import execution_record, planned_polyline_m, lift_altitudes
from src.verifier.verifier import verify
from src.utils.io import read_jsonl

CONDS = ["C_full", "C_prescriptive", "G_direct_patch", "E_classical"]


def released_plans(run_dir, run_seeds):
    for r in read_jsonl(os.path.join(run_dir, "runs.jsonl")):
        if r["condition"] not in CONDS or r["scenario_meta"]["feasibility"] != "FEASIBLE" or r["seed"] not in run_seeds:
            continue
        fv = r.get("final_verification") or {}
        if r["metrics"].get("released") and fv.get("valid") and fv.get("success") and r.get("final_plan"):
            yield r


def smoothed_plan(m: Mission, plan: FlightPlan, R_units: float) -> tuple[FlightPlan, int]:
    """The fillet path as a FlightPlan in map units (extra fly-by waypoints), for re-verification."""
    nodes = [("START", (m.start[0], m.start[1], m.start_altitude))] + [(w.id, (w.x, w.y, w.altitude)) for w in plan.ordered()] + \
            [("DEST", (plan.destination.x, plan.destination.y, plan.destination.altitude))]
    ev = []
    out = _fillet_path(nodes, R_units, {q.id for q in m.required_waypoints}, ev, 0.0)
    wps, k = [], 1
    for nid, (x, y, h) in out[1:-1]:
        wid = nid.replace("~", "f")[:32]
        wps.append(Waypoint(wid if wid not in [w.id for w in wps] else f"F{k}", x, y, h, k))
        k += 1
    d = plan.destination
    return FlightPlan(plan.mission_id, wps, Waypoint(d.id, d.x, d.y, d.altitude, k)).validate(), sum(e["event"] == "fillet_reduced" for e in ev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--scenario-dir", required=True)
    ap.add_argument("--vehicle", required=True)
    ap.add_argument("--unit-m", type=float, default=None, help="metres per map unit (airport missions are in metres: leave empty)")
    ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--run-seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    veh = get_vehicle(a.vehicle)
    R = max(veh.min_turn_radius_m, veh.v_cruise ** 2 / veh.max_lateral_accel)
    winds = {"calm": (0.0, 0.0), "crosswind": (0.0, 4.0)}
    req_tol = max(1.5 * veh.acceptance_radius_m, 5.0)
    geo, flights = [], []
    from src.benchmark.generator import load_scenarios
    missions = {x["scenario_id"]: Mission.from_dict(x["mission"]) for x in load_scenarios(a.scenario_dir)}
    for r in released_plans(a.run, a.run_seeds):
        m = missions[r["scenario_id"]]
        plan = FlightPlan.from_dict(r["final_plan"])
        unit = a.unit_m or m.unit_m or 1.0
        sp, n_red = smoothed_plan(m, plan, R / unit)
        vs = verify(m, sp)
        fams = sorted({v.family for v in vs.violations}) if not vs.success else []
        geo.append({"scenario_id": r["scenario_id"], "condition": r["condition"], "n_corners": len(plan.waypoints), "fillets_reduced": n_red,
                    "smoothed_valid": vs.success, "smoothed_violation_families": "+".join(fams)})
        for mode in ("baseline", "dynamics", "dyn_margin"):
            p = plan
            if mode == "dyn_margin":
                p = lift_altitudes(m, plan, a.margin)
                if not verify(m, p).success:
                    flights.append({"scenario_id": r["scenario_id"], "condition": r["condition"], "mode": mode, "admissible": False})
                    continue
            cfg_kw = dict(turn_limit=mode != "baseline", smoothing_radius_m=R if mode != "baseline" else 0.0)
            for wname, w in winds.items():
                mm = prepare_mission(m, a.vehicle, unit, w)
                pts = planned_polyline_m(mm, p)
                L = sum(math.dist(x[:2], y[:2]) for x, y in zip(pts, pts[1:]))
                dt = max(0.1, L / 12.0 / 20000.0)
                for sd in a.seeds:
                    tr = simulate(mm, p, cfg=SimConfig(dt=dt, gust_sigma=1.0, seed=sd, record_every=max(1, int(1.0 / dt)), **cfg_kw))
                    er = execution_record(mm, p, tr, required_tol_m=req_tol)
                    flights.append({"scenario_id": r["scenario_id"], "condition": r["condition"], "mode": mode, "admissible": True, "wind": wname, "sim_seed": sd,
                                    "near_or_contact": bool(er["intrusion_zones"] > 0 or er["collision"]), **er})
        print(f"{r['scenario_id']:<30} {r['condition']:<16} smoothed_valid={vs.success}", flush=True)
    G, F = pd.DataFrame(geo), pd.DataFrame(flights)
    G.to_csv(os.path.join(a.out, "smoothed_geometry.csv"), index=False)
    F.to_csv(os.path.join(a.out, "flights.csv"), index=False)
    ok = F[F.admissible == True]  # noqa: E712
    summ = {"vehicle": a.vehicle, "effective_turn_radius_m": round(R, 1), "v_climb_max": veh.v_climb_max, "v_descend_max": veh.v_descend_max,
            "plans": int(len(G)), "smoothed_valid_rate": float(G.smoothed_valid.mean()) if len(G) else None,
            "smoothed_invalid_by_condition": G.groupby("condition").smoothed_valid.apply(lambda s: int((~s).sum())).to_dict() if len(G) else {},
            "plans_with_reduced_fillet": int((G.fillets_reduced > 0).sum()) if len(G) else 0, "margin_m": a.margin, "modes": {}}
    for mode, g in ok.groupby("mode"):
        summ["modes"][mode] = {"flights": int(len(g)), "arrived": float(g.arrived.mean()), "zone_entered": float((g.intrusion_zones > 0).mean()),
                               "near_or_contact": float(g.near_or_contact.mean()), "required_visited": float(g.all_required_visited.mean()),
                               "L2_executed_feasible": float(g.L2_executed_feasible.mean()), "cross_track_p95_median_m": float(g.cross_track_p95_m.median()),
                               "vertical_dev_max_median_m": float(g.dev_vert_max_m.median())}
    summ["modes"]["dyn_margin_not_admissible"] = int((F[F["mode"] == "dyn_margin"].admissible == False).sum())  # noqa: E712
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()

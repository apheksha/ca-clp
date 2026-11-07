"""Consistent SIMULATION STUDY: every verified, released plan is flown under defined wind/gust scenarios and seeds, and the same measurable outcomes are recorded for every
plan and condition -- no animations, only numbers:

  planned vs actual   : deviation of the flown path from the planned polyline (mean / RMS / p95 / max horizontal and vertical), time vs planned time
  cross-track error   : controller's cross-track error statistics
  mission completion  : arrived AND every required waypoint visited AND order kept
  energy consumption  : battery energy used in flight (Wh, physical power model), predicted energy of the verifier's physical model, error %, remaining usable battery
  obstacle intrusions : zones entered by the flown path (count, first time, time inside, MAX PENETRATION DEPTH in metres), plus the simulator's collision event
  recovery            : see src/dynamic/online_experiment.py (detection delay, activation latency, recovery rate, extra distance/energy vs the event-free flight)

Claim levels (docs/SAFETY_CLAIMS.md): L1 = geometric route feasibility (verifier), L2 = simulated execution feasibility (this module), L3 = actual flight safety (NOT assessed here:
`L3_flight_safety` is always the string 'not assessed').
"""
from __future__ import annotations
import csv
import glob
import json
import math
import os
from collections import defaultdict
from src.models.schemas import Mission, FlightPlan
from src.models.energy import plan_energy
from src.simulator.sim import simulate, SimConfig, TrajectoryResult
from src.simulator.evaluation import evaluate_execution
from src.simulator.plots import plot_planned_vs_actual
from src.simulator.sensors import _dist_to_zone_m
from src.evaluation.report import wilson
from src.utils.io import read_jsonl

DEFAULT_WINDS = {"calm": (0.0, 0.0), "crosswind": (0.0, 4.0)}


def planned_polyline_m(mission: Mission, plan: FlightPlan):
    u = mission.unit_m
    alts = [mission.start_altitude] + [w.altitude for w in plan.ordered()] + [plan.destination.altitude]
    return [(p[1][0] * u, p[1][1] * u, a) for p, a in zip(plan.route_points(mission), alts)]


def deviation_stats(planned: list, actual: list) -> dict:
    """Distance of every flown sample to the planned 3D polyline (horizontal, vertical, 3D)."""
    h, v, d3 = [], [], []
    for p in actual:
        bh = bv = b3 = float("inf")
        for a, b in zip(planned, planned[1:]):
            ab = [b[i] - a[i] for i in range(3)]
            L2 = sum(x * x for x in ab)
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, sum((p[i] - a[i]) * ab[i] for i in range(3)) / L2))
            q = [a[i] + t * ab[i] for i in range(3)]
            dd = math.dist(p, q)
            if dd < b3:
                b3, bh, bv = dd, math.hypot(p[0] - q[0], p[1] - q[1]), abs(p[2] - q[2])
        h.append(bh); v.append(bv); d3.append(b3)

    def q95(x):
        s = sorted(x)
        return s[min(len(s) - 1, int(0.95 * len(s)))] if s else None
    n = max(len(d3), 1)
    return {"dev_mean_m": sum(d3) / n, "dev_rms_m": math.sqrt(sum(x * x for x in d3) / n), "dev_p95_m": q95(d3), "dev_max_m": max(d3) if d3 else None,
            "dev_horiz_max_m": max(h) if h else None, "dev_vert_max_m": max(v) if v else None}


def intrusion_stats(mission: Mission, traj: TrajectoryResult, zones=None) -> dict:
    """Zones entered by the flown samples: count, first time, time inside, max penetration depth (m)."""
    zs = list(zones if zones is not None else (mission.no_fly_zones if mission.is_active("no_fly_zone") else []))
    u = mission.unit_m
    per = {}
    for z in zs:
        depth, first, inside_n = 0.0, None, 0
        for k, p in enumerate(traj.pos):
            if not z.vertical_overlap(p[2], p[2]):
                continue
            pp = (p[0] / u, p[1] / u)
            if not z.contains(pp):
                continue
            if z.kind == "circle":
                d = (z.radius - math.dist(pp, z.centroid())) * u
            else:
                from src.utils import geometry as G
                d = G.min_dist_to_polygon_boundary(pp, z._verts()) * u
            depth = max(depth, d)
            inside_n += 1
            if first is None:
                first = traj.t[k]
        if inside_n:
            dt = (traj.t[1] - traj.t[0]) if len(traj.t) > 1 else 0.0
            per[z.id] = {"first_t": first, "time_inside_s": round(inside_n * dt, 2), "max_depth_m": round(depth, 2)}
    return {"n_zones_entered": len(per), "max_penetration_m": max((v["max_depth_m"] for v in per.values()), default=0.0),
            "first_intrusion_t": min((v["first_t"] for v in per.values()), default=None), "time_inside_s": sum(v["time_inside_s"] for v in per.values()), "zones": per}


def execution_record(mission: Mission, plan: FlightPlan, traj: TrajectoryResult, truth_zones=None, margin_m: float = 1.0, required_tol_m: float | None = None) -> dict:
    """All measurable outcomes of ONE flight, in one flat dict (the same schema for every condition / scenario / seed)."""
    ev = evaluate_execution(mission, plan, traj, truth_zones=truth_zones, margin_m=margin_m, required_tol_m=required_tol_m)
    dev = deviation_stats(planned_polyline_m(mission, plan), traj.pos)
    intr = intrusion_stats(mission, traj, truth_zones)
    pm = Mission.from_dict({**mission.to_dict(), "energy_model": "physical", "max_energy": 1e9}) if mission.vehicle else None
    pe = plan_energy(pm, plan) if pm is not None else None
    e_used = traj.energy_used_wh
    e_pred = pe["energy_wh"] if pe else None
    ct = traj.cross_track_m
    srt = sorted(ct)
    veh_usable = pe["usable_wh"] if pe else None
    rec = {
        "L1_geometric_verified": True,                       # selection criterion: only verifier-accepted plans are flown
        "L2_executed_feasible": bool(ev["executed_feasible"]),
        "L3_flight_safety": "not assessed",
        "status": traj.status, "arrived": ev["arrived"], "all_required_visited": all(ev["required_visited"].values()), "order_ok": ev["order_ok"],
        "mission_completed": bool(ev["arrived"] and all(ev["required_visited"].values()) and ev["order_ok"] and not intr["n_zones_entered"] and traj.collision is None),
        "duration_s": traj.duration_s, "planned_time_s": (pe["time_s"] if pe else None), "distance_flown_m": ev["distance_flown_m"],
        "cross_track_mean_m": (sum(ct) / len(ct)) if ct else None, "cross_track_rms_m": math.sqrt(sum(c * c for c in ct) / len(ct)) if ct else None,
        "cross_track_p95_m": srt[min(len(srt) - 1, int(0.95 * len(srt)))] if srt else None, "cross_track_max_m": max(ct) if ct else None,
        **dev,
        "energy_used_wh": e_used, "energy_predicted_wh": e_pred, "energy_error_pct": (100.0 * (e_used - e_pred) / e_pred) if e_pred else None,
        "battery_remaining_wh": (veh_usable - e_used) if veh_usable is not None else None, "battery_ok": (veh_usable is None) or e_used <= veh_usable,
        "intrusion_zones": intr["n_zones_entered"], "intrusion_max_penetration_m": intr["max_penetration_m"], "intrusion_first_t": intr["first_intrusion_t"],
        "intrusion_time_inside_s": intr["time_inside_s"], "collision": bool(traj.collision), "collision_kind": (traj.collision or {}).get("kind"),
    }
    return rec


def lift_altitudes(mission: Mission, plan: FlightPlan, margin_m: float) -> FlightPlan:
    """Execution margin (v12): move every waypoint altitude into [h_min + d, h_max - d] with d = min(margin, band/2). The verifier only checks the waypoints, so a plan that
    sits exactly on the floor is 'valid' but is left with no room for vertical tracking / GNSS error. The lifted plan must be RE-VERIFIED (climb costs energy)."""
    from src.models.schemas import Waypoint
    lo, hi = mission.min_altitude, mission.max_altitude
    d = min(margin_m, max(hi - lo, 0.0) / 2.0)
    clamp = lambda a: min(max(a, lo + d), hi - d)
    wps = [Waypoint(w.id, w.x, w.y, clamp(w.altitude), w.order) for w in plan.ordered()]
    dst = plan.destination
    return FlightPlan(plan.mission_id, wps, Waypoint(dst.id, dst.x, dst.y, clamp(dst.altitude), dst.order)).validate()


def simulate_run_study(run_dir: str, vehicle="generic_quad_large", unit_m: float | None = None, winds: dict | None = None, gust_sigma: float = 1.0, seeds=(0,),
                       conditions: list | None = None, limit: int | None = None, out_dir: str | None = None, save_trajectories: int = 30, plots: int = 6,
                       run_seeds: list | None = None, altitude_margin_m: float = 0.0) -> str:
    """run_seeds: fly only records of these EXPERIMENT seeds (None = all); independent of the simulation seeds `seeds`.
    altitude_margin_m > 0: fly the margin-lifted version of each plan (lift_altitudes), only if it still passes the verifier; otherwise the plan is recorded as
    'margin_not_admissible' and not flown."""
    from src.verifier.verifier import verify
    from src.simulator.batch import prepare_mission
    winds = winds or DEFAULT_WINDS
    recs = read_jsonl(os.path.join(run_dir, "runs.jsonl"))
    out_dir = out_dir or os.path.join(run_dir, "simulation")
    os.makedirs(os.path.join(out_dir, "trajectories"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    rows, n_plans, n_saved, n_plots = [], 0, 0, 0
    n_unverified_skipped = 0
    margin_rejected: list = []
    for r in recs:
        fv = r.get("final_verification")
        if conditions and r["condition"] not in conditions:
            continue
        if run_seeds is not None and r.get("seed") not in run_seeds:
            continue
        if not (r.get("final_plan") and fv and fv.get("success") and r["metrics"].get("released")):
            n_unverified_skipped += 1          # not L1-verified: never flown (reported, not hidden)
            continue
        mj = (r.get("dynamic") or {}).get("mission")
        if mj is None:
            f = next(iter(glob.glob(os.path.join("benchmark", "*", r["scenario_id"] + ".json"))), None)
            if f is None:
                continue
            mj = json.load(open(f))["mission"]
        base = Mission.from_dict(mj)
        u = unit_m if unit_m is not None else (1.0 if "real" in r["scenario_id"] else 20.0)
        plan = FlightPlan.from_dict(r["final_plan"])
        if altitude_margin_m > 0:
            lifted = lift_altitudes(base, plan, altitude_margin_m)
            if not verify(base, lifted).success:
                margin_rejected.append(r["run_id"])
                n_plans += 1
                if limit and n_plans >= limit:
                    break
                continue
            plan = lifted
        req_tol = None
        for wname, w in winds.items():
            m = prepare_mission(base, vehicle, u, w)
            if req_tol is None:   # a waypoint counts as visited when the true path passes within 1.5x the vehicle's own acceptance radius (the controller cuts corners by design)
                from src.models.vehicle import get_vehicle
                req_tol = max(1.5 * get_vehicle(vehicle).acceptance_radius_m, 5.0)
            L = sum(math.dist(a[:2], b[:2]) for a, b in zip(planned_polyline_m(m, plan), planned_polyline_m(m, plan)[1:]))
            dt = max(0.1, L / 12.0 / 20000.0)
            for sd in seeds:
                tr = simulate(m, plan, cfg=SimConfig(dt=dt, gust_sigma=gust_sigma, seed=sd, record_every=max(1, int(1.0 / dt))))
                er = execution_record(m, plan, tr, required_tol_m=req_tol)
                row = {"run_id": r["run_id"], "scenario_id": r["scenario_id"], "condition": r["condition"], "wind": wname, "wind_east_north": list(w), "gust_sigma": gust_sigma, "sim_seed": sd,
                       "altitude_margin_m": altitude_margin_m, **er}
                rows.append(row)
                tag = f"{r['run_id']}__{wname}__s{sd}"
                if n_saved < save_trajectories:
                    from src.simulator.evaluation import save_trajectory_csv
                    save_trajectory_csv(tr, os.path.join(out_dir, "trajectories", tag + ".csv"))
                    n_saved += 1
                if n_plots < plots and (not er["mission_completed"] or n_plots < 2):
                    plot_planned_vs_actual(m, plan, tr, os.path.join(out_dir, "plots", tag + ".png"), title=f"{r['scenario_id']} | {r['condition']} | {wname} | {tr.status}")
                    n_plots += 1
        n_plans += 1
        if limit and n_plans >= limit:
            break
    if rows:
        with open(os.path.join(out_dir, "executions.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    agg = defaultdict(list)
    for x in rows:
        agg[(x["condition"], x["wind"])].append(x)
    summ = []
    num = lambda g, k: [x[k] for x in g if x[k] is not None]
    for (cond, wn), g in sorted(agg.items()):
        n = len(g)
        k_c, k_l2 = sum(x["mission_completed"] for x in g), sum(x["L2_executed_feasible"] for x in g)
        k_i = sum(x["intrusion_zones"] > 0 or x["collision"] for x in g)
        lo, hi = wilson(k_c, n)
        mean = lambda v: (sum(v) / len(v)) if v else None
        summ.append({"condition": cond, "wind": wn, "n_flights": n, "mission_completion_rate": k_c / n, "completion_ci_lo": lo, "completion_ci_hi": hi, "L2_executed_feasible_rate": k_l2 / n,
                     "intrusion_rate": k_i / n, "max_penetration_m_max": max(num(g, "intrusion_max_penetration_m"), default=0.0), "cross_track_p95_m_mean": mean(num(g, "cross_track_p95_m")),
                     "cross_track_max_m_max": max(num(g, "cross_track_max_m"), default=None), "dev_p95_m_mean": mean(num(g, "dev_p95_m")), "energy_used_wh_mean": mean(num(g, "energy_used_wh")),
                     "energy_error_pct_mean": mean(num(g, "energy_error_pct")), "battery_depleted": sum(x["status"] == "battery_depleted" for x in g), "collisions": sum(x["collision"] for x in g)})
    if summ:
        with open(os.path.join(out_dir, "execution_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summ[0]))
            w.writeheader()
            w.writerows(summ)
    gap = {"n_flights": len(rows), "L1_verified": len(rows), "L2_executed_feasible": sum(x["L2_executed_feasible"] for x in rows),
           "verified_but_not_executable": sum(not x["L2_executed_feasible"] for x in rows),
           "reasons": {k: sum(1 for x in rows if not x["L2_executed_feasible"] and cond(x, k)) for k, cond in (
               ("zone_intrusion_or_collision", lambda x, k: x["intrusion_zones"] > 0 or x["collision"]), ("battery_depleted", lambda x, k: x["status"] == "battery_depleted"),
               ("not_arrived_or_waypoint_missed", lambda x, k: not x["arrived"] or not x["all_required_visited"]), ("order", lambda x, k: not x["order_ok"]))},
           "plans_skipped_not_L1_verified": n_unverified_skipped, "L3_flight_safety": "not assessed",
           "altitude_margin_m": altitude_margin_m, "plans_margin_not_admissible": len(margin_rejected), "margin_not_admissible_run_ids": margin_rejected[:50],
           "note": "L1 = verifier-accepted geometry; L2 = executed in the built-in simulator (uncalibrated vehicle profile); L3 (real flight safety) is outside this repository's evidence."}
    json.dump(gap, open(os.path.join(out_dir, "claim_gap.json"), "w"), indent=1)
    json.dump({"vehicle": vehicle if isinstance(vehicle, str) else dict(vehicle), "winds": {k: list(v) for k, v in winds.items()}, "gust_sigma": gust_sigma, "seeds": list(seeds),
               "run_seeds": None if run_seeds is None else list(run_seeds), "conditions": conditions, "limit": limit, "altitude_margin_m": altitude_margin_m,
               "required_waypoint_tolerance": "1.5 x vehicle acceptance radius (min 5 m)", "note": "built-in kinematic simulator, representative vehicle parameters; NOT PX4/Gazebo, not calibrated"}, open(os.path.join(out_dir, "study_config.json"), "w"), indent=1)
    return out_dir

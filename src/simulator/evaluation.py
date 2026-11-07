"""Evaluation of EXECUTED trajectories against the mission (not just the plan): tracking error, continuous 3D zone/terrain checks on the flown path, altitude band,
energy and distance actually used vs. the verifier's prediction, required-waypoint visits and a single `executed_feasible` verdict.

This is the bridge between 'verified plan' and 'what the vehicle actually did' (simulated here; the same function evaluates PX4-SITL logs converted by
src/simulator/px4_sitl.py)."""
from __future__ import annotations
import csv
import math
from src.models.schemas import Mission, FlightPlan
from src.simulator.sim import TrajectoryResult, _inflate_zone
from src.utils import geometry as G


def min_distance_to_zones_m(mission: Mission, pos_m, zones=None) -> float:
    from src.simulator.sensors import _dist_to_zone_m
    zs = zones if zones is not None else (mission.no_fly_zones if mission.is_active("no_fly_zone") else [])
    return min((_dist_to_zone_m(z, pos_m, mission.unit_m) for z in zs), default=float("inf"))


def evaluate_execution(mission: Mission, plan: FlightPlan, traj: TrajectoryResult, truth_zones=None, margin_m: float = 0.0, required_tol_m: float | None = None) -> dict:
    """truth_zones: zones that really existed (defaults to the mission's). margin_m: vehicle/safety radius for the continuous zone check."""
    u = mission.unit_m
    zones = list(truth_zones if truth_zones is not None else (mission.no_fly_zones if mission.is_active("no_fly_zone") else []))
    infl = [_inflate_zone(z, margin_m / u) for z in zones]
    P = traj.pos
    out = {"status": traj.status, "n_samples": len(P), "duration_s": traj.duration_s}
    ct = traj.cross_track_m
    out["tracking"] = {"max_cross_track_m": max(ct) if ct else None, "mean_cross_track_m": (sum(ct) / len(ct)) if ct else None,
                       "rms_cross_track_m": math.sqrt(sum(c * c for c in ct) / len(ct)) if ct else None}
    # continuous zone check on the flown polyline (3D prisms, closed sets)
    zone_hits = {}
    for i in range(len(P) - 1):
        a, b = (P[i][0] / u, P[i][1] / u), (P[i + 1][0] / u, P[i + 1][1] / u)
        for z in infl:
            if z.id not in zone_hits and z.intersects_segment_3d(a, b, P[i][2], P[i + 1][2]):
                zone_hits[z.id] = round(traj.t[i + 1], 2)
    out["zone_intrusions"] = zone_hits
    out["min_zone_distance_m"] = min((min_distance_to_zones_m(mission, p, zones) for p in P[::max(1, len(P) // 400)]), default=float("inf")) if zones else None
    # altitude band
    if mission.is_active("altitude"):
        d_end = (mission.destination[0] * u, mission.destination[1] * u)
        first_in = next((i for i, p in enumerate(P) if p[2] >= mission.min_altitude - 1.0), len(P))     # take-off climb is exempt until the band is first reached
        cruise = [p[2] for p in P[first_in:] if math.hypot(p[0] - d_end[0], p[1] - d_end[1]) > 60.0] or [p[2] for p in P]   # and so is the final descent
        out["altitude"] = {"min": min(cruise), "max": max(cruise), "band_violation_m": max(0.0, mission.min_altitude - min(cruise), max(cruise) - mission.max_altitude),
                           "exempt": "take-off climb until the band is first reached; last 60 m before the destination"}
    # terrain
    g = mission.terrain_grid
    if g is not None:
        ex = mission.terrain_exempt_m
        s0, d0 = (mission.start[0] * u, mission.start[1] * u), (mission.destination[0] * u, mission.destination[1] * u)
        agls = [p[2] - g.height_at(p[0] / u, p[1] / u) for p in P]
        agls_chk = [a for a, p in zip(agls, P) if not (ex > 0 and (math.hypot(p[0] - s0[0], p[1] - s0[1]) <= ex or math.hypot(p[0] - d0[0], p[1] - d0[1]) <= ex))] or agls
        out["terrain"] = {"min_agl_m": min(agls), "max_agl_m": max(agls), "clearance_violation": min(agls_chk) < mission.min_agl - 1e-6 - 2.0,   # 2 m tracking allowance at take-off/landing
                          "above_max_agl": mission.max_agl is not None and max(agls) > mission.max_agl + 1e-6}
    # resources actually used vs predicted
    dist_m = sum(math.dist(P[i], P[i + 1]) for i in range(len(P) - 1))
    out["distance_flown_m"] = dist_m
    out["distance_budget_ok"] = (not mission.is_active("distance")) or dist_m <= mission.max_distance * u * 1.02    # 2 % allowance: the flown path is smooth/longer than the polyline only by tracking
    e = traj.energy_used_wh
    out["energy_used_wh"] = e
    if mission.energy_model == "physical":
        out["energy_predicted_wh"] = traj.planned_energy_wh
        out["energy_model_error_pct"] = None if not traj.planned_energy_wh else 100.0 * (e - traj.planned_energy_wh) / traj.planned_energy_wh
        out["energy_budget_ok"] = (not mission.is_active("energy")) or mission.max_energy is None or e <= mission.max_energy
    # required waypoints visited (3D proximity of the TRUE path) in order
    tol = required_tol_m if required_tol_m is not None else max(mission.position_tolerance * u, 5.0)
    idx = {}
    for r in mission.required_waypoints:
        rp = (r.x * u, r.y * u)
        hits = [k for k, p in enumerate(P) if math.hypot(p[0] - rp[0], p[1] - rp[1]) <= tol]
        idx[r.id] = hits[0] if hits else None
    out["required_visited"] = {k: v is not None for k, v in idx.items()}
    order_ok = True
    if mission.is_active("ordering"):
        for a, b in mission.ordering:
            if idx.get(a) is None or idx.get(b) is None or idx[a] > idx[b]:
                order_ok = False
    out["order_ok"] = order_ok
    out["arrived"] = traj.status == "arrived"
    out["executed_feasible"] = bool(out["arrived"] and not zone_hits and out["distance_budget_ok"] and all(out["required_visited"].values()) and order_ok
                                    and out.get("energy_budget_ok", True) and not (g is not None and (out["terrain"]["clearance_violation"] or out["terrain"]["above_max_agl"]))
                                    and (not mission.is_active("altitude") or out["altitude"]["band_violation_m"] <= 3.0))
    return out


def save_trajectory_csv(traj: TrajectoryResult, path: str) -> str:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_m", "y_m", "z_m", "est_x", "est_y", "est_z", "vx", "vy", "vz", "power_w", "energy_wh", "cross_track_m"])
        for i in range(len(traj.t)):
            w.writerow([traj.t[i], *traj.pos[i], *traj.est[i], *traj.vel[i], traj.power_w[i], traj.energy_wh[i], traj.cross_track_m[i]])
    return path

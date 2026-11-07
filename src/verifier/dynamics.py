"""Post-hoc FLIGHT-DYNAMICS feasibility screen for plans that already passed the geometric verifier.

Why: a geometrically valid polyline is not necessarily a trajectory a vehicle can fly (reviewer point 4.B). This module does NOT replace a physics simulator
(PX4 SITL / Gazebo: see src/simulator/export_px4.py for a .plan export). It adds cheap, transparent necessary checks with explicit vehicle limits:

  1. climb / descent rate : |delta altitude| / (leg length * unit_m / cruise_speed)  <= max_climb_rate / max_descent_rate   [m/s]
  2. turn feasibility     : at every interior waypoint the corner is replaced by a circular fillet of radius R = min_turn_radius. A fillet needs
                            tangent length d = R*tan(theta/2) (theta = deflection angle); two neighbouring fillets must fit on the shared leg: d_i + d_j <= leg length
  3. fillet vs. zones     : the fillet arcs (sampled) must not enter a zone that the polyline itself avoids (corner cutting can clip a zone: tracking error proxy)
  4. endurance            : total time = path length / cruise_speed (+ climb time) <= max_flight_time_s (when given)
  5. tracking margin      : maximum deviation of the fillet from the polyline corner, R*(1/cos(theta/2) - 1), reported per plan

Units: map coordinates are multiplied by `unit_m` (metres per map unit) before comparison with vehicle limits.
Limits are explicit inputs (VehicleLimits); defaults describe a small multirotor/VTOL class vehicle and are NOT calibrated to any aircraft.
"""
from __future__ import annotations
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from src.models.schemas import Mission, FlightPlan
from src.utils import geometry as G


@dataclass
class VehicleLimits:
    cruise_speed: float = 12.0          # m/s
    max_climb_rate: float = 3.0         # m/s
    max_descent_rate: float | None = None   # m/s; None -> same as climb
    min_turn_radius: float = 30.0       # m (horizontal)
    max_flight_time_s: float | None = None
    arc_samples: int = 12


def _unit(a: G.Point, b: G.Point):
    d = G.dist(a, b)
    return ((b[0] - a[0]) / d, (b[1] - a[1]) / d) if d > 0 else (0.0, 0.0)


def check_dynamics(mission: Mission, plan: FlightPlan, lim: VehicleLimits, unit_m: float = 1.0) -> dict:
    pts = plan.route_points(mission)
    xy = [p for _, p in pts]
    alts = [mission.start_altitude] + [w.altitude for w in plan.ordered()] + [plan.destination.altitude]
    leg = [G.dist(xy[i], xy[i + 1]) * unit_m for i in range(len(xy) - 1)]
    out = {"n_legs": len(leg), "path_length_m": sum(leg), "climb_rate_violations": 0, "max_climb_rate_obs": 0.0, "turn_violations": 0, "fillet_zone_violations": 0,
           "max_tracking_deviation_m": 0.0, "max_deflection_deg": 0.0, "flight_time_s": 0.0, "endurance_violation": False}
    descent = lim.max_descent_rate if lim.max_descent_rate is not None else lim.max_climb_rate
    t_total = 0.0
    for i, L in enumerate(leg):
        t = L / lim.cruise_speed if L > 0 else 0.0
        dz = alts[i + 1] - alts[i]
        t_total += t
        if t > 0:
            rate = abs(dz) / t
            out["max_climb_rate_obs"] = max(out["max_climb_rate_obs"], rate)
            if (dz > 0 and rate > lim.max_climb_rate + 1e-9) or (dz < 0 and rate > descent + 1e-9):
                out["climb_rate_violations"] += 1
        elif abs(dz) > 1e-9:
            out["climb_rate_violations"] += 1      # vertical jump over zero horizontal distance
    tang = [0.0] * len(xy)
    for i in range(1, len(xy) - 1):
        u1, u2 = _unit(xy[i - 1], xy[i]), _unit(xy[i], xy[i + 1])
        if u1 == (0.0, 0.0) or u2 == (0.0, 0.0):
            continue
        cosd = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        theta = math.acos(cosd)                  # deflection angle (0 = straight on)
        out["max_deflection_deg"] = max(out["max_deflection_deg"], math.degrees(theta))
        if theta < 1e-6:
            continue
        if theta > math.pi - 1e-6:               # U-turn: no finite fillet; treat as violation
            out["turn_violations"] += 1
            continue
        R = lim.min_turn_radius
        d = R * math.tan(theta / 2)
        tang[i] = d
        out["max_tracking_deviation_m"] = max(out["max_tracking_deviation_m"], R * (1 / math.cos(theta / 2) - 1))
    for i in range(len(leg)):
        if tang[i] + tang[i + 1] > leg[i] + 1e-9:
            out["turn_violations"] += 1
    # fillet arcs vs. zones (2D, zones that apply at the corner altitude)
    zones = [z for z in mission.no_fly_zones] if mission.is_active("no_fly_zone") else []
    for i in range(1, len(xy) - 1):
        if tang[i] <= 0:
            continue
        P = xy[i]
        u1, u2 = _unit(xy[i - 1], P), _unit(P, xy[i + 1])
        d_map = tang[i] / unit_m
        T1 = (P[0] - u1[0] * d_map, P[1] - u1[1] * d_map)
        T2 = (P[0] + u2[0] * d_map, P[1] + u2[1] * d_map)
        # quadratic Bezier through T1,P,T2 approximates the fillet arc and lies inside the corner triangle (never leaves the corner hull)
        arc = [T1]
        for k in range(1, lim.arc_samples):
            t = k / lim.arc_samples
            arc.append(((1 - t) ** 2 * T1[0] + 2 * (1 - t) * t * P[0] + t * t * T2[0], (1 - t) ** 2 * T1[1] + 2 * (1 - t) * t * P[1] + t * t * T2[1]))
        arc.append(T2)
        a_i = alts[i]
        for z in zones:
            if z.vertical_overlap(a_i, a_i) and any(z.intersects_segment(arc[k], arc[k + 1]) for k in range(len(arc) - 1)):
                # only a NEW violation if the sharp polyline corner itself was clear of this zone
                if not z.contains(P):
                    out["fillet_zone_violations"] += 1
                    break
    out["flight_time_s"] = t_total
    if lim.max_flight_time_s is not None and t_total > lim.max_flight_time_s:
        out["endurance_violation"] = True
    out["dynamically_feasible"] = (out["climb_rate_violations"] == 0 and out["turn_violations"] == 0 and out["fillet_zone_violations"] == 0 and not out["endurance_violation"])
    return out


def evaluate_run_dynamics(run_dir: str, lim: VehicleLimits, unit_m: float = 1.0, out_dir: str | None = None) -> str:
    """Screen every RELEASED + successful final plan of a raw run. Writes dynamics_report.csv (per run) and dynamics_summary.json (per condition)."""
    from src.utils.io import read_jsonl
    recs = read_jsonl(os.path.join(run_dir, "runs.jsonl"))
    out_dir = out_dir or os.path.join(run_dir, "dynamics")
    os.makedirs(out_dir, exist_ok=True)
    rows, summ = [], {}
    for r in recs:
        fv = r.get("final_verification")
        if not (r.get("final_plan") and fv and fv.get("success") and r["metrics"].get("released")):
            continue
        mdict = r["dynamic"]["mission"] if "dynamic" in r else None
        if mdict is None:
            import glob
            f = next(iter(glob.glob(os.path.join("benchmark", "*", r["scenario_id"] + ".json"))), None)
            if f is None:
                continue
            mdict = json.load(open(f))["mission"]
        res = check_dynamics(Mission.from_dict(mdict), FlightPlan.from_dict(r["final_plan"]), lim, unit_m)
        rows.append({"run_id": r["run_id"], "scenario_id": r["scenario_id"], "condition": r["condition"], "seed": r["seed"], **res})
        s = summ.setdefault(r["condition"], {"n": 0, "dynamically_feasible": 0})
        s["n"] += 1
        s["dynamically_feasible"] += int(res["dynamically_feasible"])
    if rows:
        with open(os.path.join(out_dir, "dynamics_report.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    json.dump({"limits": asdict(lim), "unit_m": unit_m, "per_condition": summ,
               "note": "necessary checks only; not a substitute for PX4 SITL/Gazebo execution"}, open(os.path.join(out_dir, "dynamics_summary.json"), "w"), indent=1)
    return out_dir

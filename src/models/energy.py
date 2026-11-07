"""PHYSICAL energy model (battery energy in Wh) for a vehicle profile, with wind, payload, speed, climb/descent and per-waypoint hover.

Replaces the proxy E = L + k_v*A + k_wp*n when `Mission.energy_model == "physical"`; the proxy stays the default so earlier results remain reproducible.

Model (documented assumptions, none calibrated to a specific airframe -- see config/vehicles.yaml):
  mass          m = vehicle.mass_kg + payload_kg ;  thrust T = m g
  multirotor    P_ind(v) = T v_h r(v) / eta,   v_h = sqrt(T / (2 rho A)),  r = sqrt(sqrt(1 + (mu^4)/4) - mu^2/2),  mu = v / v_h   (momentum theory, forward flight)
                P_par(v) = 0.5 rho CdA v^3 / eta
  fixed wing    P(v)    = m g v / ((L/D) eta)
  climb         P_climb = T * max(vz, 0) / eta     (descent recovers no energy: conservative)
  avionics      constant P_av
  wind          steady wind w = (east, north) [m/s]; the vehicle holds airspeed v_cruise and crabs so that its ground track follows the leg:
                ground speed g = w.u + sqrt(v_cruise^2 - |w_perp|^2); a leg with |w_perp| >= v_cruise or g < 0.5 m/s is NOT FLYABLE (reported as an energy violation)
  leg time      t = max(d / g, climb_up / v_climb_max, descent / v_descend_max);   if the vertical rate limits the leg the horizontal speed is scaled down
  waypoint      hover for hover_time_per_waypoint_s at P_hover on every intermediate waypoint
  budget        Mission.max_energy [Wh]; the helper `battery_budget_wh(vehicle)` = battery_wh * usable_fraction.
Energy lower bound for ANY airspeed policy (used by the feasibility oracle): E >= L * min_v P_level(v)/(v + |w|) + T * ascent_min / eta."""
from __future__ import annotations
import math
from src.models.vehicle import VehicleProfile, RHO, G0, get_vehicle


def total_mass(v: VehicleProfile, payload_kg: float = 0.0) -> float:
    return v.mass_kg + max(0.0, payload_kg)


def power_level_w(v: VehicleProfile, airspeed: float, payload_kg: float = 0.0) -> float:
    """Electrical power in level flight at the given airspeed (m/s), including avionics."""
    m = total_mass(v, payload_kg)
    T = m * G0
    if v.kind == "fixed_wing":
        return m * G0 * max(airspeed, 0.0) / (v.lift_to_drag * v.efficiency) + v.avionics_w
    v_h = math.sqrt(T / (2 * RHO * v.rotor_disc_area_m2))
    mu = max(airspeed, 0.0) / v_h
    r = math.sqrt(math.sqrt(1 + mu ** 4 / 4) - mu ** 2 / 2)
    p_ind = T * v_h * r / v.efficiency
    p_par = 0.5 * RHO * v.drag_area_m2 * max(airspeed, 0.0) ** 3 / v.efficiency
    return p_ind + p_par + v.avionics_w


def power_w(v: VehicleProfile, airspeed: float, climb_rate: float, payload_kg: float = 0.0) -> float:
    m = total_mass(v, payload_kg)
    return power_level_w(v, airspeed, payload_kg) + m * G0 * max(climb_rate, 0.0) / v.efficiency


def hover_power_w(v: VehicleProfile, payload_kg: float = 0.0) -> float:
    return power_level_w(v, 0.0, payload_kg) if v.kind == "multirotor" else power_level_w(v, v.v_cruise, payload_kg)


def battery_budget_wh(v: VehicleProfile) -> float:
    return v.usable_wh


def ground_speed(v: VehicleProfile, wind, ux: float, uy: float, airspeed: float | None = None) -> float | None:
    """Ground speed along unit track (ux, uy) when holding `airspeed` (default cruise) in steady wind; None if the leg cannot be flown."""
    va = v.v_cruise if airspeed is None else airspeed
    wx, wy = (wind or (0.0, 0.0))
    w_par = wx * ux + wy * uy
    w_perp2 = max(0.0, wx * wx + wy * wy - w_par * w_par)
    if va * va <= w_perp2:
        return None
    g = w_par + math.sqrt(va * va - w_perp2)
    return g if g >= 0.5 else None


def plan_energy(mission, plan) -> dict:
    """Energy / time of a plan under the mission's physical model. Returns totals, per-leg detail, state-of-charge profile and non-flyable legs."""
    v = get_vehicle(mission.vehicle)
    pts = plan.route_points(mission)
    alts = [mission.start_altitude] + [w.altitude for w in plan.ordered()] + [plan.destination.altitude]
    payload, wind, unit = mission.payload_kg, mission.wind, mission.unit_m
    legs, total_wh, total_t, bad = [], 0.0, 0.0, []
    for i in range(len(pts) - 1):
        (na, a), (nb, b) = pts[i], pts[i + 1]
        d = math.hypot(b[0] - a[0], b[1] - a[1]) * unit
        dz = alts[i + 1] - alts[i]
        up, dn = max(dz, 0.0), max(-dz, 0.0)
        t_vert = max(up / v.v_climb_max, dn / v.v_descend_max)
        if d < 1e-9:
            t, g, va = t_vert, 0.0, 0.0
        else:
            ux, uy = (b[0] - a[0]) * unit / d, (b[1] - a[1]) * unit / d
            g = ground_speed(v, wind, ux, uy)
            if g is None:
                bad.append({"from": na, "to": nb, "reason": "wind exceeds airspeed authority (or net ground speed < 0.5 m/s)"})
                legs.append({"from": na, "to": nb, "distance_m": d, "wh": float("inf"), "time_s": float("inf")})
                total_wh = float("inf")
                continue
            t_h = d / g
            t = max(t_h, t_vert)
            va = v.v_cruise * (t_h / t)
        vz = (dz / t) if t > 0 else 0.0
        wh = power_w(v, va, vz, payload) * t / 3600.0
        legs.append({"from": na, "to": nb, "distance_m": d, "ground_speed": g, "airspeed": va, "time_s": t, "climb_rate": vz, "wh": wh})
        total_wh += wh
        total_t += t
    hover_wh = len(plan.waypoints) * v.hover_time_per_waypoint_s * hover_power_w(v, payload) / 3600.0
    total_wh += hover_wh
    total_t += len(plan.waypoints) * v.hover_time_per_waypoint_s
    soc, acc = [], 0.0
    for lg in legs:
        acc += lg["wh"]
        soc.append(max(0.0, 1.0 - acc / v.battery_wh))
    return {"energy_wh": total_wh, "time_s": total_t, "legs": legs, "hover_wh": hover_wh, "non_flyable_legs": bad, "soc_after_leg": soc,
            "vehicle": v.name, "battery_wh": v.battery_wh, "usable_wh": v.usable_wh}


def remaining_battery_wh(mission, plan, leg_index: int, fraction_of_leg: float = 0.0, used_wh: float | None = None) -> float:
    """Estimated remaining USABLE battery (Wh) when the vehicle is `fraction_of_leg` into leg `leg_index` of `plan` (planned consumption), or given measured use."""
    v = get_vehicle(mission.vehicle)
    if used_wh is not None:
        return v.usable_wh - used_wh
    e = plan_energy(mission, plan)
    done = sum(l["wh"] for l in e["legs"][:leg_index]) + fraction_of_leg * (e["legs"][leg_index]["wh"] if leg_index < len(e["legs"]) else 0.0)
    return v.usable_wh - done


def energy_lower_bound_wh(mission, length_map_units: float, min_ascent_m: float = 0.0) -> float:
    """Lower bound of the battery energy needed to fly `length_map_units` of horizontal path (+ `min_ascent_m` of climb) for ANY airspeed policy.
    Level power is minimised over a fine airspeed grid and the ground speed is bounded by v + |wind|; a 0.2 % safety factor covers the grid resolution."""
    v = get_vehicle(mission.vehicle)
    wmag = math.hypot(*(mission.wind or (0.0, 0.0)))
    best = min(power_level_w(v, s / 10.0, mission.payload_kg) / (s / 10.0 + wmag) for s in range(5, int(v.v_max * 10) + 1))
    L = length_map_units * mission.unit_m
    climb = total_mass(v, mission.payload_kg) * G0 * max(min_ascent_m, 0.0) / v.efficiency / 3600.0
    return 0.998 * (best * L / 3600.0) + climb

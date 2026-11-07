"""Objective 4: controlled mid-flight environment changes with preserved mission state.

Pipeline: valid plan -> UAV flies `progress` of the route -> change (add/enlarge zone, tighten distance, reduce battery) -> new Mission built from the
CURRENT state (position, altitude, remaining required waypoints, remaining distance/energy) -> remaining plan re-verified -> repaired by the same
generate-verify-repair loop (or re-planned from scratch for the baseline)."""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from src.models.schemas import Mission, FlightPlan, Waypoint, NoFlyZone, RequiredWaypoint
from src.benchmark.feasibility import assess, shortest_route, visiting_order, extent
from src.benchmark.generator import nl_description
from src.verifier.verifier import verify, resources
from src.utils import geometry as G

CHANGES = ("add_zone", "enlarge_zone", "tighten_distance", "reduce_energy")


@dataclass
class ExecState:
    progress: float
    position: tuple
    altitude: float
    seg_index: int
    visited: list
    flown_distance: float
    energy_used: float
    remaining_waypoints: list = field(default_factory=list)   # Waypoint objects not yet reached (incl. extra detour points)

    def to_dict(self):
        return {"progress": self.progress, "position": list(self.position), "altitude": round(self.altitude, 3), "visited": self.visited,
                "flown_distance": round(self.flown_distance, 3), "energy_used": round(self.energy_used, 3),
                "remaining_waypoint_ids": [w.id for w in self.remaining_waypoints]}


def comfortable_plan(m: Mission) -> FlightPlan | None:
    """A valid but non-minimal plan (conservative clearance, mid-band altitude) so that later tightening can be satisfied by a better plan."""
    order = visiting_order(m)
    if order is None:
        return None
    pts = [m.start_xy] + [m.required(i).xy for i in order] + [m.dest_xy]
    base = max(0.5, extent(m) * 0.002)
    for mult in (6, 3, 1.5, 1):
        route = shortest_route(m, pts, clear=base * mult)
        if route is None:
            continue
        for frac in (0.5, 0.25, 0.0):
            alt = m.min_altitude + frac * (m.max_altitude - m.min_altitude) if m.is_active("altitude") else m.min_altitude
            req_at = {m.required(i).xy: i for i in order}
            wps, k = [], 0
            for p in route[1:-1]:
                if p in req_at:
                    wid = req_at[p]
                else:
                    k += 1
                    wid = f"W{k}"
                wps.append(Waypoint(wid, p[0], p[1], alt, len(wps) + 1))
            plan = FlightPlan(m.mission_id, wps, Waypoint("DEST", m.destination[0], m.destination[1], alt, len(wps) + 1)).validate()
            if verify(m, plan).success:
                return plan
    return None


def execution_state(m: Mission, plan: FlightPlan, progress: float) -> ExecState:
    pts = plan.route_points(m)
    alts = [m.start_altitude] + [w.altitude for w in plan.ordered()] + [plan.destination.altitude]
    seg = [G.dist(pts[i][1], pts[i + 1][1]) for i in range(len(pts) - 1)]
    total = sum(seg)
    target = progress * total
    cum, i = 0.0, 0
    while i < len(seg) - 1 and cum + seg[i] <= target:
        cum += seg[i]
        i += 1
    t = 0.0 if seg[i] == 0 else (target - cum) / seg[i]
    a, b = pts[i][1], pts[i + 1][1]
    pos = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    alt = alts[i] + t * (alts[i + 1] - alts[i])
    visited = [pts[j][0] for j in range(1, i + 1)]
    flown_alts = alts[:i + 1] + [alt]
    ascent = sum(max(0.0, y - x) for x, y in zip(flown_alts, flown_alts[1:]))
    energy = target + m.energy_k_v * ascent + m.energy_k_wp * len(visited)
    remaining = [w for w in plan.ordered() if w.id not in visited]
    return ExecState(progress, pos, alt, i, visited, target, energy, remaining)


def _scale_zone(z: NoFlyZone, f: float) -> NoFlyZone:
    z2 = copy.deepcopy(z)
    if z.kind == "circle":
        z2.radius = z.radius * f
    else:
        c = z.centroid()
        z2.vertices = [[c[0] + (v[0] - c[0]) * f, c[1] + (v[1] - c[1]) * f] for v in z.vertices]
    z2.description = (z.description + " (enlarged)").strip()
    return z2.validate()


def build_dynamic_mission(m: Mission, plan: FlightPlan, st: ExecState, change: str, rng: random.Random, fixed: dict | None = None):
    """Return (new_mission, remaining_plan, info) or None when the change cannot be applied sensibly.
    `fixed` (used for delayed detection) bypasses the change generator and rebuilds the replanning mission at state `st` from already decided
    zones / limits: {"zones", "active", "max_dist", "max_energy", "text", "info"}."""
    req_left = [r for r in m.required_waypoints if r.id not in st.visited]
    if not req_left:
        return None
    keep = {r.id for r in req_left}
    zones = copy.deepcopy(m.no_fly_zones)
    active = list(m.active)
    rem_pts = [st.position] + [w.xy for w in st.remaining_waypoints] + [plan.destination.xy]
    must_clear = [st.position, m.dest_xy] + [r.xy for r in req_left]
    ex = extent(m)
    max_dist = m.max_distance - st.flown_distance
    max_energy = None if m.max_energy is None else m.max_energy - st.energy_used
    info = {"change": change}
    text = ""
    if fixed is not None:
        zones, active, max_dist, max_energy, text = copy.deepcopy(fixed["zones"]), list(fixed["active"]), fixed["max_dist"], fixed["max_energy"], fixed["text"]
        info.update(fixed.get("info", {}))
    cur_rem = FlightPlan(m.mission_id + "_dyn", [Waypoint(w.id, w.x, w.y, w.altitude, i + 1) for i, w in enumerate(st.remaining_waypoints)],
                         Waypoint("DEST", plan.destination.x, plan.destination.y, plan.destination.altitude, len(st.remaining_waypoints) + 1))
    base = Mission(m.mission_id + "_dyn", "", [st.position[0], st.position[1]], list(m.destination), [RequiredWaypoint(r.id, r.x, r.y) for r in req_left],
                   m.min_altitude, m.max_altitude, max(max_dist, 1e-3), zones, [p for p in m.ordering if p[0] in keep and p[1] in keep],
                   m.position_tolerance, start_altitude=st.altitude, max_energy=max_energy, energy_k_v=m.energy_k_v, energy_k_wp=m.energy_k_wp,
                   speed_h=m.speed_h, speed_v=m.speed_v, active=active)
    cur = resources(cur_rem, base)

    if fixed is not None:
        pass
    elif change == "add_zone":
        segs = [(rem_pts[i], rem_pts[i + 1]) for i in range(len(rem_pts) - 1)]
        order = list(range(len(segs)))
        rng.shuffle(order)
        for i in order:
            a, b = segs[i]
            L = G.dist(a, b)
            if L < 1e-6:
                continue
            u = rng.uniform(0.4, 0.6)
            c = (a[0] + u * (b[0] - a[0]), a[1] + u * (b[1] - a[1]))
            r = min(0.07 * ex, 0.4 * L)
            z = NoFlyZone("ZN", "circle", center=[round(c[0], 3), round(c[1], 3)], radius=round(r, 3), description="newly announced restricted area").validate()
            if r > 0 and not any(z.contains(p) for p in must_clear) and all(G.dist(z.centroid(), q.centroid()) > 0 for q in zones):
                zones.append(z)
                if "no_fly_zone" not in active:
                    active.append("no_fly_zone")   # the verifier only enforces ACTIVE families; a new restricted area must be enforced
                info.update(zone_id="ZN", radius=z.radius)
                break
        else:
            return None
        text = "A new restricted area ZN has just been announced on the planned route."
    elif change == "enlarge_zone":
        if not zones:
            return None
        zs = sorted(zones, key=lambda z: min(G.closest_on_segment(z.centroid(), rem_pts[i], rem_pts[i + 1])[2] for i in range(len(rem_pts) - 1)))
        done = False
        for z in zs[:2]:
            for f in (1.8, 1.5, 1.25):
                z2 = _scale_zone(z, f)
                if not any(z2.contains(p) for p in must_clear):
                    zones[zones.index(z)] = z2
                    if "no_fly_zone" not in active:
                        active.append("no_fly_zone")
                    info.update(zone_id=z.id, scale=f)
                    text = f"Restricted area {z.id} has just been enlarged by a factor of {f}."
                    done = True
                    break
            if done:
                break
        if not done:
            return None
    elif change == "tighten_distance":
        if "distance" not in active:
            active.append("distance")
        probe = Mission(base.mission_id, "", base.start, base.destination, base.required_waypoints, base.min_altitude, base.max_altitude, 1e9, zones, base.ordering,
                        base.position_tolerance, start_altitude=base.start_altitude, active=[a for a in active if a != "energy"])
        f = assess(probe)
        if f.status != "FEASIBLE":
            return None
        max_dist = round(f.witness_distance * 1.03, 3)
        info.update(new_remaining_distance_limit=max_dist, current_remaining_distance=round(cur["distance"], 3))
        text = f"The distance allowance has been reduced: only {max_dist:g} units of route length remain."
    elif change == "reduce_energy":
        if "energy" not in active:
            active.append("energy")
        probe = Mission(base.mission_id, "", base.start, base.destination, base.required_waypoints, base.min_altitude, base.max_altitude, 1e9, zones, base.ordering,
                        base.position_tolerance, start_altitude=base.start_altitude, energy_k_v=base.energy_k_v, energy_k_wp=base.energy_k_wp,
                        active=[a for a in active if a != "energy"])
        f = assess(probe)
        if f.status != "FEASIBLE":
            return None
        e_min = resources(f.witness_plan, base)["energy"]
        if cur["energy"] <= e_min * 1.02:
            return None                       # no slack to remove (plan already at minimum energy)
        max_energy = round(e_min + 0.5 * (cur["energy"] - e_min), 3)
        info.update(new_remaining_energy_limit=max_energy, current_remaining_energy=round(cur["energy"], 3), minimum_energy=round(e_min, 3))
        text = f"Battery has dropped: only {max_energy:g} energy units remain for the rest of the mission."
    else:
        raise ValueError(change)

    new = Mission(base.mission_id, "", base.start, base.destination, base.required_waypoints, base.min_altitude, base.max_altitude, max(max_dist, 1e-3), zones,
                  base.ordering, base.position_tolerance, start_altitude=base.start_altitude, max_energy=max_energy, energy_k_v=base.energy_k_v,
                  energy_k_wp=base.energy_k_wp, speed_h=base.speed_h, speed_v=base.speed_v, active=active).validate()
    body = nl_description(new, "formal").replace("Fly from the base at", "Continue from the current position at").replace(
        "total horizontal route length at or below", "remaining horizontal route length at or below").replace("battery budget of", "remaining battery budget of")
    vis = ", ".join(st.visited) if st.visited else "nothing yet"
    new.description = (f"UPDATE DURING FLIGHT: the UAV is airborne at ({st.position[0]:.1f}, {st.position[1]:.1f}), altitude {st.altitude:.0f} m, "
                       f"and has already passed: {vis}. {text} Re-plan only the REMAINING route. " + body)
    return new, cur_rem, info


def build_delayed_dynamic_mission(m: Mission, plan: FlightPlan, st_change: ExecState, st_detect: ExecState, change: str, rng: random.Random):
    """DETECTION-LATENCY variant of the mid-flight change (survey / reviewer request: 'dynamic-event latency').
    The world changes when the UAV is at `st_change`, but the UAV only LEARNS of it at `st_detect` (further along the stale plan). Re-planning starts at
    `st_detect`. Besides the usual re-planning success, the case records `exposure`: whether the stale route flown during the blind interval
    already entered the changed zone (a safety event that no re-planner can undo).
    Returns (new_mission, remaining_plan, info) like build_dynamic_mission, or None."""
    r = build_dynamic_mission(m, plan, st_change, change, rng)
    if r is None:
        return None
    new_c, _rem_c, info_c = r
    flown_blind = st_detect.flown_distance - st_change.flown_distance
    max_dist = new_c.max_distance - flown_blind
    max_energy = None if new_c.max_energy is None else new_c.max_energy - (st_detect.energy_used - st_change.energy_used)
    if max_dist <= 1e-3 or (max_energy is not None and max_energy <= 1e-3):
        return None
    text = f"A change was announced earlier ({change}); the UAV has only now received it."
    fixed = {"zones": new_c.no_fly_zones, "active": new_c.active, "max_dist": max_dist, "max_energy": max_energy, "text": text, "info": dict(info_c)}
    rr = build_dynamic_mission(m, plan, st_detect, change, rng, fixed=fixed)
    if rr is None:
        return None
    new, rem, info = rr
    # exposure: stale polyline between the two states vs. the zone that changed
    pts = plan.route_points(m)
    alts = [m.start_altitude] + [w.altitude for w in plan.ordered()] + [plan.destination.altitude]
    path = [(st_change.position, st_change.altitude)] + [(pts[j][1], alts[j]) for j in range(st_change.seg_index + 1, st_detect.seg_index + 1)] + [(st_detect.position, st_detect.altitude)]
    zid = info_c.get("zone_id")
    exposed = False
    if zid is not None:
        z = next((z for z in new_c.no_fly_zones if z.id == zid), None)
        if z is not None:
            exposed = any(z.intersects_segment_3d(a[0], b[0], a[1], b[1]) for a, b in zip(path, path[1:]))
    info.update(detection_delay_progress=round(st_detect.progress - st_change.progress, 4), blind_distance=round(flown_blind, 3), exposure=bool(exposed))
    return new, rem, info


def classify_failure(rec: dict) -> str | None:
    """Likely cause of a failed dynamic run (survey Objective 4): the constraint itself, resource limits at the time of the change,
    or how much of the route was still open to modification."""
    fv = rec["final_verification"]
    if rec["final_status"] == "SUCCESS":
        return None
    f = rec.get("feasibility") or {}
    if f.get("status") == "INFEASIBLE":
        return "resource_limit_at_change" if "lower bound" in (f.get("reason") or "") else "change_makes_mission_infeasible"
    d = rec["dynamic"]
    if d["remaining_waypoints"] <= 1 or d["progress"] >= 0.6 and d["remaining_required"] <= 1:
        return "little_route_left_to_modify"
    if fv is None:
        return "no_parsable_plan"
    fams = [v["family"] for v in fv["violations"]] + [v["family"] for v in fv["preservation_violations"]]
    if not fams:
        return "unknown"
    dom = max(set(fams), key=fams.count)
    return f"unresolved_{dom}"

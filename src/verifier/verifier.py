"""Deterministic verifier. No LLM involvement. Only this module can declare a plan valid.

Five constraint families (altitude, no-fly zones/obstacles, distance, ordering, energy) are reported separately from
mission-requirement preservation (required waypoints present/unmoved, destination unchanged).

Resource model (documented assumption, configurable per mission; an experimental COST PROXY, not a physical battery model):
  distance = horizontal Euclidean route length START -> waypoints -> DEST            [map units; km or m for the real set, see scenario metadata]
  ascent   = sum of positive altitude changes, starting from mission.start_altitude  [m]
  energy   = distance + energy_k_v * ascent + energy_k_wp * n_waypoints              [route-length-equivalent units, i.e. the same unit as distance]
             energy_k_v  : route-length units charged per metre of climb (dimension: map-unit / m)
             energy_k_wp : route-length units charged per waypoint (hover/turn overhead; a modelling choice, 0 disables it)
  time     = distance / speed_h + ascent / speed_v

Zones may carry an optional vertical extent [z_min, z_max] (3D-aware obstacles); without it a zone is a full-height prism (original 2D behaviour).
Only the constraint families listed in mission.active are enforced (this includes "no_fly_zone").
"""
from __future__ import annotations
import time
from src.models.schemas import Mission, FlightPlan, Violation, VerificationResult
from src.utils import geometry as G

ALT_LOW, ALT_HIGH = "ALTITUDE_BELOW_MIN", "ALTITUDE_ABOVE_MAX"
NFZ_WP, NFZ_SEG = "NFZ_WAYPOINT", "NFZ_SEGMENT"
DIST, ORDER, ENERGY = "DISTANCE_EXCEEDED", "ORDER_VIOLATION", "ENERGY_EXCEEDED"
MISSING, MOVED, DEST_CHANGED, MID_MISMATCH = ("MISSING_REQUIRED_WAYPOINT", "REQUIRED_WAYPOINT_MOVED",
                                              "DESTINATION_CHANGED", "MISSION_ID_MISMATCH")
TERRAIN_LOW, TERRAIN_HIGH = "TERRAIN_CLEARANCE", "ABOVE_MAX_AGL"
FAMILIES = ["altitude", "no_fly_zone", "distance", "ordering", "energy"]   # "terrain" is reported additionally when the mission carries a terrain grid

HINTS = {
    ALT_LOW: "Raise this waypoint's altitude into the allowed band.", ALT_HIGH: "Lower this waypoint's altitude into the allowed band.",
    NFZ_WP: "Move this waypoint out of the zone; if it is a required waypoint it cannot move, so reroute the neighbouring segments instead.",
    NFZ_SEG: "Insert a detour waypoint so that both new segments stay outside the zone.",
    DIST: "Shorten the route: remove non-required waypoints or straighten detours.",
    ENERGY: "Reduce energy: climbing costs energy, so keep altitudes low inside the band, cut detours and unnecessary waypoints.",
    TERRAIN_LOW: "Raise the altitude of the waypoints around this location so that the route stays above the terrain by at least the required clearance (check both end points of the leg).",
    TERRAIN_HIGH: "Lower the route: it is higher above the ground than the maximum allowed height above ground level (AGL).",
    ORDER: "Reorder the waypoints so the 'before' waypoint is visited first.",
    MISSING: "Restore the required waypoint at its given coordinates.", MOVED: "Put the required waypoint back at its given coordinates.",
    DEST_CHANGED: "Restore the destination coordinates given in the mission."}


def _sev(ratio: float, lo: float, hi: float) -> str:
    return "low" if ratio < lo else ("medium" if ratio < hi else "high")


def route_distance(plan: FlightPlan, mission: Mission) -> tuple[float, list[tuple[str, str, float]]]:
    pts = plan.route_points(mission)
    segs = [(pts[i][0], pts[i + 1][0], G.dist(pts[i][1], pts[i + 1][1])) for i in range(len(pts) - 1)]
    return sum(s[2] for s in segs), segs


def resources(plan: FlightPlan, mission: Mission) -> dict:
    d, _ = route_distance(plan, mission)
    alts = [mission.start_altitude] + [w.altitude for w in plan.ordered()] + [plan.destination.altitude]
    ascent = sum(max(0.0, b - a) for a, b in zip(alts, alts[1:]))
    if mission.energy_model == "physical":
        from src.models.energy import plan_energy
        pe = plan_energy(mission, plan)
        return {"distance": d, "ascent": ascent, "energy": pe["energy_wh"], "time_s": pe["time_s"], "energy_detail": pe}
    return {"distance": d, "ascent": ascent, "energy": d + mission.energy_k_v * ascent + mission.energy_k_wp * len(plan.waypoints),
            "time_s": d / mission.speed_h + ascent / mission.speed_v}


def verify(mission: Mission, plan: FlightPlan) -> VerificationResult:
    t0 = time.perf_counter()
    viol: list[Violation] = []
    pres: list[Violation] = []
    wps = plan.ordered()
    nodes = wps + [plan.destination]
    label = lambda w: "DEST" if w is plan.destination else w.id
    band = max(mission.max_altitude - mission.min_altitude, 1e-9)

    def add(v: Violation):
        v.hint = HINTS.get(v.type, "")
        viol.append(v)

    # 1. altitude (inclusive limits)
    if mission.is_active("altitude"):
        for w in nodes:
            if w.altitude < mission.min_altitude - G.TOL:
                dev = mission.min_altitude - w.altitude
                add(Violation(ALT_LOW, "altitude", f"waypoint {label(w)}", w.altitude, f">= {mission.min_altitude}",
                              f"altitude {w.altitude} is below minimum {mission.min_altitude} by {dev:.3f}", waypoint_id=label(w),
                              evidence={"shortfall": round(dev, 4)}, severity=_sev(dev / band, 0.1, 0.3)))
            elif w.altitude > mission.max_altitude + G.TOL:
                dev = w.altitude - mission.max_altitude
                add(Violation(ALT_HIGH, "altitude", f"waypoint {label(w)}", w.altitude, f"<= {mission.max_altitude}",
                              f"altitude {w.altitude} exceeds maximum {mission.max_altitude} by {dev:.3f}", waypoint_id=label(w),
                              evidence={"excess": round(dev, 4)}, severity=_sev(dev / band, 0.1, 0.3)))

    # 2a. no-fly zones / obstacles: waypoint inside or on the boundary (closed sets; vertical extent respected when a zone has one)
    inside: dict[str, set[str]] = {}
    zones_on = mission.no_fly_zones if mission.is_active("no_fly_zone") else []
    for w in nodes:
        for z in zones_on:
            if z.contains_at(w.xy, w.altitude):
                inside.setdefault(z.id, set()).add(label(w))
                ev = z.point_evidence(w.xy)
                ev["zone_category"] = z.category
                if z.is_vertically_bounded:
                    ev["zone_altitude_range"] = [z.z_min, z.z_max]
                ratio = (ev["penetration_depth"] / z.radius) if z.kind == "circle" else (ev["distance_to_boundary"] / max(z.bounding_radius(), 1e-9))
                add(Violation(NFZ_WP, "no_fly_zone", f"waypoint {label(w)}", [w.x, w.y], f"outside {z.category} {z.id}",
                              f"waypoint {label(w)} at ({w.x:g}, {w.y:g}) lies inside or on the boundary of {z.category} {z.id}",
                              waypoint_id=label(w), zone_id=z.id, evidence=ev, severity=_sev(max(ratio, 0.0), 0.25, 0.6)))
    # 2b. segments (even when both endpoints are legal)
    pts = plan.route_points(mission)
    seg_alts = [mission.start_altitude] + [w.altitude for w in wps] + [plan.destination.altitude]
    for i in range(len(pts) - 1):
        (na, pa), (nb, pb) = pts[i], pts[i + 1]
        for z in zones_on:
            if (na != "START" and na in inside.get(z.id, ())) or (nb in inside.get(z.id, ())):
                continue
            if z.intersects_segment_3d(pa, pb, seg_alts[i], seg_alts[i + 1]):
                ev = z.segment_evidence(pa, pb)
                ev["zone_category"] = z.category
                if z.is_vertically_bounded:
                    ev["zone_altitude_range"] = [z.z_min, z.z_max]
                sev = _sev(max(ev.get("penetration_depth", 0) / z.radius, 0.0), 0.25, 0.6) if z.kind == "circle" else ("high" if ev.get("n_edge_crossings", 0) >= 2 else "medium")
                add(Violation(NFZ_SEG, "no_fly_zone", f"segment {na}->{nb}", [list(pa), list(pb)], f"no intersection with {z.category} {z.id}",
                              f"segment {na}->{nb} intersects (or touches) {z.category} {z.id}; its endpoints are not inside it",
                              segment=[na, nb], zone_id=z.id, evidence=ev, severity=sev))

    # 3. distance
    res = resources(plan, mission)
    total = res["distance"]
    _, segs = route_distance(plan, mission)
    if mission.is_active("distance") and total > mission.max_distance + G.TOL:
        ex = total - mission.max_distance
        add(Violation(DIST, "distance", "total route", round(total, 4), f"<= {mission.max_distance}",
                      f"route length {total:.3f} exceeds limit {mission.max_distance} by {ex:.3f}",
                      evidence={"excess": round(ex, 4), "segment_lengths": [{"from": a, "to": b, "length": round(l, 3)} for a, b, l in segs]},
                      severity=_sev(ex / mission.max_distance, 0.05, 0.15)))

    # 4. ordering over required waypoints
    pos = {w.id: i for i, w in enumerate(wps)}
    req_ids = [r.id for r in mission.required_waypoints]
    actual_seq = [w.id for w in wps if w.id in req_ids]
    if mission.is_active("ordering"):
        for before, after in mission.ordering:
            if before in pos and after in pos and pos[before] > pos[after]:
                add(Violation(ORDER, "ordering", f"waypoints {before},{after}", actual_seq, f"{before} before {after}",
                              f"{before} must be visited before {after}; plan visits required waypoints in order {actual_seq}", waypoint_id=after,
                              evidence={"expected": [before, after], "actual_required_sequence": actual_seq, "positions": {before: pos[before] + 1, after: pos[after] + 1}},
                              severity="high"))

    # 5. energy (battery)
    if mission.is_active("energy") and mission.max_energy is not None and mission.energy_model == "physical":
        det = res["energy_detail"]
        if det["non_flyable_legs"]:
            add(Violation(ENERGY, "energy", "total mission", "not flyable", f"<= {mission.max_energy} Wh",
                          f"{len(det['non_flyable_legs'])} leg(s) cannot be flown in the given wind (first: {det['non_flyable_legs'][0]['from']} -> {det['non_flyable_legs'][0]['to']})",
                          evidence={"non_flyable_legs": det["non_flyable_legs"], "wind": mission.wind, "model": "physical", "unit": "Wh"}, severity="high"))
        elif res["energy"] > mission.max_energy + G.TOL:
            ex = res["energy"] - mission.max_energy
            add(Violation(ENERGY, "energy", "total mission", round(res["energy"], 4), f"<= {mission.max_energy} Wh",
                          f"energy {res['energy']:.2f} Wh exceeds battery budget {mission.max_energy} Wh by {ex:.2f} Wh (vehicle {det['vehicle']}, wind {mission.wind}, payload {mission.payload_kg} kg)",
                          evidence={"excess": round(ex, 4), "model": "physical", "unit": "Wh", "hover_wh": round(det["hover_wh"], 4), "flight_time_s": round(det["time_s"], 2),
                                    "wh_per_leg": [{"from": l["from"], "to": l["to"], "wh": round(l["wh"], 4)} for l in det["legs"]]},
                          severity=_sev(ex / mission.max_energy, 0.05, 0.15)))
    elif mission.is_active("energy") and mission.max_energy is not None and res["energy"] > mission.max_energy + G.TOL:
        ex = res["energy"] - mission.max_energy
        add(Violation(ENERGY, "energy", "total mission", round(res["energy"], 4), f"<= {mission.max_energy}",
                      f"energy {res['energy']:.2f} exceeds battery budget {mission.max_energy} by {ex:.2f} (distance {total:.1f} + {mission.energy_k_v}*climb {res['ascent']:.1f} + {mission.energy_k_wp}*waypoints)",
                      evidence={"excess": round(ex, 4), "distance_part": round(total, 3), "climb_part": round(mission.energy_k_v * res["ascent"], 3),
                                "waypoint_part": round(mission.energy_k_wp * len(plan.waypoints), 3), "total_ascent": round(res["ascent"], 3),
                                "model": "energy = distance + k_v*ascent + k_wp*n_waypoints", "k_v": mission.energy_k_v, "k_wp": mission.energy_k_wp},
                      severity=_sev(ex / mission.max_energy, 0.05, 0.15)))

    # 6. terrain clearance (only when the mission carries a terrain grid): continuous along every flight segment (sampled at <= half a cell, linear altitude)
    terr = mission.terrain_grid
    if terr is not None:
        pts_t = plan.route_points(mission)
        alts_t = [mission.start_altitude] + [w.altitude for w in wps] + [plan.destination.altitude]
        worst_low, worst_high = None, None
        for i in range(len(pts_t) - 1):
            (na, pa), (nb, pb) = pts_t[i], pts_t[i + 1]
            for t, x, y, gh in terr.sample_segment(pa, pb):
                h = alts_t[i] + t * (alts_t[i + 1] - alts_t[i])
                agl = h - gh
                ex = mission.terrain_exempt_m / mission.unit_m
                near_ends = ex > 0 and (G.dist((x, y), mission.start_xy) <= ex or G.dist((x, y), mission.dest_xy) <= ex)
                if agl < mission.min_agl - 1e-6 and not near_ends and (worst_low is None or agl < worst_low[0]):
                    worst_low = (agl, na, nb, x, y, gh, h)
                if mission.max_agl is not None and agl > mission.max_agl + 1e-6 and (worst_high is None or agl > worst_high[0]):
                    worst_high = (agl, na, nb, x, y, gh, h)
        if worst_low:
            agl, na, nb, x, y, gh, h = worst_low
            add(Violation(TERRAIN_LOW, "terrain", f"segment {na}->{nb}", round(agl, 3), f">= {mission.min_agl} m AGL",
                          f"route is only {agl:.1f} m above the terrain at ({x:.1f}, {y:.1f}) (ground {gh:.1f} m, route {h:.1f} m); {mission.min_agl} m clearance required",
                          waypoint_id=nb if nb != "DEST" else "DEST", evidence={"location": [round(x, 2), round(y, 2)], "ground_height": round(gh, 2), "route_altitude": round(h, 2),
                                                                                  "required_min_altitude_here": round(gh + mission.min_agl, 2), "segment": [na, nb]},
                          severity=_sev(max(0.0, mission.min_agl - agl) / max(mission.min_agl, 1.0), 0.2, 0.6)))
        if worst_high:
            agl, na, nb, x, y, gh, h = worst_high
            add(Violation(TERRAIN_HIGH, "terrain", f"segment {na}->{nb}", round(agl, 3), f"<= {mission.max_agl} m AGL",
                          f"route is {agl:.1f} m above ground at ({x:.1f}, {y:.1f}), above the {mission.max_agl} m AGL limit",
                          waypoint_id=nb if nb != "DEST" else "DEST", evidence={"location": [round(x, 2), round(y, 2)], "ground_height": round(gh, 2), "route_altitude": round(h, 2),
                                                                                  "allowed_max_altitude_here": round(gh + mission.max_agl, 2), "segment": [na, nb]},
                          severity=_sev((agl - mission.max_agl) / max(mission.max_agl, 1.0), 0.1, 0.3)))

    # mission requirement preservation (separate from constraint validity)
    if plan.mission_id != mission.mission_id:
        pres.append(Violation(MID_MISMATCH, "mission_preservation", "plan", plan.mission_id, mission.mission_id, "plan mission_id differs", severity="high"))
    for r in mission.required_waypoints:
        w = next((w for w in wps if w.id == r.id), None)
        if w is None:
            pres.append(Violation(MISSING, "mission_preservation", f"required waypoint {r.id}", "absent", "present",
                                  f"required waypoint {r.id} at ({r.x:g}, {r.y:g}) is missing", waypoint_id=r.id, evidence={"required_xy": [r.x, r.y]},
                                  severity="high", hint=HINTS[MISSING]))
        elif G.dist(w.xy, r.xy) > mission.position_tolerance + G.TOL:
            pres.append(Violation(MOVED, "mission_preservation", f"required waypoint {r.id}", [w.x, w.y], [r.x, r.y],
                                  f"required waypoint {r.id} moved {G.dist(w.xy, r.xy):.3f} from its required location (tolerance {mission.position_tolerance})",
                                  waypoint_id=r.id, evidence={"required_xy": [r.x, r.y]}, severity="high", hint=HINTS[MOVED]))
    if G.dist(plan.destination.xy, mission.dest_xy) > mission.position_tolerance + G.TOL:
        pres.append(Violation(DEST_CHANGED, "mission_preservation", "destination", [plan.destination.x, plan.destination.y], list(mission.destination),
                              "plan destination differs from mission destination", waypoint_id="DEST", evidence={"required_xy": list(mission.destination)},
                              severity="high", hint=HINTS[DEST_CHANGED]))

    per = {f: {"satisfied": not any(v.family == f for v in viol), "n_violations": sum(v.family == f for v in viol),
               "types": sorted({v.type for v in viol if v.family == f}), "active": mission.is_active(f)} for f in FAMILIES}
    if terr is not None:
        per["terrain"] = {"satisfied": not any(v.family == "terrain" for v in viol), "n_violations": sum(v.family == "terrain" for v in viol),
                          "types": sorted({v.type for v in viol if v.family == "terrain"}), "active": True}
    per["mission_preservation"] = {"satisfied": not pres, "n_violations": len(pres), "types": sorted({v.type for v in pres}), "active": True}
    return VerificationResult(valid=not viol, mission_preserved=not pres, violations=viol, preservation_violations=pres,
                              route_distance=round(total, 6), per_constraint=per, verification_time_s=time.perf_counter() - t0,
                              energy=round(res["energy"], 6), flight_time_s=round(res["time_s"], 6), ascent=round(res["ascent"], 6))

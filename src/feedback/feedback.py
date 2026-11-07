"""Builds verifier-derived feedback and the repair prompt. Everything here is computed from VerificationResult only."""
from __future__ import annotations
import json
from dataclasses import dataclass
from src.models.schemas import Mission, FlightPlan, VerificationResult
from src.planner.prompts import mission_block
from src.repair.diff import edit_scope
from src.benchmark.feasibility import shortest_route, zones_at_altitude

LEVELS = ("binary", "localized", "evidence", "prescriptive")   # prescriptive = evidence + concrete detour coordinates computed on the verifier side
MODES = ("regenerate", "targeted")


@dataclass
class Feedback:
    level: str
    mode: str
    payload: dict
    scope: dict | None
    notes: list[str]

    def to_dict(self) -> dict:
        return {"level": self.level, "mode": self.mode, "payload": self.payload, "scope": self.scope, "notes": self.notes}


def _loc(v) -> dict:
    d = {"type": v.type, "severity": v.severity, "location": v.location, "waypoint_id": v.waypoint_id, "segment": v.segment, "zone_id": v.zone_id,
         "observed": v.actual, "required": v.required, "details": v.details}
    return {k: x for k, x in d.items() if x is not None}


def _clip_alt(mission: Mission, a: float) -> float:
    return round(min(max(a, mission.min_altitude), mission.max_altitude), 1) if mission.is_active("altitude") else round(a, 1)


def suggest_detour(mission: Mission, plan: FlightPlan, v) -> dict | None:
    """PRESCRIPTIVE hint: a legal detour (visibility-graph shortest route around the zones) for ONE no-fly-zone violation.
    The LLM still has to edit the plan correctly, but it no longer has to do the geometry. Returns None when no detour is needed/possible."""
    rp = plan.route_points(mission)
    ids = [i for i, _ in rp]
    alts = [mission.start_altitude] + [w.altitude for w in plan.ordered()] + [plan.destination.altitude]
    try:
        if v.type == "NFZ_SEGMENT" and v.segment and v.segment[0] in ids:
            i = ids.index(v.segment[0])
            alt = _clip_alt(mission, (alts[i] + alts[i + 1]) / 2)
            path = shortest_route(mission, [rp[i][1], rp[i + 1][1]], zones=zones_at_altitude(mission, alt))
            if not path or len(path) < 3:
                return None
            out = {"action": "insert", "between": [ids[i], ids[i + 1]],
                   "new_waypoints_in_order": [{"x": round(x, 2), "y": round(y, 2), "altitude": alt} for x, y in path[1:-1]]}
            z = mission.zone(v.zone_id) if v.zone_id else None
            if z is not None and z.is_vertically_bounded and z.z_max is not None and z.z_max + 0.5 <= mission.max_altitude:
                out["or_fly_over"] = {"zone": z.id, "set_altitude_of_both_end_waypoints_to_at_least": round(z.z_max + 0.5, 1), "note": "zone has a finite height; climbing above it avoids the detour (costs climb energy)"}
            return out
        if v.type == "NFZ_WAYPOINT" and v.waypoint_id in ids and v.waypoint_id not in {r.id for r in mission.required_waypoints} | {"DEST"}:
            i = ids.index(v.waypoint_id)
            alt = _clip_alt(mission, alts[i])
            path = shortest_route(mission, [rp[i - 1][1], rp[i + 1][1]], zones=zones_at_altitude(mission, alt))
            if not path:
                return None
            return {"action": "replace", "waypoint": v.waypoint_id,
                    "with_waypoints_in_order": [{"x": round(x, 2), "y": round(y, 2), "altitude": alt} for x, y in path[1:-1]]}
    except Exception:
        return None
    return None


def build_feedback(mission: Mission, plan: FlightPlan, res: VerificationResult, level: str, mode: str, notes: list[str] | None = None) -> Feedback:
    assert level in LEVELS and mode in MODES
    notes = notes or []
    if level == "binary":
        return Feedback(level, mode, {"valid": False}, None, notes)
    allv = res.all_violations()
    payload = {"valid": False, "n_violations": len(allv), "by_severity": {k: sum(v.severity == k for v in allv) for k in ("low", "medium", "high")},
               "violations": [_loc(v) for v in res.violations],
               "mission_requirement_violations": [_loc(v) for v in res.preservation_violations]}
    scope = None
    if level in ("evidence", "prescriptive"):
        for p, v in zip(payload["violations"], res.violations):
            p["evidence"] = v.evidence
            p["suggested_fix"] = v.hint
            if level == "prescriptive":
                d = suggest_detour(mission, plan, v)
                if d:
                    p["suggested_detour"] = d
        for p, v in zip(payload["mission_requirement_violations"], res.preservation_violations):
            p["evidence"] = v.evidence
        scope = edit_scope(mission, plan, allv)
        locked = [{"id": r.id, "x": r.x, "y": r.y} for r in mission.required_waypoints]
        payload["must_remain_unchanged"] = {
            "required_waypoints_xy": locked, "destination_xy": mission.destination, "start_xy": mission.start,
            "ordering_pairs": mission.ordering, "altitude_range": [mission.min_altitude, mission.max_altitude],
            "max_route_distance": mission.max_distance if mission.is_active("distance") else None,
            "max_energy": mission.max_energy if mission.is_active("energy") else None, "no_fly_zones": "all zones remain restricted"}
        payload["current_route_distance"] = res.route_distance
        if mode == "targeted":
            payload["edit_scope"] = scope
    return Feedback(level, mode, payload, scope, notes)


def repair_prompt(mission: Mission, plan: FlightPlan, fb: Feedback, prompt_mode: str = "spec") -> str:
    plan_json = json.dumps({"waypoints": [{"id": w.id, "x": w.x, "y": w.y, "altitude": w.altitude} for w in plan.ordered()],
                            "destination": {"x": plan.destination.x, "y": plan.destination.y, "altitude": plan.destination.altitude}})
    if fb.mode == "targeted" and fb.level in ("evidence", "prescriptive"):
        instr = ("Repair the plan with MINIMAL changes. Only waypoints listed in edit_scope.editable_waypoint_ids may be modified or removed; "
                 "new waypoints may be inserted only between the segments in edit_scope.insertion_allowed_between; "
                 "every other waypoint must be returned exactly unchanged. Required waypoints must keep their given coordinates and the ordering pairs must hold.")
        if fb.level == "prescriptive":
            instr += (" Where a violation has a suggested_detour, apply it exactly: insert (or substitute) its waypoints with the given coordinates, "
                      "in the given order, using new ids.")
    else:
        instr = "Produce a corrected, complete flight plan that satisfies every mission constraint while still visiting all required waypoints at their given coordinates."
    note = ("\n" + "\n".join("NOTE: " + n for n in fb.notes)) if fb.notes else ""
    return ("[ROLE]repair\n" + mission_block(mission, prompt_mode) +
            f"\n\nYOUR PREVIOUS PLAN WAS REJECTED BY A DETERMINISTIC VERIFIER.\n[CURRENT_PLAN]{plan_json}[/CURRENT_PLAN]\n"
            f"[FEEDBACK_LEVEL]{fb.level}[/FEEDBACK_LEVEL] [REPAIR_MODE]{fb.mode}[/REPAIR_MODE]\n"
            f"[FEEDBACK]{json.dumps(fb.payload)}[/FEEDBACK]{note}\n\n{instr}\nReply with the corrected plan JSON only.")

"""LLM-free "direct execution" baseline (reviewer request: 'LLM versus direct classical-plan execution').

C_prescriptive hands the model verifier-computed detours (`suggested_detour`) and asks it to copy them into the plan. If that copy step is all the
model does, a deterministic program should reach the same success rate. This module IS that program: it applies exactly the same prescriptive
feedback programmatically (zero LLM calls) to the SAME shared initial plan, then re-verifies, for up to `max_iterations` rounds.

What it implements (nothing else, so the comparison is fair):
  * every `suggested_detour` of kind "insert" / "replace" produced by feedback.suggest_detour;
  * altitude violations: clamp the offending waypoint into the allowed band (the verifier hint is literally 'move into the band');
It deliberately does NOT reorder waypoints, restore missing/moved required waypoints, or fix distance/energy violations -- those are exactly the
violation classes for which the prescriptive feedback carries no computed fix, so any remaining gain there is attributable to the LLM.
"""
from __future__ import annotations
from src.models.schemas import Mission, FlightPlan, Waypoint, VerificationResult
from src.feedback.feedback import suggest_detour
from src.verifier.verifier import MISSING, MOVED, DEST_CHANGED, MID_MISMATCH, ORDER


def apply_prescriptive(mission: Mission, plan: FlightPlan, res: VerificationResult) -> tuple[FlightPlan, dict]:
    """Return (patched plan, log). The input plan is not modified."""
    seq = [(w.id, w.x, w.y, w.altitude) for w in plan.ordered()]          # mutable ordered list of non-destination waypoints
    dest = (plan.destination.x, plan.destination.y, plan.destination.altitude)
    used_ids = {w[0] for w in seq}
    counter = [0]
    log = {"inserts": 0, "replaces": 0, "altitude_clamps": 0, "unfixable": []}

    def new_id() -> str:
        while True:
            counter[0] += 1
            nid = f"D{counter[0]}"
            if nid not in used_ids:
                used_ids.add(nid)
                return nid

    done_between: set = set()
    for v in res.violations:
        if v.type in ("NFZ_SEGMENT", "NFZ_WAYPOINT"):
            d = suggest_detour(mission, plan, v)
            if not d:
                log["unfixable"].append(v.type)
                continue
            if d["action"] == "insert":
                key = tuple(d["between"])
                if key in done_between:      # one detour per segment already covers every zone on it (visibility-graph route avoids all zones)
                    continue
                done_between.add(key)
                a = d["between"][0]
                idx = -1 if a == "START" else next((i for i, w in enumerate(seq) if w[0] == a), None)
                if idx is None:
                    log["unfixable"].append(v.type)
                    continue
                for k, p in enumerate(d["new_waypoints_in_order"]):
                    seq.insert(idx + 1 + k, (new_id(), p["x"], p["y"], p["altitude"]))
                log["inserts"] += 1
            elif d["action"] == "replace":
                idx = next((i for i, w in enumerate(seq) if w[0] == d["waypoint"]), None)
                if idx is None:
                    log["unfixable"].append(v.type)
                    continue
                seq[idx:idx + 1] = [(new_id(), p["x"], p["y"], p["altitude"]) for p in d["with_waypoints_in_order"]]
                log["replaces"] += 1
        elif v.type in ("ALTITUDE_BELOW_MIN", "ALTITUDE_ABOVE_MAX"):
            lo, hi = mission.min_altitude, mission.max_altitude
            if v.waypoint_id == "DEST":
                dest = (dest[0], dest[1], min(max(dest[2], lo), hi))
            else:
                seq = [(i, x, y, min(max(a, lo), hi)) if i == v.waypoint_id else (i, x, y, a) for i, x, y, a in seq]
            log["altitude_clamps"] += 1
        else:
            log["unfixable"].append(v.type)
    wps = [Waypoint(i, x, y, a, k + 1) for k, (i, x, y, a) in enumerate(seq)]
    return FlightPlan(mission.mission_id, wps, Waypoint("DEST", dest[0], dest[1], dest[2], len(wps) + 1)).validate(), log


STRUCTURAL = {MISSING, MOVED, DEST_CHANGED, MID_MISMATCH, ORDER}


def apply_full(mission: Mission, plan: FlightPlan, res: VerificationResult) -> tuple[FlightPlan, dict]:
    """STRONGER LLM-free repair ("direct_full"): everything `apply_prescriptive` does, plus deterministic handling of the structural violation classes the prescriptive feedback
    leaves to the model -- missing / moved required waypoints, a changed destination and waypoint ORDER. If any of those is present the plan is rebuilt as a skeleton
    (required waypoints at their coordinates in the first precedence-feasible order, destination restored, altitudes clamped into the band) and the zone detours are then
    computed by the same verifier routine. Distance/energy/terrain violations remain unfixable here (they need the witness fallback or the model).
    Together with the optional witness fallback this is the 'best deterministic pipeline that uses only information the verifier already has' and is the fair control for
    the question 'does the LLM add anything once the verifier can compute the route?'."""
    # Structural classes live in BOTH lists: ORDER_VIOLATION is a constraint violation, while missing / moved required waypoints, a changed destination and
    # a mission-id mismatch are mission-preservation violations (VerificationResult.preservation_violations). Looking only at `res.violations` (as before)
    # meant a plan that merely dropped a required waypoint was never rebuilt, so the control under-reported what deterministic repair can do.
    types = {v.type for v in res.all_violations()}
    if not (types & (STRUCTURAL - {MID_MISMATCH})):        # a wrong mission id alone needs no rebuild: apply_prescriptive always writes mission.mission_id
        return apply_prescriptive(mission, plan, res)
    from src.benchmark.feasibility import visiting_order
    order = visiting_order(mission) or [r.id for r in mission.required_waypoints]
    alts = [w.altitude for w in plan.waypoints] + [plan.destination.altitude]
    alt = sum(alts) / len(alts) if alts else mission.min_altitude
    alt = min(max(alt, mission.min_altitude), mission.max_altitude)
    wps = [Waypoint(i, mission.required(i).x, mission.required(i).y, alt, k + 1) for k, i in enumerate(order)]
    skeleton = FlightPlan(mission.mission_id, wps, Waypoint("DEST", mission.destination[0], mission.destination[1], alt, len(wps) + 1)).validate()
    from src.verifier.verifier import verify
    r2 = verify(mission, skeleton)
    patched, log = apply_prescriptive(mission, skeleton, r2) if r2.violations else (skeleton, {"inserts": 0, "replaces": 0, "altitude_clamps": 0, "unfixable": []})
    log = dict(log, rebuilt_skeleton=True, structural_violations=sorted(types & STRUCTURAL))
    return patched, log

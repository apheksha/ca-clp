from __future__ import annotations
from src.models.schemas import FlightPlan, Mission, Violation


def plan_diff(prev: FlightPlan | None, new: FlightPlan | None, tol: float = 1e-6) -> dict:
    if prev is None or new is None:
        return {"added": [], "removed": [], "modified": [], "reordered": False, "n_changes": 0}
    p = {w.id: w for w in prev.waypoints}
    n = {w.id: w for w in new.waypoints}
    added = sorted(set(n) - set(p))
    removed = sorted(set(p) - set(n))
    modified = sorted(i for i in set(p) & set(n) if abs(p[i].x - n[i].x) > tol or abs(p[i].y - n[i].y) > tol or abs(p[i].altitude - n[i].altitude) > tol)
    common_p = [w.id for w in prev.ordered() if w.id in n]
    common_n = [w.id for w in new.ordered() if w.id in p]
    reordered = common_p != common_n
    dest_mod = abs(prev.destination.x - new.destination.x) > tol or abs(prev.destination.y - new.destination.y) > tol or abs(prev.destination.altitude - new.destination.altitude) > tol
    if dest_mod:
        modified.append("DEST")
    return {"added": added, "removed": removed, "modified": modified, "reordered": reordered,
            "n_changes": len(added) + len(removed) + len(modified) + (1 if reordered else 0)}


def edit_scope(mission: Mission, plan: FlightPlan, violations: list[Violation]) -> dict:
    """Waypoints the verifier evidence implicates; targeted repair is only allowed to touch these."""
    req = {r.id for r in mission.required_waypoints}
    ids = {w.id for w in plan.waypoints} | {"DEST"}
    editable, insert_between, reorder = set(), [], False
    for v in violations:
        if v.type in ("ALTITUDE_BELOW_MIN", "ALTITUDE_ABOVE_MAX"):
            editable.add(v.waypoint_id)
        elif v.type == "NFZ_WAYPOINT":
            editable.add(v.waypoint_id)
        elif v.type == "NFZ_SEGMENT":
            insert_between.append(v.segment)
            editable.update(s for s in v.segment if s in ids and s not in req)
        elif v.type in ("DISTANCE_EXCEEDED", "ENERGY_EXCEEDED"):
            if v.type == "ENERGY_EXCEEDED":
                editable.update(w.id for w in plan.waypoints)   # altitudes of every waypoint may be lowered
                editable.add("DEST")
            editable.update(w.id for w in plan.waypoints if w.id not in req)
            insert_between.append(["ANY", "ANY"])
        elif v.type == "ORDER_VIOLATION":
            reorder = True
            editable.update(v.evidence.get("expected", []))
        elif v.type in ("MISSING_REQUIRED_WAYPOINT", "REQUIRED_WAYPOINT_MOVED", "DESTINATION_CHANGED"):
            editable.add(v.waypoint_id)
            reorder = reorder or v.type == "MISSING_REQUIRED_WAYPOINT"
    return {"editable_waypoint_ids": sorted(editable), "insertion_allowed_between": insert_between, "reorder_allowed": reorder}


def collateral_edits(diff: dict, scope: dict | None) -> int:
    """Changes to existing waypoints outside the permitted scope (only meaningful for targeted mode)."""
    if scope is None:
        return 0
    ok = set(scope["editable_waypoint_ids"])
    return sum(1 for i in diff["modified"] + diff["removed"] if i not in ok)

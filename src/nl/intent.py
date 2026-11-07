"""Intent preservation: does the extracted constraint set mean what the operator meant?

Two levels:
  1. field level  : compare extracted constraints with the ground truth (precision/recall over constraint atoms, hallucinated constraints counted separately);
  2. mission level (`cross_validate`): build a Mission from the extraction, obtain a verified witness plan for it, and verify that plan against the TRUE mission.
     A plan that is valid for the extracted mission but violates the true one proves that the extraction changed the intent in a way that matters operationally."""
from __future__ import annotations
from src.nl.dataset import MapDef, constraints_to_mission


def atoms(c: dict) -> set:
    a = set()
    for k in ("start", "destination"):
        if c.get(k):
            a.add((k, c[k]))
    for w in c.get("waypoints") or []:
        a.add(("waypoint", w))
    for p in c.get("order") or []:
        a.add(("order", tuple(p)))
    for k in ("min_altitude_m", "max_altitude_m", "max_distance_m"):
        if c.get(k) is not None:
            a.add((k, round(float(c[k]), 3)))
    for z in c.get("avoid_zones") or []:
        a.add(("avoid", z))
    return a


def compare(extracted: dict, truth: dict) -> dict:
    ea, ta = atoms(extracted), atoms(truth)
    tp = ea & ta
    wrong_value = {k for (k, v) in ea - ta if k in ("min_altitude_m", "max_altitude_m", "max_distance_m", "start", "destination") and any(k == k2 for (k2, _) in ta)}
    hallucinated = {(k, v) for (k, v) in ea - ta if k not in {x for x, _ in ta} or k in ("waypoint", "order", "avoid")}
    prec = len(tp) / len(ea) if ea else 1.0
    rec = len(tp) / len(ta) if ta else 1.0
    return {"precision": prec, "recall": rec, "f1": (2 * prec * rec / (prec + rec)) if prec + rec else 0.0, "n_true": len(ta), "n_extracted": len(ea),
            "missed": sorted(map(str, ta - ea)), "spurious": sorted(map(str, ea - ta)), "wrong_value_fields": sorted(wrong_value), "intent_preserved": ea == ta}


def cross_validate(extracted: dict, truth: dict, mdef: MapDef) -> dict:
    """Operational intent check. `defaults` make unspecified numbers non-binding so that only stated constraints matter."""
    defaults = {"min_altitude_m": 0.0, "max_altitude_m": 120.0}
    me = constraints_to_mission(extracted, mdef, "ext", defaults)
    mt = constraints_to_mission(truth, mdef, "truth", defaults)
    if me is None or mt is None:
        return {"checked": False, "reason": "extracted or true mission is incomplete"}
    from src.benchmark.feasibility import assess
    from src.verifier.verifier import verify
    fe, ft = assess(me), assess(mt)
    out = {"checked": True, "extracted_status": fe.status, "true_status": ft.status}
    if fe.status == "FEASIBLE":
        plan = fe.witness_plan
        plan.mission_id = mt.mission_id
        r = verify(mt, plan)
        out["plan_for_extracted_valid_for_truth"] = bool(r.success)
        out["violations_under_truth"] = [v.type for v in r.violations]
    return out

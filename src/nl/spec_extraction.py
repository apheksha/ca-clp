"""LLM AS THE LANGUAGE INTERFACE (v12): free-text mission -> structured specification -> classical planner -> verifier against the TRUE mission.

Motivation (review, point D): with the structured specification in the prompt, a visibility-graph planner already solves every feasible mission, so the open
question is whether a language model can supply that specification from the operator's text. This module measures exactly that, with one model call per mission:

  1. the model receives the mission's natural-language description plus the plain-text zone geometry (the same text as prompt_mode "nl_only") and must return the
     specification as JSON (it is told to return null for anything the text does not state);
  2. field-level comparison with the ground-truth Mission (start, destination, required waypoints, altitude band, distance / energy budgets and coefficients,
     ordering pairs, zones) with a numeric tolerance that matches how numbers are printed in the text;
  3. operational check: the feasibility oracle plans on the EXTRACTED mission; its verified witness is then verified against the TRUE mission.
     end_to_end_success = the extracted mission led to a plan that is valid and mission-preserving for the real mission.

Nothing here is a stand-in model: the clients come from src/planner/llm.py (gemini, groq, ollama, ...). Provider errors stop the run; finished missions are kept and
`--resume` continues (scripts/spec_extraction_eval.py)."""
from __future__ import annotations
import json
import math
import re
from src.models.schemas import Mission, RequiredWaypoint, NoFlyZone, ALL_FAMILIES, SchemaError
from src.planner.prompts import zone_geometry_text

SPEC_SYSTEM = """You convert a UAV mission written in plain English into a structured specification.
Copy numbers exactly as written. Never invent a value: if the text does not state something, use null (or an empty list).
Reply with ONE JSON object and nothing else."""

SPEC_SCHEMA = """{"start": [x, y], "destination": [x, y],
 "required_waypoints": [{"id": "<the short identifier written in the text, e.g. A>", "x": number, "y": number}, ...],
 "ordering": [["<earlier waypoint id>", "<later waypoint id>"], ...],
 "min_altitude": number|null, "max_altitude": number|null,
 "max_distance": number|null,
 "max_energy": number|null, "energy_k_v": number|null, "energy_k_wp": number|null,
 "zones": [{"id": "<zone id>", "kind": "circle", "center": [x, y], "radius": number, "z_min": number|null, "z_max": number|null}
           | {"id": "<zone id>", "kind": "polygon", "vertices": [[x, y], ...], "z_min": number|null, "z_max": number|null}, ...]}"""


def spec_prompt(m: Mission) -> str:
    zg = zone_geometry_text(m)
    return ("MISSION TEXT:\n" + m.description
            + (f"\n\nZONE GEOMETRY (plain text): {zg}. Zones are closed regions: touching the boundary counts as entering." if zg else "")
            + "\n\nReturn the specification as JSON with exactly these keys:\n" + SPEC_SCHEMA)


def parse_spec(text: str) -> dict:
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    mm = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if mm:
        t = mm.group(1).strip()
    if not t.startswith("{"):
        mm = re.search(r"\{.*\}", t, re.S)
        if not mm:
            raise ValueError("no JSON object in the reply")
        t = mm.group(0)
    d = json.loads(t)
    if not isinstance(d, dict):
        raise ValueError("reply is not a JSON object")
    return d


# ---------------------------------------------------------------------------------------------------------------------------------
# ground truth as the text states it (inactive families are not mentioned by the generator, so their truth is "not stated")
# ---------------------------------------------------------------------------------------------------------------------------------
def truth_spec(m: Mission) -> dict:
    act = set(m.active)
    return {"start": list(m.start), "destination": list(m.destination),
            "required_waypoints": [{"id": r.id, "x": r.x, "y": r.y} for r in m.required_waypoints],
            "ordering": [list(p) for p in m.ordering] if "ordering" in act else [],
            "min_altitude": m.min_altitude if "altitude" in act else None, "max_altitude": m.max_altitude if "altitude" in act else None,
            "max_distance": m.max_distance if "distance" in act else None,
            "max_energy": m.max_energy if ("energy" in act and m.max_energy is not None) else None,
            "energy_k_v": m.energy_k_v if ("energy" in act and m.max_energy is not None) else None,
            "zones": [z for z in m.no_fly_zones] if "no_fly_zone" in act else []}


def _tol(v: float) -> float:
    """Numbers are printed with one decimal below 1000 and as integers above (generator._fmt); zone geometry uses %g (6 significant digits)."""
    return 0.051 + (0.51 if abs(v) >= 1000 else 0.0) + 1e-5 * abs(v)


def _num_eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return abs(float(a) - float(b)) <= _tol(float(b))
    except (TypeError, ValueError):
        return False


def _pt_eq(a, b) -> bool:
    return isinstance(a, (list, tuple)) and len(a) == 2 and _num_eq(a[0], b[0]) and _num_eq(a[1], b[1])


def _zone_eq(e: dict, z: NoFlyZone) -> bool:
    if not isinstance(e, dict) or e.get("kind") != z.kind:
        return False
    if z.kind == "circle":
        ok = _pt_eq(e.get("center"), z.center) and _num_eq(e.get("radius"), z.radius)
    else:
        vs = e.get("vertices") or []
        ok = len(vs) == len(z.vertices) and all(_pt_eq(a, b) for a, b in zip(vs, z.vertices))
    zmax_e = e.get("z_max")
    return ok and (_num_eq(zmax_e, z.z_max) if (z.z_max is not None or zmax_e is not None) else True)


def compare_spec(ext: dict, m: Mission) -> dict:
    """Per-field correctness (True/False) of an extracted spec against the truth, plus counts of missed / spurious zones and waypoints."""
    t = truth_spec(m)
    f = {}
    f["start"] = _pt_eq(ext.get("start"), t["start"])
    f["destination"] = _pt_eq(ext.get("destination"), t["destination"])
    ew = {str(w.get("id")): w for w in (ext.get("required_waypoints") or []) if isinstance(w, dict)}
    tw = {w["id"]: w for w in t["required_waypoints"]}
    f["required_waypoints"] = set(ew) == set(tw) and all(_num_eq(ew[i].get("x"), tw[i]["x"]) and _num_eq(ew[i].get("y"), tw[i]["y"]) for i in tw)
    f["ordering"] = sorted(map(tuple, (p for p in (ext.get("ordering") or []) if isinstance(p, (list, tuple)) and len(p) == 2))) == sorted(map(tuple, t["ordering"]))
    f["altitude_band"] = _num_eq(ext.get("min_altitude"), t["min_altitude"]) and _num_eq(ext.get("max_altitude"), t["max_altitude"])
    f["max_distance"] = _num_eq(ext.get("max_distance"), t["max_distance"])
    if t["max_energy"] is None:
        f["energy"] = ext.get("max_energy") is None
    else:
        f["energy"] = _num_eq(ext.get("max_energy"), t["max_energy"]) and _num_eq(ext.get("energy_k_v"), t["energy_k_v"])
    ez = {str(z.get("id")): z for z in (ext.get("zones") or []) if isinstance(z, dict)}
    tz = {z.id: z for z in t["zones"]}
    zone_ok = {i: (i in ez and _zone_eq(ez[i], tz[i])) for i in tz}
    f["zones"] = set(ez) == set(tz) and all(zone_ok.values())
    return {"fields": f, "all_fields_correct": all(f.values()), "n_fields": len(f), "n_fields_correct": sum(f.values()),
            "zones_true": len(tz), "zones_correct": sum(zone_ok.values()), "zones_missed": len(set(tz) - set(ez)), "zones_spurious": len(set(ez) - set(tz)),
            "waypoints_true": len(tw), "waypoints_missed": len(set(tw) - set(ew)), "waypoints_spurious": len(set(ew) - set(tw))}


def mission_from_spec(ext: dict, truth: Mission) -> Mission:
    """Build a verifiable Mission from the extracted spec. Families the extraction leaves unstated are switched off (that is what the extraction says);
    verification parameters that are not mission content (position tolerance, start altitude, units, speeds) are copied from the truth."""
    act = []
    lo, hi = ext.get("min_altitude"), ext.get("max_altitude")
    if lo is not None and hi is not None:
        act.append("altitude")
    zones = []
    for k, z in enumerate(ext.get("zones") or []):
        zid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(z.get("id") or f"Z{k + 1}"))[:32] or f"Z{k + 1}"
        while zid in {q.id for q in zones}:
            zid = (zid + "_")[:32]
        if z.get("kind") == "polygon":
            nz = NoFlyZone(zid, "polygon", vertices=z.get("vertices"), z_min=z.get("z_min"), z_max=z.get("z_max"))
        else:
            nz = NoFlyZone(zid, "circle", center=z.get("center"), radius=z.get("radius"), z_min=z.get("z_min"), z_max=z.get("z_max"))
        zones.append(nz)
    if zones:
        act.append("no_fly_zone")
    if ext.get("max_distance") is not None:
        act.append("distance")
    ids = []
    reqs = []
    for w in ext.get("required_waypoints") or []:
        wid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(w.get("id")))[:32]
        reqs.append(RequiredWaypoint(wid, w.get("x"), w.get("y")))
        ids.append(wid)
    ordering = [[re.sub(r"[^A-Za-z0-9_\-]", "_", str(a))[:32], re.sub(r"[^A-Za-z0-9_\-]", "_", str(b))[:32]] for a, b in
                (p for p in (ext.get("ordering") or []) if isinstance(p, (list, tuple)) and len(p) == 2)]
    ordering = [p for p in ordering if p[0] in ids and p[1] in ids]
    if ordering:
        act.append("ordering")
    if ext.get("max_energy") is not None:
        act.append("energy")
    m = Mission(truth.mission_id, truth.description, ext.get("start"), ext.get("destination"), reqs,
                float(lo) if lo is not None else 0.0, float(hi) if hi is not None else 1e6, float(ext.get("max_distance") or 1e12), zones, ordering,
                position_tolerance=truth.position_tolerance, start_altitude=truth.start_altitude,
                max_energy=None if ext.get("max_energy") is None else float(ext["max_energy"]),
                energy_k_v=float(ext.get("energy_k_v") if ext.get("energy_k_v") is not None else truth.energy_k_v),
                energy_k_wp=float(ext.get("energy_k_wp") if ext.get("energy_k_wp") is not None else 0.0),
                speed_h=truth.speed_h, speed_v=truth.speed_v, active=[a for a in ALL_FAMILIES if a in act], unit_m=truth.unit_m)
    return m.validate()


def end_to_end(ext: dict, truth: Mission) -> dict:
    """Plan on the extracted mission with the oracle's verified witness, then verify that plan against the TRUE mission."""
    from src.benchmark.feasibility import assess
    from src.verifier.verifier import verify
    try:
        me = mission_from_spec(ext, truth)
    except (SchemaError, ValueError, TypeError, KeyError) as e:
        return {"built": False, "error": f"{type(e).__name__}: {e}", "extracted_status": None, "end_to_end_success": False, "violations_under_truth": []}
    f = assess(me)
    out = {"built": True, "error": None, "extracted_status": f.status, "end_to_end_success": False, "violations_under_truth": []}
    if f.status == "FEASIBLE" and f.witness_plan is not None:
        plan = f.witness_plan
        plan.mission_id = truth.mission_id
        r = verify(truth, plan)
        out["end_to_end_success"] = bool(r.success)
        out["violations_under_truth"] = sorted({v.type for v in r.all_violations()})
    return out


def evaluate_reply(text: str, truth: Mission) -> dict:
    try:
        ext = parse_spec(text)
    except (ValueError, TypeError) as e:
        return {"parsed": False, "parse_error": f"{type(e).__name__}: {e}", "all_fields_correct": False, "end_to_end_success": False}
    cmp_ = compare_spec(ext, truth)
    e2e = end_to_end(ext, truth)
    return {"parsed": True, "parse_error": None, **cmp_, **e2e, "extracted": ext}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)

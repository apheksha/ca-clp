"""Uncertainty detection, conflict detection and clarification requests for extracted mission constraints.

Decision for an instruction: `ready` (can be planned), `needs_clarification` (essential constraints missing/ambiguous) or `conflict` (the stated constraints contradict
each other or are provably impossible). Conflicts are detected DETERMINISTICALLY (altitude band, cyclic order, waypoint inside an excluded zone, and the feasibility
oracle's proofs for distance/energy), not by asking the language model. Nothing is silently defaulted: a missing essential value produces a question."""
from __future__ import annotations
from dataclasses import dataclass, field
from src.nl.dataset import MapDef, constraints_to_mission
from src.nl.extraction import Extraction, NUMERIC_FIELDS

ESSENTIAL = ("start", "destination", "max_altitude_m")     # a plan cannot be verified without them (policy; configurable)
QUESTIONS = {
    "start": "Where should the flight start?",
    "destination": "Where should the flight end?",
    "max_altitude_m": "What is the maximum altitude (in metres)?",
    "min_altitude_m": "What is the minimum altitude (in metres)?",
    "max_distance_m": "What is the maximum total flight distance (in metres or km)?",
    "avoid_zones": "Which restricted zones must be avoided?",
}


@dataclass
class Analysis:
    decision: str = "ready"
    missing: list = field(default_factory=list)
    ambiguities: list = field(default_factory=list)      # [{"field","phrase","reason"}]
    conflicts: list = field(default_factory=list)        # [{"type","detail"}]
    unknown_places: list = field(default_factory=list)
    questions: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def _hedged_fields(text: str) -> dict:
    """Map a hedged phrase to the constraint it most likely qualifies (very small lexicon; the LLM extractor reports its own mapping)."""
    low = text.lower()
    out = {}
    for phrase in ("pretty low", "not too high", "fairly low", "quite low", "very low", "low", "high", "up", "altitude", "height", "give or take"):
        if phrase in low and any(w in low for w in ("low", "high", "up", "altitude", "height")):
            out["max_altitude_m"] = phrase
            break
    for phrase in ("near", "close to", "try not", "area"):
        if phrase in low:
            out["avoid_zones"] = phrase
    if any(p in low for p in ("about", "roughly", "approximately")) and "km" in low:
        out["max_distance_m"] = "approximate distance"
    return out


def analyze(ext: Extraction, mdef: MapDef, text: str | None = None, essential: tuple = ESSENTIAL) -> Analysis:
    a = Analysis()
    text = text if text is not None else ext.raw
    c = ext.constraints()
    # ---- missing essentials (from the text itself, not defaulted) ----
    for f in essential:
        if c.get(f) in (None, "", []):
            a.missing.append(f)
    # ---- ambiguity ----
    hedge = _hedged_fields(text)
    for f, phrase in hedge.items():
        # a hedge only matters when the value it qualifies was NOT given as a clean number
        if f == "max_altitude_m" and c.get("max_altitude_m") is None:
            a.ambiguities.append({"field": f, "phrase": phrase, "reason": "vague altitude wording without a number"})
        elif f == "max_distance_m" and ("about" in text.lower() or "roughly" in text.lower()) and c.get("max_distance_m") is not None:
            a.ambiguities.append({"field": f, "phrase": phrase, "reason": "limit stated approximately: is it a hard limit?"})
        elif f == "avoid_zones" and (("try not" in text.lower()) or (("near" in text.lower()) and not c.get("avoid_zones"))):
            a.ambiguities.append({"field": f, "phrase": phrase, "reason": "avoidance wording is soft or the zone is not named"})
        elif f == "avoid_zones" and "area" in text.lower() and not c.get("avoid_zones"):
            a.ambiguities.append({"field": f, "phrase": phrase, "reason": "zone referred to by description, not by name"})
    for e in ext.ambiguous:      # ambiguities the LLM extractor reported
        if e not in a.ambiguities and e.get("field") not in ("text",):
            a.ambiguities.append(e)
    # ---- unknown places ----
    unknown = [n for n in ([ext.start, ext.destination] + list(ext.waypoints)) if n and n not in mdef.sites] + list(ext.unknown_places)
    a.unknown_places = sorted(set(unknown))
    # ---- conflicts (deterministic) ----
    if ext.min_altitude_m is not None and ext.max_altitude_m is not None and ext.min_altitude_m > ext.max_altitude_m:
        a.conflicts.append({"type": "empty_altitude_band", "detail": f"minimum altitude {ext.min_altitude_m} m is above maximum altitude {ext.max_altitude_m} m"})
    names = list(ext.waypoints)
    pairs = [tuple(p) for p in ext.order]
    if _cyclic(pairs):
        a.conflicts.append({"type": "cyclic_ordering", "detail": "the stated visiting orders contradict each other"})
    for n in [ext.start, ext.destination] + names:
        if n in mdef.sites:
            for zn in ext.avoid_zones:
                z = mdef.zones.get(zn)
                if z and _inside(mdef.sites[n], z):
                    a.conflicts.append({"type": "point_in_zone", "detail": f"{n} lies inside the excluded zone {zn}"})
    if not a.conflicts:
        m = constraints_to_mission(c, mdef, "nl_check", defaults={"min_altitude_m": 0.0, "max_altitude_m": 120.0})
        if m is not None and (c.get("max_distance_m") is not None):
            from src.benchmark.feasibility import assess
            f = assess(m)
            if f.status == "INFEASIBLE":
                a.conflicts.append({"type": f.proof_kind or "infeasible", "detail": f.reason})
    for cf in ext.conflicts:     # conflicts reported by the LLM extractor (kept, flagged as unverified)
        if not any(x["type"] == cf.get("type") for x in a.conflicts):
            a.conflicts.append({"type": cf.get("type", "reported"), "detail": cf.get("detail", "") + " (reported by the extractor, not verified)"})
    # ---- decision + questions ----
    if a.conflicts:
        a.decision = "conflict"
        a.questions = [f"Your instruction contains a conflict ({x['type']}): {x['detail']}. How should it be resolved?" for x in a.conflicts]
    elif a.missing or a.ambiguities or a.unknown_places:
        a.decision = "needs_clarification"
        for f in a.missing:
            a.questions.append(QUESTIONS.get(f, f"Please specify {f}."))
        for x in a.ambiguities:
            q = f"'{x['phrase']}' is not precise ({x['reason']}). " + QUESTIONS.get(x["field"], "Please give an exact value.")
            if q not in a.questions and QUESTIONS.get(x["field"]) not in a.questions:
                a.questions.append(q)
        for u in a.unknown_places:
            a.questions.append(f"I cannot find '{u}' on the map. Which known place do you mean ({', '.join(mdef.sites)})?")
    return a


def _inside(p, z) -> bool:
    import math
    if z["kind"] == "circle":
        return math.dist(p, z["center"]) <= z["radius"]
    from src.utils import geometry as G
    return G.point_in_polygon(tuple(p), [tuple(v) for v in z["vertices"]]) != "outside"


def _cyclic(pairs) -> bool:
    adj: dict = {}
    for a, b in pairs:
        adj.setdefault(a, set()).add(b)
    seen, stack = set(), set()

    def dfs(u):
        seen.add(u)
        stack.add(u)
        for v in adj.get(u, ()):
            if v in stack or (v not in seen and dfs(v)):
                return True
        stack.discard(u)
        return False
    return any(dfs(u) for u in list(adj) if u not in seen)


def apply_answers(ext: Extraction, answers: dict) -> Extraction:
    """Fold the operator's answers (field -> value) into the extraction and clear the corresponding gaps."""
    import copy
    e = copy.deepcopy(ext)
    for k, v in answers.items():
        if k in ("waypoints", "avoid_zones", "order"):
            setattr(e, k, list(v))
        else:
            setattr(e, k, v)
    e.ambiguous = [x for x in e.ambiguous if x.get("field") not in answers]
    e.missing = [x for x in e.missing if x not in answers]
    e.unknown_places = [] if "unknown_places_resolved" in answers else e.unknown_places
    return e

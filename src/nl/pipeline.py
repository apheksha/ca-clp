"""Clarification-aware planning front end: text -> extraction -> analysis -> (questions | verified-planning-ready Mission).

`understand(text, extractor, mdef, ask=None)`: if essential information is missing/ambiguous the pipeline does NOT guess: it returns the clarification questions, or,
when an `ask` callback (operator) is supplied, asks them, folds the answers in and re-analyzes (up to `max_rounds`). Conflicts are returned to the operator the same way.
Nothing is defaulted silently; numeric defaults exist only for the verification-level intent check in src/nl/intent.py."""
from __future__ import annotations
from dataclasses import dataclass, field
from src.nl.analysis import analyze, apply_answers, Analysis
from src.nl.dataset import MapDef, constraints_to_mission
from src.nl.extraction import Extraction


@dataclass
class Understanding:
    decision: str
    extraction: Extraction
    analysis: Analysis
    mission: object = None
    rounds: list = field(default_factory=list)       # [{"questions": [...], "answers": {...}}]


def understand(text: str, extractor, mdef: MapDef, ask=None, max_rounds: int = 2, mission_id: str = "nl") -> Understanding:
    ext = extractor.extract(text)
    an = analyze(ext, mdef, text)
    rounds = []
    for _ in range(max_rounds):
        if an.decision == "ready" or ask is None:
            break
        answers = ask(an)
        rounds.append({"questions": list(an.questions), "answers": answers})
        if not answers:
            break
        ext = apply_answers(ext, answers)
        # answers resolve ambiguity: remove vagueness the operator just clarified
        an = analyze(ext, mdef, "")
    mission = None
    if an.decision == "ready":
        mission = constraints_to_mission(ext.constraints(), mdef, mission_id, defaults={"min_altitude_m": 0.0, "max_distance_m": 1e9})
        if mission is not None:
            from src.planner.prompts import mission_block   # (the description for the planner is the operator's text, now backed by confirmed constraints)
            mission.description = text
    return Understanding(an.decision, ext, an, mission, rounds)

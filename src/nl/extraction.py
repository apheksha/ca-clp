"""Constraint extraction from free-form mission text.

Two extractors share one output schema (`Extraction`):
  RuleExtractor : deterministic regex/gazetteer baseline (no model, offline) -- also the reference for tests;
  LLMExtractor  : asks a language model for the same JSON, with explicit permission (and instruction) to say 'missing' / 'ambiguous' instead of guessing.
The extractor does NOT decide whether the mission is acceptable; `src/nl/analysis.py` does (missing fields, ambiguity, conflicts, clarification questions)."""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field

HEDGES = ["pretty", "roughly", "about", "around", "approximately", "give or take", "somewhere", "near", "close to", "a bit", "not too", "fairly", "somewhat", "try not",
          "try to", "or so", "maybe", "perhaps", "usual", "same as", "as needed", "if possible", "ish", "kind of", "low", "high"]
NUMERIC_FIELDS = ("min_altitude_m", "max_altitude_m", "max_distance_m")


@dataclass
class Extraction:
    start: str | None = None
    destination: str | None = None
    waypoints: list = field(default_factory=list)
    order: list = field(default_factory=list)
    min_altitude_m: float | None = None
    max_altitude_m: float | None = None
    max_distance_m: float | None = None
    avoid_zones: list = field(default_factory=list)
    unknown_places: list = field(default_factory=list)       # place names mentioned but not on the map
    ambiguous: list = field(default_factory=list)            # [{"field", "phrase", "reason"}]
    missing: list = field(default_factory=list)              # fields the text does not specify
    conflicts: list = field(default_factory=list)            # [{"type", "detail"}] reported by the extractor itself (LLM)
    raw: str = ""
    error: str | None = None

    def constraints(self) -> dict:
        return {"start": self.start, "destination": self.destination, "waypoints": self.waypoints, "order": self.order, "min_altitude_m": self.min_altitude_m,
                "max_altitude_m": self.max_altitude_m, "max_distance_m": self.max_distance_m, "avoid_zones": self.avoid_zones}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


class RuleExtractor:
    """Regex + gazetteer baseline. Deliberately simple: it is a lower bound that shows what the LLM extractor must beat, and it makes the pipeline testable offline."""
    name = "rule"

    def __init__(self, sites: list[str], zones: list[str]):
        self.sites = sorted(sites, key=len, reverse=True)
        self.zones = sorted(zones, key=len, reverse=True)

    def _find(self, text: str, names: list[str]) -> list[tuple[int, str]]:
        low, hits, taken = text.lower(), [], []
        for n in names:
            for m in re.finditer(r"\b" + re.escape(n.lower()) + r"\b", low):
                if not any(a < m.end() and m.start() < b for a, b in taken):
                    hits.append((m.start(), n))
                    taken.append((m.start(), m.end()))
        return sorted(hits)

    def extract(self, text: str, mission_id: str | None = None) -> Extraction:
        low = text.lower()
        e = Extraction(raw=text)
        zone_hits = self._find(text, self.zones)
        zone_spans = [(p, p + len(n)) for p, n in zone_hits]
        site_hits = [(p, n) for p, n in self._find(text, self.sites) if not any(a <= p < b for a, b in zone_spans)]
        # a zone name that contains a site name (e.g. "Hospital helipad") must not count the site once more
        S = "|".join(re.escape(n) for n in self.sites)
        # destination: explicit cues in priority order (landing > ending > return > stop), then sequence heuristics
        dest = None
        for pat in (r"\bland(?:s|ing)?\s+(?:at|on|in)\s+(?:the )?(" + S + r")\b", r"\b(?:end(?:s|ing)?|finish(?:es|ing)?|terminat(?:e|es|ed|ing))\s+(?:at|in)\s+(?:the )?(" + S + r")\b",
                    r"\b(?:return(?:s|ing)?\s+to|come back to|back to)\s+(?:the )?(" + S + r")\b", r"\bstop(?:s|ping)?\s+(?:there|at)\s+(?:the )?(" + S + r")\b"):
            mm = re.search(pat, text, re.I)
            if mm:
                dest = self._canon(mm.group(1))
                break
        if dest is None and re.search(r"\b(come back|returns? home|back home|then return|and return)\b", low):
            dest = start0 = None
        # start
        start = None
        ms = re.search(r"(?:take ?off from|depart(?:s)?(?: from)?|leaves?|start(?:s|ing)?(?: at| from)?|begin(?:s)? at|\bfrom)\s*(?:the )?(" + S + r")\b", text, re.I)
        if ms:
            start = self._canon(ms.group(1))
        else:
            mm = re.search(r"^\W*(?:mission:\s*)?(" + S + r")\s*(?:to|->)", text, re.I)
            if mm:
                start = self._canon(mm.group(1))
        if start is None and "->" in text and site_hits:
            start = site_hits[0][1]
        if dest is None and re.search(r"\b(come back|returns? home|back home|then return|and return|come home)\b", low):
            dest = start
        # sequence heuristics when nothing explicit: "X to Y then Z", "X -> Y -> Z", "from X to Y"
        if dest is None:
            mm = re.search(r"\bfrom (?:the )?(" + S + r")\b[^.]*?\bto (?:the )?(" + S + r")\b", text, re.I)
            if mm and not re.search(r"\bvia\b|\bthen\b", low):
                dest = self._canon(mm.group(2))
        if dest is None and start is not None and site_hits and (re.search(r"\bthen\b", low) or "->" in text):
            seq = [n for _, n in site_hits]
            if seq and seq[-1] != start:
                dest = seq[-1]
        if dest is None and start is not None:
            mm = re.search(r"\b(" + S + r")\s+to\s+(?:the )?(" + S + r")\b", text, re.I)
            if mm and re.search(r"\bvia\b", low):
                dest = self._canon(mm.group(2))
            elif mm and not re.search(r"\bthen\b", low):
                dest = self._canon(mm.group(2))
        e.start, e.destination = start, dest
        wps = []
        for _, n in site_hits:
            if n not in wps and n != start and n != dest:
                wps.append(n)
        # sites mentioned only as start/destination are not waypoints; a site that is both start and destination can still be a waypoint only if mentioned between
        e.waypoints = wps
        # ordering: explicit 'A before B' (overlapping allowed), else the sequence of waypoints when the text says then / -> / in that order
        pairs = []
        for mm in re.finditer(r"(?=\b(" + S + r")\b\s+before\s+(?:the )?(" + S + r")\b)", text, re.I):
            pairs.append([self._canon(mm.group(1)), self._canon(mm.group(2))])
        for mm in re.finditer(r"(?=\b(?:before|ahead of)\s+(?:the )?(" + S + r")\b)", text, re.I):
            pass
        mm = re.search(r"\b(?:overfly|visit|check|inspect)\s+(?:the )?(" + S + r")\s+before\s+(?:the )?(" + S + r")\b", text, re.I)
        if mm and [self._canon(mm.group(1)), self._canon(mm.group(2))] not in pairs:
            pairs.append([self._canon(mm.group(1)), self._canon(mm.group(2))])
        if not pairs and len(wps) >= 2 and (re.search(r"\bthen\b|in that order", low) or "->" in text):
            pairs = [[a, b] for a, b in zip(wps, wps[1:])]
        e.order = [p for p in pairs if p[0] in wps + [start, dest] and p[1] in wps + [start, dest]]
        # altitude
        mm = re.search(r"between (\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)?\s*(?:and|to|-|\.\.)\s*(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)?", text, re.I) or re.search(r"(\d+)\s*(?:-|\.\.|to)\s*(\d+)\s*(?:m|metres|meters)\b", text, re.I) \
            or re.search(r"(?:altitude|height)[^0-9]{0,25}?(\d+)\s*(?:to|-|\.\.)\s*(\d+)", text, re.I) or re.search(r"min (\d+) ?m,? max (\d+) ?m", text, re.I)
        if mm:
            e.min_altitude_m, e.max_altitude_m = float(mm.group(1)), float(mm.group(2))
        mlow = re.search(r"(?:no lower than|at least|not lower than|min(?:imum)?)\s*(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)?", text, re.I)
        mhigh = re.search(r"(?:no higher than|never above|not above|below|under|max(?:imum)? (?:height|altitude)|max height|keep (?:it )?under|stay below)\s*(\d+(?:\.\d+)?)\s*(?:m|meters?|metres?)\b", text, re.I)
        if mlow and e.min_altitude_m is None:
            e.min_altitude_m = float(mlow.group(1))
        if mhigh and e.max_altitude_m is None:
            e.max_altitude_m = float(mhigh.group(1))
        if e.max_altitude_m is None:
            mmx = re.search(r"\bmax(?:imum)?\s*(\d{2,3})\s*(?:m|metres|meters)?\b(?!\s*(?:km|total|of flight))", text, re.I)
            if mmx and float(mmx.group(1)) <= 300 and e.max_distance_m != float(mmx.group(1)):
                e.max_altitude_m = float(mmx.group(1))
        mfly = re.search(r"no lower than (\d+) and no higher than (\d+)", text, re.I)
        if mfly:
            e.min_altitude_m, e.max_altitude_m = float(mfly.group(1)), float(mfly.group(2))
        mhi = re.search(r"at least (\d+)\s*m[^.]*?never above (\d+)", text, re.I)
        if mhi:
            e.min_altitude_m, e.max_altitude_m = float(mhi.group(1)), float(mhi.group(2))
        # distance
        md = re.search(r"(\d+(?:\.\d+)?)\s*km", text, re.I)
        if md:
            e.max_distance_m = float(md.group(1)) * 1000
        else:
            md = re.search(r"(?:distance limit|route length|range limit|range|total(?: flight)?(?: distance)?|distance)[^0-9.]{0,14}(\d{3,6})\s*(?:m|metres|meters)\b", text, re.I) or \
                 re.search(r"(?:max(?:imum)?|no more than|not exceed|at most)\s*(\d{3,6})\s*(?:m|metres|meters)?\s*(?:total|of flight|route|distance|of route)", text, re.I) or \
                 re.search(r"(\d{3,6})\s*(?:m|metres|meters)\s*(?:max|at most|total|limit|of flight)", text, re.I)
            if md:
                e.max_distance_m = float(md.group(1))
        # zones to avoid
        e.avoid_zones = [n for _, n in zone_hits if re.search(r"(avoid|keep out|clear of|away from|out of|exclusion|prohibited|don't go|do not enter|not enter|stay out|not go|through the)", low)]
        # plain-name zone cues like "the helipad area" are NOT resolved by this baseline (the LLM extractor / clarification handles them)
        # unknown places: capitalised or 'old factory'-like phrases after 'to'/'via' that are not known
        for mm2 in re.finditer(r"\b(?:to|via|visit|over|at) (?:the )?((?:old|new|usual)\s+[a-z]+|usual spot|[a-z]+ spot)\b", low):
            if not any(n.lower() in mm2.group(1) for n in self.sites):
                e.unknown_places.append(mm2.group(1))
        for h in HEDGES:
            for mh in re.finditer(r"([A-Za-z]+\s+)?\b" + re.escape(h) + r"\b(\s+[A-Za-z0-9]+)?", text, re.I):
                e.ambiguous.append({"field": "text", "phrase": mh.group(0).strip(), "reason": f"hedge word '{h}'"})
        return e

    def _canon(self, s: str) -> str:
        return next((n for n in self.sites if n.lower() == s.lower()), s)


EXTRACTION_SYSTEM = """You extract UAV mission constraints from an operator's free-text instruction.
Use ONLY the places and zones of the given map. Never guess: if a value is not stated, return null and list it under "missing"; if the text is vague
(e.g. "pretty low", "near", "about", "the usual spot"), return null for that value and list the phrase under "ambiguous". If two statements contradict each other,
list them under "conflicts". Answer with ONE JSON object and nothing else."""

EXTRACTION_SCHEMA = """{"start": place|null, "destination": place|null, "waypoints": [place,...], "order": [[earlier_place, later_place],...],
 "min_altitude_m": number|null, "max_altitude_m": number|null, "max_distance_m": number|null, "avoid_zones": [zone,...],
 "unknown_places": [phrase,...], "ambiguous": [{"field": str, "phrase": str, "reason": str}], "missing": [field,...], "conflicts": [{"type": str, "detail": str}]}"""


def extraction_prompt(text: str, sites: list[str], zones: list[str]) -> str:
    return (f"MAP PLACES: {', '.join(sites)}\nMAP ZONES: {', '.join(zones)}\nUnits: metres.\n\nINSTRUCTION:\n\"\"\"{text}\"\"\"\n\nReturn JSON with exactly this schema:\n{EXTRACTION_SCHEMA}")


def parse_extraction_json(raw: str) -> Extraction:
    e = Extraction(raw=raw)
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        e.error = "no JSON object in model output"
        return e
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError as ex:
        e.error = f"invalid JSON: {ex}"
        return e
    e.start, e.destination = d.get("start"), d.get("destination")
    e.waypoints = list(d.get("waypoints") or [])
    e.order = [list(p) for p in (d.get("order") or []) if isinstance(p, (list, tuple)) and len(p) == 2]
    for k in NUMERIC_FIELDS:
        setattr(e, k, _num(d.get(k)))
    e.avoid_zones = list(d.get("avoid_zones") or [])
    e.unknown_places = list(d.get("unknown_places") or [])
    e.ambiguous = [x for x in (d.get("ambiguous") or []) if isinstance(x, dict)]
    e.missing = list(d.get("missing") or [])
    e.conflicts = [x for x in (d.get("conflicts") or []) if isinstance(x, dict)]
    return e


class LLMExtractor:
    name = "llm"

    def __init__(self, client, sites: list[str], zones: list[str], temperature: float = 0.0, max_tokens: int = 1200):
        self.client, self.sites, self.zones, self.temperature, self.max_tokens = client, sites, zones, temperature, max_tokens

    def extract(self, text: str, mission_id: str | None = None) -> Extraction:
        r = self.client.generate(EXTRACTION_SYSTEM, extraction_prompt(text, self.sites, self.zones), self.temperature, self.max_tokens, seed=0)
        return parse_extraction_json(r.text)

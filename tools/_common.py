"""Shared helpers for the analysis scripts: load runs.jsonl and rebuild each mission from the logged prompt (works for static and dynamic runs)."""
import json, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.models.schemas import Mission, FlightPlan


def runs(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def mission_of(r):
    if r.get("dynamic") and r["dynamic"].get("mission"):
        d = dict(r["dynamic"]["mission"]); d.setdefault("description", ""); return Mission.from_dict(d).validate()
    for it in r["iterations"]:
        for c in it["calls"]:
            mt = re.search(r"\[MISSION_SPEC\](.*?)\[/MISSION_SPEC\]", c.get("prompt", ""), re.S)
            if mt:
                spec = json.loads(mt.group(1)); spec.setdefault("description", ""); return Mission.from_dict(spec).validate()
    return None


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** .5) / d
    return c - h, c + h


def mission_map(rows):
    """scenario_id -> Mission, using any record of that scenario that logged a prompt (non-LLM conditions have no prompt of their own)."""
    mp = {}
    for r in rows:
        if r["scenario_id"] not in mp:
            m = mission_of(r)
            if m is not None:
                mp[r["scenario_id"]] = m
    return mp

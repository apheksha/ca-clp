"""Loader for the human-written mission collection (data/human_missions/missions.yaml) and conversion of constraint dictionaries into verifiable Missions."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
import yaml
from src.models.schemas import Mission, RequiredWaypoint, NoFlyZone

DEFAULT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "human_missions", "missions.yaml")
CATEGORIES = ("complete", "incomplete", "ambiguous", "conflicting")


@dataclass
class MapDef:
    name: str
    unit_m: float
    sites: dict
    zones: dict

    def site_names(self) -> list[str]:
        return list(self.sites)

    def zone_names(self) -> list[str]:
        return list(self.zones)


@dataclass
class HumanMission:
    id: str
    category: str
    text: str
    truth: dict
    expect: dict = field(default_factory=dict)


def load_collection(path: str | None = None) -> tuple[MapDef, list[HumanMission]]:
    with open(path or DEFAULT_PATH) as fh:
        d = yaml.safe_load(fh)
    mp = d["map"]
    mdef = MapDef(mp.get("name", "map"), float(mp.get("unit_m", 1.0)), {k: list(map(float, v)) for k, v in mp["sites"].items()}, mp["zones"])
    out = []
    for e in d["missions"]:
        if e["category"] not in CATEGORIES:
            raise ValueError(f"mission {e['id']}: unknown category {e['category']!r}")
        out.append(HumanMission(e["id"], e["category"], e["text"], e["truth"], e.get("expect", {}) or {}))
    return mdef, out


def _zone(name: str, spec: dict) -> NoFlyZone:
    zid = "Z_" + name.replace(" ", "_")
    if spec["kind"] == "circle":
        return NoFlyZone(zid, "circle", center=list(spec["center"]), radius=float(spec["radius"]), description=name).validate()
    return NoFlyZone(zid, "polygon", vertices=[list(v) for v in spec["vertices"]], description=name).validate()


def constraints_to_mission(c: dict, mdef: MapDef, mission_id: str = "nl", defaults: dict | None = None) -> Mission | None:
    """Build a Mission from a constraint dict (same keys as `truth`). Returns None if a required element (start/destination) is missing. `defaults` fills unspecified
    numbers (e.g. {"min_altitude_m": 0, "max_altitude_m": 120, "max_distance_m": 1e9}) -- used only for the verification-level intent check, never silently by the assistant."""
    d = dict(defaults or {})
    d.update({k: v for k, v in c.items() if v is not None})
    if not d.get("start") or not d.get("destination"):
        return None
    if d["start"] not in mdef.sites or d["destination"] not in mdef.sites:
        return None
    names = list(d.get("waypoints") or [])
    if any(n not in mdef.sites for n in names):
        return None
    ids = {n: f"W{i + 1}" for i, n in enumerate(names)}
    req = [RequiredWaypoint(ids[n], mdef.sites[n][0], mdef.sites[n][1]) for n in names]
    zones = [_zone(n, mdef.zones[n]) for n in (d.get("avoid_zones") or []) if n in mdef.zones]
    order = [[ids[a], ids[b]] for a, b in (d.get("order") or []) if a in ids and b in ids]
    return Mission(mission_id, "", mdef.sites[d["start"]], mdef.sites[d["destination"]], req, float(d.get("min_altitude_m", 0.0)), float(d.get("max_altitude_m", 120.0)),
                   float(d.get("max_distance_m", 1e9)), zones, order, unit_m=mdef.unit_m).validate()

"""Event-detection interface for the dynamic (mid-flight) study.

Until v9 a change was INJECTED into the mission at a known instant. Here a change is a real-world EVENT (`Event`) that happens in the simulated environment at
time t_appear; the UAV only learns of it through a sensing channel:
  * onboard  : `ZoneEventSensor` (src/simulator/sensors.py) sees a new/enlarged zone when the ESTIMATED position is within sensor range (+ processing delay);
  * broadcast: a NOTAM / UTM-style message arrives `broadcast_latency_s` after the event, independent of position (the only channel for resource-limit events).
`EventEnvironment` holds the ground truth (used for collision checking) and `SensedMap` holds what the UAV currently knows (used for planning). The map is updated
when an event is detected; `online.py` connects detection to automatic re-planning.

Existing change generators (src/dynamic/changes.py) are reused: `events_from_change` converts a generated change into an Event."""
from __future__ import annotations
import copy
import random
from dataclasses import dataclass, field
from src.models.schemas import Mission, FlightPlan, NoFlyZone


@dataclass
class Event:
    kind: str                       # zone_add | zone_enlarge | limit_distance | limit_energy
    t_appear: float                 # simulation time at which the world changes
    zone: NoFlyZone | None = None   # the new zone, or the enlarged version of an existing zone (same id)
    value: float | None = None      # new absolute remaining budget for limit events (map units / Wh) -- as an ABSOLUTE limit on the whole mission
    delivery: str = "onboard"       # onboard | broadcast  (limit events are always broadcast)
    note: str = ""

    @property
    def event_id(self) -> str:
        return f"{self.kind}:{self.zone.id if self.zone else 'limit'}@{self.t_appear:g}"


class EventEnvironment:
    """Ground truth: base zones + events that become real at their t_appear."""

    def __init__(self, mission: Mission, events: list[Event]):
        self.m = mission
        self.events = sorted(events, key=lambda e: e.t_appear)

    def truth_zones(self, t: float) -> list[NoFlyZone]:
        zs = {z.id: z for z in (self.m.no_fly_zones if self.m.is_active("no_fly_zone") else [])}
        for e in self.events:
            if e.zone is not None and t >= e.t_appear:
                zs[e.zone.id] = e.zone            # enlarged zone replaces the original; a new zone is added
        return list(zs.values())

    def appear_times(self) -> dict[str, float]:
        return {e.zone.id: e.t_appear for e in self.events if e.zone is not None}

    def truth_mission(self, t_end: float) -> Mission:
        """The mission as it is after all events up to t_end (zones and limits), for the final success evaluation."""
        m = copy.deepcopy(self.m)
        m.no_fly_zones = self.truth_zones(t_end)
        if "no_fly_zone" not in m.active and any(e.zone is not None for e in self.events):
            m.active = list(m.active) + ["no_fly_zone"]
        for e in self.events:
            if t_end >= e.t_appear:
                if e.kind == "limit_distance":
                    m.max_distance = min(m.max_distance, e.value)
                elif e.kind == "limit_energy" and m.max_energy is not None:
                    m.max_energy = min(m.max_energy, e.value)
        return m


class SensedMap:
    """What the UAV believes: the mission's zones and limits, updated as events are detected."""

    def __init__(self, mission: Mission):
        self.zones = {z.id: copy.deepcopy(z) for z in mission.no_fly_zones}
        self.max_distance = mission.max_distance
        self.max_energy = mission.max_energy
        self.detected: list[dict] = []

    def apply(self, event: Event, t_detect: float, channel: str) -> None:
        if event.zone is not None:
            self.zones[event.zone.id] = copy.deepcopy(event.zone)
        if event.kind == "limit_distance":
            self.max_distance = min(self.max_distance, event.value)
        if event.kind == "limit_energy" and self.max_energy is not None:
            self.max_energy = min(self.max_energy, event.value)
        self.detected.append({"event": event.event_id, "kind": event.kind, "t_appear": event.t_appear, "t_detect": round(t_detect, 2),
                              "detection_delay_s": round(t_detect - event.t_appear, 2), "channel": channel})


def events_from_change(m: Mission, plan: FlightPlan, progress: float, change: str, rng: random.Random, t_appear: float, delivery: str = "onboard") -> Event | None:
    """Turn a change produced by src/dynamic/changes.py (same random geometry) into an Event that happens at `t_appear` seconds."""
    from src.dynamic.changes import execution_state, build_dynamic_mission
    st = execution_state(m, plan, progress)
    r = build_dynamic_mission(m, plan, st, change, rng)
    if r is None:
        return None
    new, _rem, info = r
    if change in ("add_zone", "enlarge_zone"):
        zid = info.get("zone_id")
        z = next((z for z in new.no_fly_zones if z.id == zid), None) if zid else (new.no_fly_zones[-1] if new.no_fly_zones else None)
        if z is None:
            return None
        return Event("zone_add" if change == "add_zone" else "zone_enlarge", t_appear, zone=copy.deepcopy(z), delivery=delivery, note=info.get("text", ""))
    if change == "tighten_distance":
        return Event("limit_distance", t_appear, value=st.flown_distance + new.max_distance, delivery="broadcast", note="remaining distance allowance reduced")
    if change == "reduce_energy" and new.max_energy is not None:
        return Event("limit_energy", t_appear, value=st.energy_used + new.max_energy, delivery="broadcast", note="remaining energy budget reduced")
    return None

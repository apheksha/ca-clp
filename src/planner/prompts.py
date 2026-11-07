import json
from src.models.schemas import Mission

PROMPT_VERSION = "plan-v3"   # unchanged text for prompt_mode="spec"; "nl_only"/"spec_only" are ablation modes (see PROMPT_MODES)
PROMPT_MODES = ("spec", "nl_only", "spec_only")   # spec = NL + structured spec (paper default); nl_only = free text only (tests constraint EXTRACTION); spec_only = JSON only
#   # v3: START/DEST must not appear in "waypoints" (observed: llama3.1:8b put them there in 30/40 calls)

SYSTEM_PROMPT = """You are a UAV mission planner. Convert the mission into a waypoint-based flight plan.
Reply with ONLY one JSON object, no prose and no markdown, in exactly this schema:
{"waypoints":[{"id":"<string>","x":<number>,"y":<number>,"altitude":<number>}, ...],
 "destination":{"x":<number>,"y":<number>,"altitude":<number>}}
Waypoints are flown in list order after the start point. Use the given ids for required named waypoints
at their given coordinates; you may add extra waypoints with new ids (e.g. "W1"). All numbers must be JSON numbers.
The start point is implicit: do NOT list START or DEST inside "waypoints". Give the final position only in "destination".
Use no fields other than id, x, y, altitude (waypoints) and x, y, altitude (destination).
Some missions also give an energy budget: energy = horizontal route length + k_v * total climb (sum of altitude increases, counted from
start_altitude) + k_wp * number of waypoints. Lower altitudes inside the allowed band therefore use less energy.
Only the constraint families listed in "active" are enforced. Zones with category "obstacle" are buildings/terrain and must be avoided like no-fly zones."""


def _coarse_terrain(m: Mission, n: int = 6) -> list:
    """Coarse ground-elevation samples (x, y, ground_m) covering the terrain grid, so the model can reason about clearance without the full DEM."""
    t = m.terrain_grid
    if t is None:
        return []
    xs = [t.x0 + i * (t.cols - 1) * t.cell / (n - 1) for i in range(n)]
    ys = [t.y0 + j * (t.rows - 1) * t.cell / (n - 1) for j in range(n)]
    return [{"x": round(x, 1), "y": round(y, 1), "ground_m": round(t.height_at(x, y), 1)} for y in ys for x in xs]


def mission_spec(m: Mission) -> dict:
    spec = _mission_spec_core(m)
    if m.energy_model == "physical":
        spec.update(energy_model="physical", energy_unit="Wh", vehicle=(m.vehicle if isinstance(m.vehicle, str) else m.vehicle), payload_kg=m.payload_kg, wind_east_north_mps=m.wind,
                    map_unit_metres=m.unit_m)
        for k in ("energy_k_v", "energy_k_wp"):
            spec.pop(k, None)
    if m.terrain is not None:
        spec.update(min_agl=m.min_agl, max_agl=m.max_agl, ground_elevation_samples=_coarse_terrain(m),
                    terrain_note="altitudes are on the same datum as ground_m; every flight segment (not only waypoints) must stay >= min_agl above the ground")
    return spec


def _mission_spec_core(m: Mission) -> dict:
    return {"mission_id": m.mission_id, "start": m.start, "destination": m.destination,
            "required_waypoints": [{"id": r.id, "x": r.x, "y": r.y} for r in m.required_waypoints],
            "min_altitude": m.min_altitude, "max_altitude": m.max_altitude, "max_distance": m.max_distance,
            "ordering": m.ordering, "start_altitude": m.start_altitude, "max_energy": m.max_energy, "energy_k_v": m.energy_k_v,
            "energy_k_wp": m.energy_k_wp, "active": m.active, "no_fly_zones": [{k: v for k, v in z.__dict__.items() if v not in (None, "")} for z in m.no_fly_zones]}


def zone_geometry_text(m: Mission) -> str:
    """Plain-English zone geometry. The NL mission description only NAMES the zones; for the natural-language-only mode the geometry has to be
    given as text too, otherwise no model could avoid them. Contains the same numbers as the structured spec, no more."""
    parts = []
    for z in m.no_fly_zones:
        if z.kind == "circle":
            g = f"{z.id}: circle centred at ({z.center[0]:g}, {z.center[1]:g}) with radius {z.radius:g}"
        else:
            g = f"{z.id}: polygon with corners " + ", ".join(f"({v[0]:g}, {v[1]:g})" for v in z.vertices)
        if z.is_vertically_bounded:
            g += f", blocking altitudes {'-inf' if z.z_min is None else format(z.z_min, 'g')} to {'+inf' if z.z_max is None else format(z.z_max, 'g')} m only"
        parts.append(g)
    return "; ".join(parts)


def mission_block(m: Mission, mode: str = "spec") -> str:
    if mode not in PROMPT_MODES:
        raise ValueError(f"unknown prompt_mode {mode!r}; choose from {PROMPT_MODES}")
    spec = f"[MISSION_SPEC]{json.dumps(mission_spec(m))}[/MISSION_SPEC]"
    if mode == "nl_only":
        zg = zone_geometry_text(m)
        return (f"MISSION DESCRIPTION (free text; the only source of the mission constraints; x,y in map units, altitude in meters):\n{m.description}"
                + (f"\n\nZONE GEOMETRY (plain text): {zg}. Zones are closed regions: touching the boundary counts as entering." if zg else "")
                + f"\n\nMission id: {m.mission_id}")
    if mode == "spec_only":
        return f"MISSION SPECIFICATION (authoritative; units: x,y in map units, altitude in meters):\n{spec}"
    return (f"MISSION DESCRIPTION:\n{m.description}\n\n"
            f"MISSION SPECIFICATION (authoritative; units: x,y in map units, altitude in meters):\n{spec}")


def initial_prompt(m: Mission, mode: str = "spec", nonce: int | None = None) -> str:
    tail = "" if nonce is None else f"\n\n[SAMPLE_ID]{nonce}[/SAMPLE_ID] (independent attempt number {nonce}; do not copy earlier attempts)"
    return "[ROLE]initial\n" + mission_block(m, mode) + "\n\nProduce the flight plan JSON." + tail

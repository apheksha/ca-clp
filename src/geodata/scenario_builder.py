"""Build verifiable missions from geodata (authoritative airspace polygons, OSM buildings, SRTM terrain) -- see docs/GEODATA.md.

Config (YAML/dict):
  mission_id, origin: [lat, lon], bbox_m: [xmin, ymin, xmax, ymax]  (local metres around origin; default +-3000 m)
  start: [lat, lon]; destination: [lat, lon]; waypoints: [{id, lat, lon}]; ordering: [[id, id]]
  airspace: path.geojson | null; buildings: path.json | null; hgt: [paths] | null; terrain_cell_m: 30
  altitude: {min: m, max: m}     (same datum as terrain; defaults derived from terrain + min_agl/max_agl)
  min_agl, max_agl, max_distance_m, position_tolerance_m (default 5), vehicle, payload_kg, wind: [east, north], max_energy_wh
The result is a Mission in METRES (unit_m = 1) plus the feasibility oracle's verdict; infeasible/unknown missions are returned, not hidden."""
from __future__ import annotations
import json
import os
from src.geodata.projection import LocalFrame
from src.geodata.airspace import load_airspace_geojson
from src.geodata.osm import load_osm_buildings
from src.geodata.srtm import read_hgt, terrain_from_tiles
from src.models.schemas import Mission, RequiredWaypoint, ALL_FAMILIES


def build_geo_mission(cfg: dict, base_dir: str = "."):
    p = lambda x: x if os.path.isabs(x) else os.path.join(base_dir, x)
    lat0, lon0 = cfg["origin"]
    frame = LocalFrame(lat0, lon0)
    bbox = tuple(cfg.get("bbox_m", (-3000, -3000, 3000, 3000)))
    terrain = None
    if cfg.get("hgt"):
        terrain = terrain_from_tiles([read_hgt(p(h)) for h in cfg["hgt"]], frame, bbox, cfg.get("terrain_cell_m", 30.0))
    zones = []
    if cfg.get("airspace"):
        zones += load_airspace_geojson(p(cfg["airspace"]), frame, terrain, cfg.get("simplify_m", 10.0), bbox)
    if cfg.get("buildings"):
        zones += load_osm_buildings(p(cfg["buildings"]), frame, terrain, bbox_m=bbox, max_buildings=cfg.get("max_buildings"))
    s = frame.to_xy(*cfg["start"])
    d = frame.to_xy(*cfg["destination"])
    req = [RequiredWaypoint(w["id"], *frame.to_xy(w["lat"], w["lon"])) for w in cfg.get("waypoints", [])]
    ground = (lambda xy: terrain.height_at(*xy)) if terrain is not None else (lambda xy: 0.0)
    min_agl, max_agl = float(cfg.get("min_agl", 0.0)), cfg.get("max_agl")
    alt = cfg.get("altitude") or {}
    if terrain is not None:
        gmin, gmax = min(min(r) for r in terrain.heights), max(max(r) for r in terrain.heights)
        lo = alt.get("min", gmin + min_agl)
        hi = alt.get("max", gmax + (max_agl if max_agl is not None else 120.0))
    else:
        lo, hi = alt.get("min", 30.0), alt.get("max", 120.0)
    m = Mission(cfg.get("mission_id", "geo"), cfg.get("description", ""), list(s), list(d), req, float(lo), float(hi), float(cfg.get("max_distance_m", 1e9)), zones,
                [list(x) for x in cfg.get("ordering", [])], position_tolerance=float(cfg.get("position_tolerance_m", 5.0)), start_altitude=ground(s), unit_m=1.0,
                active=list(ALL_FAMILIES))
    if cfg.get("vehicle"):
        m.vehicle, m.payload_kg, m.wind = cfg["vehicle"], float(cfg.get("payload_kg", 0.0)), cfg.get("wind")
    if cfg.get("max_energy_wh") is not None:
        m.energy_model, m.max_energy = "physical", float(cfg["max_energy_wh"])
        if not cfg.get("vehicle"):
            raise ValueError("max_energy_wh requires a vehicle profile")
    if terrain is not None:
        m.terrain, m.min_agl, m.max_agl = terrain.to_dict(), min_agl, max_agl
        m.terrain_exempt_m = float(cfg.get("terrain_exempt_m", 150.0))   # take-off / landing happen at ground level
    m.validate()
    from src.benchmark.feasibility import assess
    return m, assess(m, certify=False)


def mission_to_scenario(m: Mission, feas, scenario_id: str, difficulty: str = "geo", source: str = "geodata") -> dict:
    return {"scenario_id": scenario_id, "difficulty": difficulty, "mission": m.to_dict(),
            "metadata": {"source": source, "feasibility": feas.status, "witness_distance": feas.witness_distance, "feasibility_reason": feas.reason,
                         "oracle_guarantee": feas.guarantee, "n_zones": len(m.no_fly_zones), "has_terrain": m.terrain is not None}}

#!/usr/bin/env python
"""Generates the SYNTHETIC geodata fixtures in data/geodata_samples/ (they exercise the loaders; they are NOT real airspace, buildings or terrain)."""
import json, math, os, random, struct, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.geodata.projection import LocalFrame

out = "data/geodata_samples"
os.makedirs(out, exist_ok=True)
rng = random.Random(1)
N = 301                                    # samples per degree side (~370 m cells): enough for the tests, small file
rows = []
for r in range(N):
    lat = 48.0 - r / (N - 1)
    row = []
    for c in range(N):
        lon = 8.0 + c / (N - 1)
        h = 400 + 120 * math.sin((lat - 47) * 2 * math.pi * 12) * math.cos((lon - 8) * 2 * math.pi * 10) + 60 * math.sin((lon - 8) * 2 * math.pi * 31)
        row.append(int(h))
    rows.append(row)
rows[150][150] = -32768                    # a void cell to test void filling
open(os.path.join(out, "N47E008.hgt"), "wb").write(struct.pack(f">{N * N}h", *[v for row in rows for v in row]))

frame = LocalFrame(47.5, 8.5)
def ll(x, y):
    lat, lon = frame.to_latlon(x, y)
    return [round(lon, 6), round(lat, 6)]
feats = [
  {"type": "Feature", "properties": {"name": "CTR TEST", "lowerLimit": {"value": 0, "unit": "ft", "referenceDatum": "GND"}, "upperLimit": {"value": 3500, "unit": "ft", "referenceDatum": "MSL"}},
   "geometry": {"type": "Polygon", "coordinates": [[ll(-2500, -2500), ll(-500, -2500), ll(-500, -1000), ll(-1500, -1000), ll(-1500, 500), ll(-2500, 500), ll(-2500, -2500)]]}},   # L-shaped (concave)
  {"type": "Feature", "properties": {"name": "RESTRICTED TEST", "radius_m": 450, "lower_m": 0, "upper_m": 5000}, "geometry": {"type": "Point", "coordinates": ll(1200, 800)}},
  {"type": "Feature", "properties": {"name": "TMA HIGH", "lowerLimit": {"value": 95, "unit": "FL", "referenceDatum": "STD"}, "upperLimit": {"value": 195, "unit": "FL", "referenceDatum": "STD"}},
   "geometry": {"type": "Polygon", "coordinates": [[ll(-3000, 1500), ll(3000, 1500), ll(3000, 3000), ll(-3000, 3000), ll(-3000, 1500)]]}},
  {"type": "Feature", "properties": {"name": "HOLE TEST", "lower_m": 0, "upper_m": 5000},
   "geometry": {"type": "Polygon", "coordinates": [[ll(300, -2800), ll(2200, -2800), ll(2200, -1400), ll(300, -1400), ll(300, -2800)], [ll(800, -2400), ll(1500, -2400), ll(1500, -1800), ll(800, -1800), ll(800, -2400)]]}},
]
json.dump({"type": "FeatureCollection", "features": feats, "note": "SYNTHETIC test fixture, not real airspace"}, open(os.path.join(out, "airspace_sample.geojson"), "w"), indent=1)

els = []
for i in range(24):
    cx, cy = rng.uniform(-2500, 2500), rng.uniform(-2500, 1200)
    w, h = rng.uniform(15, 60), rng.uniform(15, 60)
    tags = {"building": "yes"}
    if i % 3 == 0: tags["height"] = f"{rng.choice([12, 25, 45, 70])}"
    elif i % 3 == 1: tags["building:levels"] = str(rng.randint(2, 15))
    pts = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2), (cx - w / 2, cy - h / 2)]
    els.append({"type": "way", "id": 1000 + i, "tags": tags, "geometry": [{"lat": frame.to_latlon(*p)[0], "lon": frame.to_latlon(*p)[1]} for p in pts]})
json.dump({"version": 0.6, "note": "SYNTHETIC test fixture, not real OpenStreetMap data", "elements": els}, open(os.path.join(out, "buildings_sample.osm.json"), "w"))
cfg = {"mission_id": "geo_sample", "origin": [47.5, 8.5], "bbox_m": [-3200, -3200, 3200, 3200], "start": list(frame.to_latlon(-2800, -2800)), "destination": list(frame.to_latlon(2800, 1000)),
       "waypoints": [{"id": "W1", "lat": frame.to_latlon(-300, -200)[0], "lon": frame.to_latlon(-300, -200)[1]}],
       "airspace": "airspace_sample.geojson", "buildings": "buildings_sample.osm.json", "hgt": ["N47E008.hgt"], "terrain_cell_m": 60, "min_agl": 30, "max_agl": 120,
       "max_distance_m": 14000, "vehicle": "generic_quad_large", "max_energy_wh": 150, "wind": [2.0, 1.0]}
import yaml
yaml.safe_dump(cfg, open(os.path.join(out, "geo_sample.yaml"), "w"))
print("fixtures written to", out)

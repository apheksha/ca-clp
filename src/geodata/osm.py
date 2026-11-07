"""Building / obstacle footprints from OpenStreetMap data (Overpass JSON `out geometry;` of ways with a `building` tag, or GeoJSON polygons) as 3D obstacle prisms.

Height: tag `height` (m) > `building:levels` * level_height_m > default_height_m. With a terrain grid the prism top is ground_at_centroid + height (so z uses the plan's
datum). OSM data is crowd-sourced: heights are often missing (the default is a guess) and footprints can be incomplete -- treat results as a sample, not a survey.
Fetching is NOT done here (no network access in this code base): export with Overpass Turbo / `overpass` API and pass the file."""
from __future__ import annotations
import json
import re
from src.geodata.projection import LocalFrame, simplify_ring
from src.models.schemas import NoFlyZone


def _height(tags: dict, default_h: float, level_h: float) -> tuple[float, str]:
    h = tags.get("height")
    if h is not None:
        m = re.match(r"\s*([0-9.]+)", str(h))
        if m:
            return float(m.group(1)) * (0.3048 if "ft" in str(h) else 1.0), "height tag"
    lv = tags.get("building:levels")
    if lv is not None:
        try:
            return float(lv) * level_h, "levels"
        except ValueError:
            pass
    return default_h, "default"


def load_osm_buildings(path: str, frame: LocalFrame, terrain=None, default_height_m: float = 10.0, level_height_m: float = 3.0, min_area_m2: float = 20.0,
                       simplify_m: float = 1.0, bbox_m: tuple | None = None, max_buildings: int | None = None) -> list[NoFlyZone]:
    with open(path) as fh:
        d = json.load(fh)
    rings = []   # (id, [(lon, lat)...], tags)
    if "elements" in d:
        for el in d["elements"]:
            if el.get("type") == "way" and "geometry" in el and "building" in (el.get("tags") or {}):
                rings.append((str(el["id"]), [(p["lon"], p["lat"]) for p in el["geometry"]], el.get("tags", {})))
    else:
        for i, f in enumerate(d.get("features", [])):
            g = f.get("geometry") or {}
            if g.get("type") == "Polygon":
                rings.append((str(f.get("id", i)), [tuple(c) for c in g["coordinates"][0]], f.get("properties") or {}))
    out, n_default = [], 0
    for wid, ring, tags in rings:
        pts = [frame.to_xy(lat, lon) for lon, lat in ring]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
        if len(pts) < 3:
            continue
        area = abs(sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))) / 2
        if area < min_area_m2:
            continue
        pts = simplify_ring(pts, simplify_m)
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        if bbox_m is not None and not (bbox_m[0] <= cx <= bbox_m[2] and bbox_m[1] <= cy <= bbox_m[3]):
            continue
        h, src = _height(tags, default_height_m, level_height_m)
        n_default += src == "default"
        base = terrain.height_at(cx, cy) if terrain is not None else 0.0
        zid = "BLD_" + re.sub(r"[^A-Za-z0-9_\-]", "_", wid)[-28:]     # Overpass-turbo GeoJSON ids look like "way/37931109"; zone ids allow [A-Za-z0-9_-]{1,32}
        z = NoFlyZone(zid, "polygon", vertices=[[x, y] for x, y in pts], category="obstacle", description=f"building ({src})", z_min=None, z_max=base + h)
        out.append(z.validate())
        if max_buildings and len(out) >= max_buildings:
            break
    out_note = {"n_buildings": len(out), "n_default_height": n_default}
    for z in out[:1]:
        z.description += f" [set stats: {out_note}]"
    return out

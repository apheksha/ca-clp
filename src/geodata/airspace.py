"""Import of AUTHORITATIVE airspace restriction data (as GeoJSON) into NoFlyZone objects with real boundaries and vertical limits.

This module reads files; it does not ship or fetch authoritative data. Typical sources (check each licence): national AIP / NOTAM-derived datasets (e.g. OpenAIP
exports, FAA NASR/UAS Facility Map, EUROCONTROL/EAD, national UAS-geozone services). Export their polygons as GeoJSON (WGS84 lon/lat) and point this loader at the file.

Accepted features:
  Polygon / MultiPolygon       -> polygonal zone (outer ring; holes are IGNORED, i.e. the zone is treated as fully blocked: conservative)
  Point + properties.radius_m  -> circular zone
Vertical limits (properties): lower_m / upper_m, or lowerLimit / upperLimit = {value, unit: m|ft|FL, referenceDatum: GND|AGL|MSL|STD}. FL (flight level) = value*30.48 m.
AGL/GND limits need `terrain` to be converted to the plan's datum; the zone's ground height is then taken at its centroid (documented approximation).
Returned zones carry category 'nfz' and the source feature name in `description`."""
from __future__ import annotations
import json
import math
from src.geodata.projection import LocalFrame, simplify_ring
from src.models.schemas import NoFlyZone

FT = 0.3048


def _to_m(lim) -> tuple[float | None, str]:
    """Return (metres or None, datum)."""
    if lim is None:
        return None, "MSL"
    if isinstance(lim, (int, float)):
        return float(lim), "MSL"
    v, u = lim.get("value"), str(lim.get("unit", "m")).lower()
    datum = str(lim.get("referenceDatum", lim.get("datum", "MSL"))).upper()
    if v is None:
        return None, datum
    if u in ("ft", "feet"):
        v = v * FT
    elif u in ("fl", "flightlevel"):
        v = v * 100 * FT
    return float(v), datum


def load_airspace_geojson(path: str, frame: LocalFrame, terrain=None, simplify_m: float = 10.0, bbox_m: tuple | None = None, ceiling_unlimited_m: float = 5000.0) -> list[NoFlyZone]:
    """bbox_m = (xmin, ymin, xmax, ymax) in local metres: zones entirely outside are dropped. Raises ValueError on malformed geometry."""
    with open(path) as fh:
        d = json.load(fh)
    feats = d["features"] if d.get("type") == "FeatureCollection" else [d]
    zones: list[NoFlyZone] = []
    for fi, f in enumerate(feats):
        g, pr = f.get("geometry") or {}, f.get("properties") or {}
        name = str(pr.get("name") or pr.get("id") or f"zone{fi}")
        lo_raw = pr.get("lower_m", pr.get("lowerLimit"))
        hi_raw = pr.get("upper_m", pr.get("upperLimit"))
        lo, dlo = _to_m(lo_raw)
        hi, dhi = _to_m(hi_raw)
        polys = []
        if g.get("type") == "Polygon":
            polys = [g["coordinates"][0]]
        elif g.get("type") == "MultiPolygon":
            polys = [p[0] for p in g["coordinates"]]
        elif g.get("type") == "Point" and pr.get("radius_m"):
            x, y = frame.to_xy(g["coordinates"][1], g["coordinates"][0])
            polys = None
        else:
            raise ValueError(f"feature {name!r}: unsupported geometry {g.get('type')!r}")
        pieces = []
        if polys is None:
            pieces.append(("circle", (x, y), float(pr["radius_m"])))
        else:
            for pi, ring in enumerate(polys):
                pts = [frame.to_xy(lat, lon) for lon, lat in ring]
                if pts[0] == pts[-1]:
                    pts = pts[:-1]
                if len(pts) < 3:
                    raise ValueError(f"feature {name!r}: ring with fewer than 3 vertices")
                pts = simplify_ring(pts, simplify_m)
                pieces.append(("polygon", pts, pi))
        for k, piece in enumerate(pieces):
            zid = f"AS_{name.replace(' ', '_')}" + (f"_{k}" if len(pieces) > 1 else "")
            if piece[0] == "circle":
                z = NoFlyZone(zid, "circle", center=[piece[1][0], piece[1][1]], radius=piece[2], description=name)
                cx, cy = piece[1]
            else:
                z = NoFlyZone(zid, "polygon", vertices=[[x, y] for x, y in piece[1]], description=name)
                cx = sum(p[0] for p in piece[1]) / len(piece[1])
                cy = sum(p[1] for p in piece[1]) / len(piece[1])
            ground = terrain.height_at(cx, cy) if terrain is not None else 0.0
            z_min = None if lo is None else (lo + ground if dlo in ("GND", "AGL", "SFC") else lo)
            z_max = None if hi is None else (hi + ground if dhi in ("GND", "AGL", "SFC") else hi)
            if z_min is not None and z_min <= (ground if terrain is not None else 0.0) + 1e-9:
                z_min = None                                  # floor at the ground: unbounded below
            if z_max is not None and z_max >= ceiling_unlimited_m:
                z_max = None
            z.z_min, z.z_max = z_min, z_max
            z.validate()
            if bbox_m is not None:
                xs = [v[0] for v in z.vertices] if z.kind == "polygon" else [cx - z.radius, cx + z.radius]
                ys = [v[1] for v in z.vertices] if z.kind == "polygon" else [cy - z.radius, cy + z.radius]
                if max(xs) < bbox_m[0] or min(xs) > bbox_m[2] or max(ys) < bbox_m[1] or min(ys) > bbox_m[3]:
                    continue
            zones.append(z)
    return zones

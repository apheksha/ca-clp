#!/usr/bin/env python
"""Build a 3D urban mission set from real geodata (v12): OpenStreetMap building footprints + SRTM terrain.

  python scripts/build_bengaluru_set.py --buildings data/geodata_bengaluru/bengaluru_buildings.geojson data/geodata_bengaluru/bengaluru_buildings_east.json \
      --hgt data/geodata_bengaluru/N13E077.hgt --out benchmark/bengaluru --n-per-difficulty 10

Input formats: Overpass-turbo GeoJSON (Polygon features) and raw Overpass JSON (`out geometry;`, ways with a building tag); several files are merged.

What is real: building footprints and the tagged heights of some buildings (OSM, ODbL), ground elevation (SRTM 1 arc-second), the names and positions of the
sites. What is modeled: the altitude band (above ground, followed along every leg), a default height for buildings without height data, distance/energy budgets.

Obstacles. A building is a prism from the ground to its top over the CONVEX HULL of its footprint (a superset of the footprint). A building whose top is below the
band floor cannot be hit by a plan that respects the band, so it is left out. Of the remaining buildings a mission contains every one that a VALID plan could reach:
a plan must visit START, the required sites (chained order) and DEST and fly at most max_distance, so between consecutive points P_i, P_i+1 it stays inside the
ellipse with foci P_i, P_i+1 and major axis d_i + e, where e = max_distance - (length of the straight polyline). That ellipse lies within
b_i = sqrt(e (2 d_i + e)) / 2 of the segment; every building within b_i + 30 m of a segment is included. max_distance depends on the oracle witness, which depends on
the included buildings, so inclusion and witness are iterated until no building is added. Buildings outside that region can only be reached by plans that already
violate the distance budget, so omitting them never changes a verdict. The terrain grid is cropped to the same region.
Difficulty: easy 1 required site, medium 2 ordered sites + energy budget, hard 3 ordered sites + energy budget + at least one leg blocked at the band floor."""
import argparse, hashlib, json, math, os, random, signal, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.geodata.projection import LocalFrame
from src.geodata.osm import load_osm_buildings
from src.geodata.srtm import read_hgt, terrain_from_tiles
from src.models.schemas import Mission, RequiredWaypoint, ALL_FAMILIES
from src.models.terrain import TerrainGrid
from src.benchmark.feasibility import assess
from src.benchmark.generator import nl_description, STYLES
from src.verifier.verifier import resources

DIFF = {"easy": {"n_req": 1, "slack": 1.40, "e_slack": None, "min_block": 0},
        "medium": {"n_req": 2, "slack": 1.20, "e_slack": 1.30, "min_block": 0},
        "hard": {"n_req": 3, "slack": 1.08, "e_slack": 1.12, "min_block": 1}}
ENERGY_K_V = 30.0
INCLUDE_MARGIN_M = 30.0


class _Slow(Exception):
    pass


def _assess_limited(m, seconds):
    """The oracle can be slow on terrain-following cases with many obstacles; such candidates are skipped (and counted), not waited for.
    Without SIGALRM (Windows) there is no time limit."""
    if not hasattr(signal, "SIGALRM"):
        return assess(m)

    def _h(*_):
        raise _Slow()
    old = signal.signal(signal.SIGALRM, _h)
    signal.alarm(int(seconds))
    try:
        return assess(m)
    except _Slow:
        return None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _footprints(path):
    """(raw id, name, [(lon, lat), ...]) for every building in a GeoJSON or Overpass-JSON file (same id convention as src/geodata/osm.py)."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    if "elements" in d:
        for el in d["elements"]:
            if el.get("type") == "way" and "geometry" in el and "building" in (el.get("tags") or {}):
                yield str(el["id"]), (el.get("tags") or {}).get("name"), [(p["lon"], p["lat"]) for p in el["geometry"]]
    else:
        for i, f in enumerate(d.get("features", [])):
            g = f.get("geometry") or {}
            if g.get("type") == "Polygon":
                yield str(f.get("id", i)), (f.get("properties") or {}).get("name"), [tuple(c[:2]) for c in g["coordinates"][0]]


def _zid(raw):
    return "BLD_" + "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in raw)[-28:]


def _convex_hull(pts):
    """Monotone-chain convex hull (counter-clockwise). The hull contains the footprint, so a plan that avoids the hull avoids the building."""
    pts = sorted(set((float(p[0]), float(p[1])) for p in pts))
    if len(pts) < 3:
        return pts
    cr = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cr(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cr(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return lo[:-1] + up[:-1]


def _seg_dist_lb(p, q, z):
    """Lower bound of the distance between segment pq and zone z (distance to the bounding circle of the polygon)."""
    cx, cy = z.centroid()
    r = max(math.hypot(v[0] - cx, v[1] - cy) for v in z.vertices)
    dx, dy = q[0] - p[0], q[1] - p[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((cx - p[0]) * dx + (cy - p[1]) * dy) / L2))
    return max(0.0, math.hypot(p[0] + t * dx - cx, p[1] + t * dy - cy) - r)


def reachable(tall, pts, max_distance):
    """Buildings that a plan visiting pts in order with total length <= max_distance could touch (see module docstring)."""
    legs = [math.dist(p, q) for p, q in zip(pts, pts[1:])]
    e = max(0.0, max_distance - sum(legs))
    out = []
    for z in tall:
        if any(_seg_dist_lb(p, q, z) <= math.sqrt(e * (2 * d + e)) / 2 + INCLUDE_MARGIN_M for (p, q), d in zip(zip(pts, pts[1:]), legs)):
            out.append(z)
    radius = max(math.sqrt(e * (2 * d + e)) / 2 for d in legs) + INCLUDE_MARGIN_M
    return out, radius


def crop(terrain, box):
    """Sub-grid of `terrain` covering box = (xmin, ymin, xmax, ymax) (clipped to the grid)."""
    c0 = max(0, int(math.floor((box[0] - terrain.x0) / terrain.cell)))
    r0 = max(0, int(math.floor((box[1] - terrain.y0) / terrain.cell)))
    c1 = min(terrain.cols - 1, int(math.ceil((box[2] - terrain.x0) / terrain.cell)))
    r1 = min(terrain.rows - 1, int(math.ceil((box[3] - terrain.y0) / terrain.cell)))
    return TerrainGrid(terrain.x0 + c0 * terrain.cell, terrain.y0 + r0 * terrain.cell, terrain.cell, [row[c0:c1 + 1] for row in terrain.heights[r0:r1 + 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buildings", nargs="+", required=True)
    ap.add_argument("--hgt", nargs="+", required=True)
    ap.add_argument("--out", default="benchmark/bengaluru")
    ap.add_argument("--n-per-difficulty", type=int, default=10)
    ap.add_argument("--min-agl", type=float, default=20.0)
    ap.add_argument("--max-agl", type=float, default=120.0, help="120 m above ground is the usual open-category ceiling")
    ap.add_argument("--default-height", type=float, default=10.0, help="assumed height of buildings without a height/levels tag (m)")
    ap.add_argument("--margin-m", type=float, default=150.0, help="sites must lie this far inside the data extent")
    ap.add_argument("--min-leg-m", type=float, default=800.0, help="minimum start-destination distance")
    ap.add_argument("--max-leg-m", type=float, default=4000.0, help="maximum start-destination distance")
    ap.add_argument("--max-obstacles", type=int, default=60, help="candidates whose reachable region holds more buildings are skipped (and counted)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--oracle-timeout", type=float, default=300.0, help="seconds per oracle call; slower candidates are skipped (and counted)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    foot = []
    for path in a.buildings:
        foot += [(path, rid, nm, ring) for rid, nm, ring in _footprints(path)]
    lats = [c[1] for *_, ring in foot for c in ring]
    lons = [c[0] for *_, ring in foot for c in ring]
    lat0, lon0 = round((min(lats) + max(lats)) / 2, 6), round((min(lons) + max(lons)) / 2, 6)
    frame = LocalFrame(lat0, lon0)
    x0, y0 = frame.to_xy(min(lats), min(lons))
    x1, y1 = frame.to_xy(max(lats), max(lons))
    bbox = (math.floor(x0), math.floor(y0), math.ceil(x1), math.ceil(y1))
    terrain = terrain_from_tiles([read_hgt(h) for h in a.hgt], frame, bbox, 30.0)
    terrain.heights = [[round(h, 1) for h in row] for row in terrain.heights]      # 0.1 m is far below SRTM accuracy; keeps the scenario files small
    allb, seen_keys = [], set()
    for path in a.buildings:
        for z in load_osm_buildings(path, frame, terrain, default_height_m=a.default_height, bbox_m=bbox):
            c = z.centroid()
            key = (round(c[0], 1), round(c[1], 1), round(z.area(), 0))      # the same way exported twice (overlapping extracts) is kept once
            if key in seen_keys:
                continue
            seen_keys.add(key)
            allb.append(z)
    tall = [z for z in allb if z.z_max - terrain.height_at(*z.centroid()) >= a.min_agl - 1e-9]
    n_default = sum("(default)" in z.description for z in allb)
    for z in tall:
        z.description = z.description.split(" [set stats")[0] + ", convex hull"
        z.z_max = round(z.z_max, 1)
        z.vertices = [[round(x, 1), round(y, 1)] for x, y in _convex_hull(z.vertices)]
        z.validate()
    print(f"{len(allb)} buildings ({n_default} without height data, assumed {a.default_height:g} m); {len(tall)} reach the band floor ({a.min_agl:g} m above ground)", flush=True)

    tall_ids = {z.id for z in tall}
    by_id = {z.id: z for z in allb}
    sites, seen = [], set()
    for path, rid, nm, ring in foot:
        if not nm or nm in seen:
            continue
        z = by_id.get(_zid(rid))
        if z is None or z.id in tall_ids:
            continue
        cx, cy = z.centroid()
        if not (bbox[0] + a.margin_m <= cx <= bbox[2] - a.margin_m and bbox[1] + a.margin_m <= cy <= bbox[3] - a.margin_m):
            continue
        if any(t.contains((cx, cy)) for t in tall):
            continue
        seen.add(nm)
        sites.append({"name": nm, "x": round(cx, 1), "y": round(cy, 1)})
    print(f"{len(sites)} named sites", flush=True)
    meta_common = {"source": "osm+srtm", "region": "bengaluru", "region_center": [lat0, lon0],
                   "buildings_sha256": {os.path.basename(p): sha256(p) for p in a.buildings}, "hgt_sha256": {os.path.basename(h): sha256(h) for h in a.hgt},
                   "n_buildings_total": len(allb), "n_buildings_default_height": n_default, "n_buildings_reaching_band": len(tall),
                   "default_height_m": a.default_height, "min_agl": a.min_agl, "max_agl": a.max_agl,
                   "obstacle_rule": "convex hulls of all buildings that reach the band floor and lie within the region reachable under the distance budget (+30 m)",
                   "assumptions": "buildings without height data assumed default_height_m; band followed over SRTM terrain; local equirectangular projection"}
    for diff in ("easy", "medium", "hard"):
        cfg, made, k = DIFF[diff], 0, 0
        skipped = {"oracle_timeout": 0, "too_many_obstacles": 0, "not_feasible": 0, "too_few_blocked_legs": 0, "leg_length": 0}
        while made < a.n_per_difficulty and k < 4000:
            k += 1
            sid = f"blr_{diff}_{made:02d}"
            rng = random.Random(f"{a.seed}|{sid}|{k}")
            picks = rng.sample(sites, cfg["n_req"] + 2)
            start, dest, mids = picks[0], picks[-1], picks[1:-1]
            if not a.min_leg_m <= math.hypot(start["x"] - dest["x"], start["y"] - dest["y"]) <= a.max_leg_m:
                skipped["leg_length"] += 1
                continue
            mids = sorted(mids, key=lambda s: (s["x"] - start["x"]) * (dest["x"] - start["x"]) + (s["y"] - start["y"]) * (dest["y"] - start["y"]))
            reqs = [RequiredWaypoint(chr(65 + i), s["x"], s["y"]) for i, s in enumerate(mids)]
            pts = [(start["x"], start["y"])] + [(r.x, r.y) for r in reqs] + [(dest["x"], dest["y"])]
            blocked = sum(any(z.intersects_segment(p, q) for z in tall) for p, q in zip(pts, pts[1:]))
            if blocked < cfg["min_block"]:
                skipped["too_few_blocked_legs"] += 1
                continue
            straight = sum(math.dist(p, q) for p, q in zip(pts, pts[1:]))
            budget = straight * cfg["slack"]           # first guess; refined from the witness below
            zones, f, ok = None, None, False
            for _ in range(6):
                zones, radius = reachable(tall, pts, budget)
                if len(zones) > a.max_obstacles:
                    break
                box = (min(p[0] for p in pts) - radius - 200, min(p[1] for p in pts) - radius - 200, max(p[0] for p in pts) + radius + 200, max(p[1] for p in pts) + radius + 200)
                ter = crop(terrain, box)
                gmin, gmax = min(min(r) for r in ter.heights), max(max(r) for r in ter.heights)
                lo, hi = round(gmin + a.min_agl, 1), round(gmax + a.max_agl, 1)
                m = Mission(sid, "", [start["x"], start["y"]], [dest["x"], dest["y"]], reqs, lo, hi, 1e9, list(zones),
                            [[reqs[i].id, reqs[i + 1].id] for i in range(len(reqs) - 1)], position_tolerance=5.0,
                            start_altitude=ter.height_at(start["x"], start["y"]), unit_m=1.0, active=[x for x in ALL_FAMILIES if x != "energy"])
                m.terrain, m.min_agl, m.max_agl, m.terrain_exempt_m = ter.to_dict(), a.min_agl, a.max_agl, 150.0
                m.validate()
                f = _assess_limited(m, a.oracle_timeout)
                if f is None or f.status != "FEASIBLE":
                    break
                new_budget = round(f.witness_distance * cfg["slack"], 0)
                if reachable(tall, pts, new_budget)[0] == zones:      # fixed point: every building a valid plan could reach is in the mission
                    budget, ok = new_budget, True
                    break
                budget = max(budget, new_budget)
            if zones is not None and len(zones) > a.max_obstacles:
                skipped["too_many_obstacles"] += 1
                continue
            if f is None:
                skipped["oracle_timeout"] += 1
                continue
            if not ok:
                skipped["not_feasible"] += 1
                continue
            m.max_distance = budget
            if cfg["e_slack"]:
                m.energy_k_v = ENERGY_K_V
                m.max_energy = round(resources(f.witness_plan, m)["energy"] * cfg["e_slack"], 0)
                m.active = list(m.active) + ["energy"]
            labels = {r.id: s["name"] for r, s in zip(reqs, mids)}
            labels.update({"START": start["name"], "DEST": dest["name"]})
            style = STYLES[made % 3]
            m.description = (nl_description(m, style, labels, unit="meters") + f" (START is {start['name']}; DEST is {dest['name']}.) Coordinates are meters east/north of "
                             f"{lat0:.4f} N, {lon0:.4f} E; altitudes are meters above sea level; the ground here lies between {gmin:.0f} and {gmax:.0f} m, and every leg must stay "
                             f"{a.min_agl:g}-{a.max_agl:g} m above the ground.")
            f = _assess_limited(m, a.oracle_timeout)
            if f is None or f.status != "FEASIBLE":
                skipped["oracle_timeout" if f is None else "not_feasible"] += 1
                continue
            meta = {**meta_common, "feasibility": "FEASIBLE", "witness_distance": round(f.witness_distance, 1), "lower_bound": round(f.lower_bound, 1) if f.lower_bound else None,
                    "infeasible_kind": None, "n_zones": len(m.no_fly_zones), "n_polygon_zones": len(m.no_fly_zones), "n_obstacles": len(m.no_fly_zones),
                    "distance_slack": cfg["slack"], "energy_slack": cfg["e_slack"], "n_required": len(reqs), "active": list(m.active), "n_active": len(m.active),
                    "witness_energy": None, "nl_style": style, "site_names": labels, "n_blocked_legs": blocked, "altitude_band_width": hi - lo, "has_terrain": True,
                    "terrain_min_m": round(gmin, 1), "terrain_max_m": round(gmax, 1), "reachable_radius_m": round(radius, 1)}
            rec = {"scenario_id": sid, "difficulty": f"blr_{diff}", "base_seed": a.seed, "attempt": k, "mission": m.to_dict(), "metadata": meta}
            with open(os.path.join(a.out, sid + ".json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1)
            made += 1
            print(f"  {sid}: {len(m.no_fly_zones)} buildings, witness {f.witness_distance / 1000:.2f} km, {blocked} blocked legs", flush=True)
        print(f"[ok] {diff}: {made} missions after {k} attempts; skipped {skipped}", flush=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(a.buildings[0])), f"build_log_{diff}.json"), "w", encoding="utf-8") as fh:   # not in --out: every .json there is read as a scenario
            json.dump({"made": made, "attempts": k, "skipped": skipped}, fh, indent=1)


if __name__ == "__main__":
    main()

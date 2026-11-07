"""AIRPORT-ANCHORED scenarios with MODELLED restriction zones (NOT authoritative airspace / not regulatory no-fly data), built from the OurAirports open dataset (public domain / Unlicense):
https://davidmegginson.github.io/ourairports-data/airports.csv   (document file date + sha256 in every scenario).

What is REAL: airport/heliport identities and coordinates, names, types, scheduled-service flags.
What is MODELLED (assumptions, not legal advice): protection-zone radius per airport type, altitude band (<=120 m AGL, the common
EASA/FAA ceiling), distance budget, and the choice of which heliports are mission sites. Real regulations differ by country.
Zones are circles around real airports; sites (start/destination/required waypoints) are real heliports and non-scheduled small airfields
that lie outside every zone. Coordinates are projected to a local metric plane (equirectangular, error << 1% within ~60 km)."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import time
import urllib.request
from src.models.schemas import Mission, RequiredWaypoint, NoFlyZone
from src.benchmark.feasibility import assess
from src.benchmark.generator import nl_description, STYLES

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
EARTH_R = 6371008.8
# name: (lat, lon, radius_km). Add your own; regions with too few sites/zones are skipped with a message.
REGIONS = {
    "new_york": (40.72, -73.95, 28), "san_francisco": (37.65, -122.25, 28), "los_angeles": (34.00, -118.30, 30),
    "london": (51.50, -0.25, 30), "frankfurt": (50.05, 8.60, 30), "bengaluru": (13.00, 77.65, 30),
    "delhi": (28.60, 77.15, 30), "singapore": (1.33, 103.90, 25), "sydney": (-33.90, 151.15, 30),
    "dubai": (25.20, 55.35, 30), "chicago": (41.90, -87.70, 30), "tokyo": (35.65, 139.75, 30),
}
ZONE_RADIUS_M = {"large_airport": 8000.0, "medium_airport": 5000.0, "small_airport": 3000.0}  # small only if scheduled service
DIFF = {"easy": dict(n_req=2, slack=1.4, band=(20, 120), min_block=1, e_slack=None), "medium": dict(n_req=3, slack=1.2, band=(30, 100), min_block=1, e_slack=1.3),
        "hard": dict(n_req=4, slack=1.08, band=(40, 80), min_block=2, e_slack=1.12)}
ENERGY_K_V = 30.0   # modelled: 1 m of climb costs as much battery as 30 m of level flight


def fetch(dest_dir: str = "data/real") -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "airports.csv")
    req = urllib.request.Request(URL, headers={"User-Agent": "uav-llm-closed-loop/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    info = {"url": URL, "sha256": sha256(path), "license": "OurAirports data is released into the public domain (Unlicense)",
            "downloaded": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(os.path.join(dest_dir, "airports.source.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)
    return path


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def project(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    return (math.radians(lon - lon0) * math.cos(math.radians(lat0)) * EARTH_R, math.radians(lat - lat0) * EARTH_R)


def load_airports(path: str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["lat"], r["lon"] = float(r["latitude_deg"]), float(r["longitude_deg"])
            except (ValueError, KeyError):
                continue
            out.append(r)
    return out


def region_data(rows: list[dict], center: tuple, radius_km: float, max_zones: int = 8) -> dict:
    lat0, lon0 = center
    zones, sites = [], []
    for r in rows:
        x, y = project(r["lat"], r["lon"], lat0, lon0)
        d = math.hypot(x, y)
        if d > radius_km * 1000 or r.get("type") in ("closed", "balloonport"):
            continue
        sched = r.get("scheduled_service") == "yes"
        t = r.get("type")
        rad = ZONE_RADIUS_M.get(t) if (t in ("large_airport", "medium_airport") or (t == "small_airport" and sched)) else None
        item = dict(name=r.get("name", r.get("ident")), ident=r.get("ident"), type=t, x=x, y=y, d=d, radius=rad)
        if rad:
            zones.append(item)
        elif t in ("heliport", "small_airport"):
            sites.append(item)
    zones.sort(key=lambda z: z["d"])
    zones = zones[:max_zones]
    sites = [s for s in sites if all(math.hypot(s["x"] - z["x"], s["y"] - z["y"]) > z["radius"] + 400 for z in zones)]
    return {"zones": zones, "sites": sites}


def _mission(sid, rd, picks, diff, style, rng):
    cfg = DIFF[diff]
    zones = [NoFlyZone(f"Z{i+1}", "circle", center=[round(z["x"], 1), round(z["y"], 1)], radius=z["radius"],
                       description=f"{z['name']} ({z['ident']}, {z['type']}) protection radius {z['radius']:.0f} m") for i, z in enumerate(rd["zones"])]
    start, dest, mids = picks[0], picks[-1], picks[1:-1]
    mids = sorted(mids, key=lambda s: (s["x"] - start["x"]) * (dest["x"] - start["x"]) + (s["y"] - start["y"]) * (dest["y"] - start["y"]))
    reqs = [RequiredWaypoint(chr(65 + i), round(s["x"], 1), round(s["y"], 1)) for i, s in enumerate(mids)]
    lo, hi = cfg["band"]
    m = Mission(sid, "", [round(start["x"], 1), round(start["y"], 1)], [round(dest["x"], 1), round(dest["y"], 1)], reqs, float(lo), float(hi), 1e9,
                zones, [[reqs[i].id, reqs[i + 1].id] for i in range(len(reqs) - 1)], position_tolerance=5.0).validate()
    labels = {r.id: s["name"] for r, s in zip(reqs, mids)}
    return m, labels, start["name"], dest["name"]


def build_real_scenarios(airports_csv: str, out_dir: str, meta_dir: str, regions: dict | None = None, per_region: int = 6,
                         base_seed: int = 2024, log=print) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    rows = load_airports(airports_csv)
    digest = sha256(airports_csv)
    out, idx = [], []
    for rname, (lat, lon, rad) in (regions or REGIONS).items():
        rd = region_data(rows, (lat, lon), rad)
        need = 2 + max(c["n_req"] for c in DIFF.values())
        if len(rd["zones"]) < 2 or len(rd["sites"]) < need:
            log(f"[skip] {rname}: {len(rd['zones'])} zones, {len(rd['sites'])} sites (need >=2 zones, >={need} sites)")
            continue
        made = 0
        for k in range(per_region * 20):
            if made >= per_region:
                break
            diff = ["easy", "medium", "hard"][made % 3]
            sid = f"real_{rname}_{made:02d}"
            rng = random.Random(f"{base_seed}|{sid}|{k}")
            picks = rng.sample(rd["sites"], DIFF[diff]["n_req"] + 2)
            if math.hypot(picks[0]["x"] - picks[-1]["x"], picks[0]["y"] - picks[-1]["y"]) < 0.35 * rad * 1000:
                continue
            m, labels, sn, dn = _mission(sid, rd, picks, diff, STYLES[made % 3], rng)
            f = assess(m)
            pts = [m.start_xy] + [r.xy for r in m.required_waypoints] + [m.dest_xy]
            blocked = sum(any(z.intersects_segment(a, b) for a, b in zip(pts, pts[1:])) for z in m.no_fly_zones)
            if f.status != "FEASIBLE" or blocked < DIFF[diff]["min_block"]:
                continue
            slack = DIFF[diff]["slack"]
            m.max_distance = round(f.witness_distance * slack, 0)
            if DIFF[diff]["e_slack"]:
                from src.verifier.verifier import resources
                m.energy_k_v = ENERGY_K_V
                m.max_energy = round(resources(f.witness_plan, m)["energy"] * DIFF[diff]["e_slack"], 0)
                m.active = list(m.active) + ["energy"] if "energy" not in m.active else m.active
            labels.update({"START": sn, "DEST": dn})
            m.description = nl_description(m, STYLES[made % 3], labels, unit="meters") + f" (START is {sn}; DEST is {dn}.) Coordinates are meters east/north of the region center."
            f = assess(m)
            if f.status != "FEASIBLE":
                continue
            meta = {"source": "ourairports", "region": rname, "region_center": [lat, lon], "airports_csv_sha256": digest, "n_zones": len(m.no_fly_zones),
                    "zone_density": round(sum(z.area() for z in m.no_fly_zones) / (math.pi * (rad * 1000) ** 2), 4), "distance_slack": slack,
                    "altitude_band_width": hi_lo(m), "n_required": len(m.required_waypoints), "feasibility": "FEASIBLE",
                    "witness_distance": round(f.witness_distance, 1), "lower_bound": round(f.lower_bound, 1), "infeasible_kind": None,
                    "n_polygon_zones": 0, "active": list(m.active), "n_active": len(m.active), "n_obstacles": 0, "witness_energy": None, "energy_slack": DIFF[diff]["e_slack"], "nl_style": STYLES[made % 3], "site_names": labels, "n_blocked_legs": blocked,
                    "assumptions": "zone radii by airport type, altitude band <=120 m AGL, equirectangular projection"}
            rec = {"scenario_id": sid, "difficulty": f"real_{diff}", "base_seed": base_seed, "attempt": k, "mission": m.to_dict(), "metadata": meta}
            with open(os.path.join(out_dir, sid + ".json"), "w") as fh:
                json.dump(rec, fh, indent=1)
            out.append(rec)
            idx.append({"scenario_id": sid, "difficulty": rec["difficulty"], **{a: b for a, b in meta.items() if a not in ("site_names", "region_center")}})
            made += 1
        log(f"[ok] {rname}: {made} scenarios ({len(rd['zones'])} real zones, {len(rd['sites'])} real sites)")
    if idx:
        import pandas as pd
        pd.DataFrame(idx).to_csv(os.path.join(meta_dir, "real_scenario_index.csv"), index=False)
    return out


def hi_lo(m: Mission) -> float:
    return m.max_altitude - m.min_altitude

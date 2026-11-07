"""Reproducible synthetic UAV benchmark. Every scenario is fully determined by (difficulty, index, base_seed)."""
from __future__ import annotations
import json
import math
import os
import random
from src.models.schemas import Mission, RequiredWaypoint, NoFlyZone
from src.benchmark.feasibility import assess
from src.utils import geometry as G

MAP = 100.0
DIFF = {  # constraint complexity levels (survey Objective 1): more zones/obstacles, tighter distance AND battery, narrower altitude band
    "easy":   dict(n_req=(2, 2), n_zones=(1, 2), slack=(1.5, 1.8), band=(55, 70), poly_p=0.0, energy_slack=None, obstacle_p=0.0, shuffle_p=0.0),
    "medium": dict(n_req=(3, 3), n_zones=(3, 4), slack=(1.2, 1.35), band=(35, 45), poly_p=0.3, energy_slack=(1.35, 1.6), obstacle_p=0.3, shuffle_p=0.0),
    "hard":   dict(n_req=(4, 5), n_zones=(5, 7), slack=(1.04, 1.12), band=(18, 25), poly_p=0.5, energy_slack=(1.08, 1.18), obstacle_p=0.4, shuffle_p=0.5),
    # controlled combinations of constraint families (survey Objective 2): same geometry difficulty, constraints switched on/off
    "combo":  dict(n_req=(3, 3), n_zones=(3, 3), slack=(1.08, 1.15), band=(22, 28), poly_p=0.3, energy_slack=(1.08, 1.18), obstacle_p=0.3, shuffle_p=1.0),
}
FAM_ABBR = {"altitude": "alt", "no_fly_zone": "nfz", "distance": "dist", "ordering": "ord", "energy": "en"}
COMBOS = [("altitude",), ("no_fly_zone",), ("distance",), ("ordering",), ("energy",),
          ("altitude", "no_fly_zone"), ("no_fly_zone", "distance"), ("distance", "energy"), ("altitude", "energy"), ("ordering", "distance"),
          ("altitude", "no_fly_zone", "distance"), ("no_fly_zone", "distance", "energy"),
          ("altitude", "no_fly_zone", "distance", "ordering"), ("altitude", "no_fly_zone", "distance", "ordering", "energy")]
OBSTACLE_NAMES = ["radio tower", "building", "terrain ridge", "crane"]


STYLES = ("formal", "terse", "narrative")


def _fmt(v: float) -> str:
    return f"{v:.1f}" if abs(v) < 1000 else f"{v:.0f}"


def nl_description(m: Mission, style: str = "formal", labels: dict | None = None, unit: str = "units") -> str:
    """Deterministic NL rendering of the structured mission (3 surface styles). Only ACTIVE constraint families are mentioned."""
    lab = lambda i: f"{i} ({labels[i]})" if labels and i in labels else i
    req = ", ".join(f"{lab(r.id)} at ({_fmt(r.x)}, {_fmt(r.y)})" for r in m.required_waypoints)
    sx, sy, dx, dy = _fmt(m.start[0]), _fmt(m.start[1]), _fmt(m.destination[0]), _fmt(m.destination[1])
    has_ord = m.is_active("ordering") and m.ordering
    order = "; ".join(f"visit {a} before {b}" for a, b in m.ordering) if has_ord else ""
    def _vert(z):
        if not z.is_vertically_bounded:
            return ""
        lo_t = "ground" if z.z_min is None else f"{z.z_min:g} m"
        hi_t = "unlimited height" if z.z_max is None else f"{z.z_max:g} m"
        return f" [blocks only altitudes from {lo_t} up to {hi_t}; above that it can be overflown]"
    zones = ", ".join(z.id + (f" [{z.description}]" if z.description and labels else (f" ({z.category})" if z.category == "obstacle" else "")) + _vert(z) for z in m.no_fly_zones)
    alt = m.is_active("altitude")
    dist = m.is_active("distance")
    en = m.is_active("energy") and m.max_energy is not None
    en_txt = (f"{m.max_energy:g} energy units, where energy = route length + {m.energy_k_v:g} x total climb"
              + (f" + {m.energy_k_wp:g} per waypoint" if m.energy_k_wp else "") + " (so low altitudes save battery)") if en else ""
    if style == "terse":
        parts = [f"Route START({sx},{sy}) -> DEST({dx},{dy}) via {req}."]
        if has_ord:
            parts.append(f"Order: {order}.")
        if alt:
            parts.append(f"Altitude {m.min_altitude:g}-{m.max_altitude:g} m at all waypoints.")
        if zones:
            parts.append(f"No entry to or crossing of {zones}.")
        if dist:
            parts.append(f"Total length <= {m.max_distance:g} {unit}.")
        if en:
            parts.append(f"Battery: {en_txt}.")
        return " ".join(parts)
    if style == "narrative":
        parts = [f"The drone leaves its base at ({sx}, {sy}) and has to end up at ({dx}, {dy}). On the way it must stop at {req}" + (f"; {order}." if has_ord else ".")]
        if alt:
            parts.append(f"Keep it no lower than {m.min_altitude:g} m and no higher than {m.max_altitude:g} m.")
        if zones:
            parts.append(f"The restricted areas and obstacles ({zones}) are off limits, and that includes cutting across a corner of one on the way between two stops.")
        if dist:
            parts.append(f"The whole trip may not be longer than {m.max_distance:g} {unit}.")
        if en:
            parts.append(f"The battery only allows {en_txt}.")
        return " ".join(parts)
    parts = [f"Fly from the base at ({sx}, {sy}) to the destination at ({dx}, {dy}).", f"The route must pass through the required points {req}" + (f"; {order}." if has_ord else ".")]
    if alt:
        parts.append(f"Maintain an altitude between {m.min_altitude:g} and {m.max_altitude:g} meters at every waypoint.")
    if zones:
        parts.append(f"Avoid all restricted zones and obstacles ({zones}), including the straight paths between waypoints.")
    if dist:
        parts.append(f"Keep the total horizontal route length at or below {m.max_distance:g} {unit}.")
    if en:
        parts.append(f"Stay within the battery budget of {en_txt}.")
    return " ".join(parts)


def _poly(rng: random.Random, c, R):
    n = rng.randint(4, 6)
    angs = [2 * math.pi * k / n + rng.uniform(-0.25, 0.25) for k in range(n)]
    return [[round(c[0] + R * rng.uniform(0.8, 1.0) * math.cos(a), 2), round(c[1] + R * rng.uniform(0.8, 1.0) * math.sin(a), 2)] for a in angs]


def _witness_energy(m: Mission, plan) -> float:
    from src.verifier.verifier import resources
    return resources(plan, m)["energy"]


def _build(rng: random.Random, sid: str, diff: str, active: tuple | None = None) -> tuple[Mission, dict] | None:
    p = DIFF[diff]
    if active is None:
        active = ("altitude", "no_fly_zone", "distance", "ordering") + (("energy",) if p["energy_slack"] else ())
    n_req = rng.randint(*p["n_req"])
    start = [round(rng.uniform(5, 15), 1), round(rng.uniform(10, 90), 1)]
    dest = [round(rng.uniform(85, 95), 1), round(rng.uniform(10, 90), 1)]
    xs = [rng.uniform(25, 75) for _ in range(n_req)]
    if not (rng.random() < p["shuffle_p"]):
        xs.sort()          # shuffled x-order makes "visit A before B" conflict with the shortest tour (ordering vs distance interaction)
    reqs = [RequiredWaypoint(chr(65 + i), round(xs[i], 1), round(rng.uniform(10, 90), 1)) for i in range(n_req)]
    chain = [tuple(start)] + [r.xy for r in reqs] + [tuple(dest)]
    nz = rng.randint(*p["n_zones"]) if "no_fly_zone" in active else 0
    zones = []
    for k in range(nz):
        for _ in range(200):
            i = rng.randrange(len(chain) - 1)
            u = rng.uniform(0.2, 0.8)
            c = (chain[i][0] + u * (chain[i + 1][0] - chain[i][0]) + rng.uniform(-4, 4), chain[i][1] + u * (chain[i + 1][1] - chain[i][1]) + rng.uniform(-4, 4))
            c = (round(c[0], 1), round(c[1], 1))
            R = round(rng.uniform(5, 10 if diff != "hard" else 9), 1)
            obst = rng.random() < p["obstacle_p"]
            desc = rng.choice(OBSTACLE_NAMES) if obst else f"restricted area {k+1}"
            cat = "obstacle" if obst else "nfz"
            z = (NoFlyZone(f"Z{k+1}", "polygon", vertices=_poly(rng, c, R), description=desc, category=cat)
                 if rng.random() < p["poly_p"] else NoFlyZone(f"Z{k+1}", "circle", center=list(c), radius=R, description=desc, category=cat))
            z.validate()
            if all(G.dist(z.centroid(), q) > z.bounding_radius() + 3.0 for q in chain) and \
               all(G.dist(z.centroid(), o.centroid()) > z.bounding_radius() + o.bounding_radius() + 1.0 for o in zones):
                zones.append(z)
                break
    lo = round(rng.uniform(30, 50), 0)
    band = rng.uniform(*p["band"])
    ordering = [[reqs[i].id, reqs[i + 1].id] for i in range(n_req - 1)] if (diff != "hard" or rng.random() < 0.5) else [[reqs[i].id, reqs[i + 1].id] for i in range(0, n_req - 1, 2)]
    if "ordering" not in active:
        ordering = []
    alt_lo, alt_hi = (lo, round(lo + band, 0)) if "altitude" in active else (0.0, 1000.0)
    m = Mission(sid, "", start, dest, reqs, alt_lo, alt_hi, 1e9, zones, ordering, energy_k_v=1.0, active=list(active)).validate()
    f = assess(m)
    if f.status != "FEASIBLE":
        return None
    w_energy = _witness_energy(m, f.witness_plan)
    slack = rng.uniform(*p["slack"])
    e_slack = rng.uniform(*p["energy_slack"]) if p["energy_slack"] else None
    if "distance" in active:
        m.max_distance = round(f.witness_distance * slack, 1)
    if "energy" in active:
        m.max_energy = round(w_energy * (e_slack or 1.3), 1)
    m.description = nl_description(m, STYLES[int(sid.split('_')[-1]) % 3])
    f = assess(m)
    if f.status != "FEASIBLE":
        return None
    area = sum(z.area() for z in zones) / MAP ** 2
    meta = {"n_zones": len(zones), "n_obstacles": sum(z.category == "obstacle" for z in zones), "zone_density": round(area, 4),
            "distance_slack": round(slack, 3) if "distance" in active else None, "energy_slack": round(e_slack, 3) if ("energy" in active and e_slack) else None,
            "altitude_band_width": (alt_hi - alt_lo) if "altitude" in active else None, "n_required": n_req, "feasibility": f.status,
            "witness_distance": round(f.witness_distance, 3), "witness_energy": round(w_energy, 3), "lower_bound": round(f.lower_bound, 3),
            "infeasible_kind": None, "source": "synthetic", "nl_style": STYLES[int(sid.split('_')[-1]) % 3],
            "n_polygon_zones": sum(z.kind == "polygon" for z in zones), "active": list(active), "n_active": len(active),
            "ordering_conflicts_with_route": bool(ordering) and _order_vs_x(reqs)}
    return m, meta


def _order_vs_x(reqs) -> bool:
    xs = [r.x for r in reqs]
    return xs != sorted(xs)


def _build_corridor(rng: random.Random, sid: str):
    """A wall of two rectangles with a narrow gap plus a concave L-shaped zone: forces detours through a gap."""
    gy = rng.uniform(25, 75)
    gap = rng.uniform(9, 14)
    x0, x1 = 47.0, 53.0
    z1 = NoFlyZone("Z1", "polygon", vertices=[[x0, 0], [x1, 0], [x1, round(gy - gap / 2, 1)], [x0, round(gy - gap / 2, 1)]], description="wall segment south")
    z2 = NoFlyZone("Z2", "polygon", vertices=[[x0, round(gy + gap / 2, 1)], [x1, round(gy + gap / 2, 1)], [x1, 100], [x0, 100]], description="wall segment north")
    cx, cy = rng.uniform(68, 80), rng.uniform(20, 80)
    L = [[cx, cy], [cx + 12, cy], [cx + 12, cy + 4], [cx + 4, cy + 4], [cx + 4, cy + 14], [cx, cy + 14]]
    z3 = NoFlyZone("Z3", "polygon", vertices=[[round(a, 1), round(b, 1)] for a, b in L], description="concave L-shaped zone")
    start = [round(rng.uniform(5, 15), 1), round(rng.uniform(15, 85), 1)]
    dest = [round(rng.uniform(88, 95), 1), round(rng.uniform(15, 85), 1)]
    reqs = [RequiredWaypoint("A", round(rng.uniform(22, 38), 1), round(rng.uniform(15, 85), 1)),
            RequiredWaypoint("B", round(rng.uniform(58, 66), 1), round(rng.uniform(15, 85), 1))]
    lo = round(rng.uniform(30, 50), 0)
    m = Mission(sid, "", start, dest, reqs, lo, lo + 25, 1e6, [z1, z2, z3], [["A", "B"]], active=["altitude", "no_fly_zone", "distance", "ordering"]).validate()
    f = assess(m)
    if f.status != "FEASIBLE":
        return None
    if f.witness_distance < f.lower_bound * 1.03:  # wall must actually force a detour
        return None
    slack = rng.uniform(1.08, 1.2)
    m.max_distance = round(f.witness_distance * slack, 1)
    m.description = nl_description(m, STYLES[int(sid.split('_')[-1]) % 3])
    f = assess(m)
    if f.status != "FEASIBLE":
        return None
    meta = {"n_zones": 3, "zone_density": round(sum(z.area() for z in m.no_fly_zones) / MAP ** 2, 4), "distance_slack": round(slack, 3),
            "altitude_band_width": 25.0, "n_required": 2, "feasibility": "FEASIBLE", "witness_distance": round(f.witness_distance, 3),
            "lower_bound": round(f.lower_bound, 3), "infeasible_kind": None, "source": "synthetic", "nl_style": "auto",
            "n_polygon_zones": 3, "gap_width": round(gap, 2), "active": list(m.active), "n_active": len(m.active), "n_obstacles": 0, "witness_energy": None, "energy_slack": None}
    return m, meta


def _build_vertical(rng: random.Random, sid: str):
    """3D-aware family (reviewer: 'altitude-dependent obstacles'): an easy mission whose 1-2 obstacles have a finite HEIGHT (z_max inside the allowed band),
    so they can be overflown. The distance budget is set from the witness that is allowed to fly over them -> the model must use the third dimension
    (or a long detour) to stay within budget. Remaining zones stay full-height."""
    r = _build(rng, sid, "easy")
    if not r:
        return None
    m, meta = r
    cand = [z for z in m.no_fly_zones]
    if not cand:
        return None
    k = min(len(cand), rng.randint(1, 2))
    chosen = rng.sample(cand, k)
    for z in chosen:
        z.z_max = round(m.min_altitude + rng.uniform(0.3, 0.7) * (m.max_altitude - m.min_altitude), 0)
        z.category = "obstacle"
        z.description = rng.choice(OBSTACLE_NAMES)
        z.validate()
    m.max_distance = 1e9
    f = assess(m)
    if f.status != "FEASIBLE":
        return None
    slack = rng.uniform(1.1, 1.25)
    m.max_distance = round(f.witness_distance * slack, 1)
    m.description = nl_description(m, STYLES[int(sid.split("_")[-1]) % 3])
    f = assess(m)
    if f.status != "FEASIBLE":
        return None
    meta = dict(meta, feasibility="FEASIBLE", witness_distance=round(f.witness_distance, 3), distance_slack=round(slack, 3), n_vertical_zones=k,
                witness_altitude=f.witness_plan.waypoints[0].altitude if f.witness_plan.waypoints else f.witness_plan.destination.altitude)
    return m, meta


def _infeasible(rng: random.Random, sid: str, kind: str):
    for _ in range(50):
        r = _build(rng, sid, "medium")
        if r:
            break
    m, meta = r
    if kind == "required_in_zone":
        z = m.no_fly_zones[0]
        c = z.centroid()
        m.required_waypoints[0] = RequiredWaypoint(m.required_waypoints[0].id, round(c[0], 1), round(c[1], 1))
    elif kind == "distance_below_bound":
        pts = [m.start_xy] + [r.xy for r in m.required_waypoints] + [m.dest_xy]
        m.max_distance = round(0.85 * sum(G.dist(a, b) for a, b in zip(pts, pts[1:])), 1)
    elif kind == "energy_below_bound":
        pts = [m.start_xy] + [r.xy for r in m.required_waypoints] + [m.dest_xy]
        m.max_energy = round(0.8 * sum(G.dist(a, b) for a, b in zip(pts, pts[1:])), 1)
        if "energy" not in m.active:
            m.active.append("energy")
    elif kind == "cyclic_order":
        m.ordering = m.ordering + [[m.ordering[0][1], m.ordering[0][0]]]
    m.validate()
    m.description = nl_description(m, STYLES[int(sid.split("_")[-1]) % 3])
    f = assess(m)
    assert f.status == "INFEASIBLE", (sid, kind, f)
    meta = dict(meta, feasibility="INFEASIBLE", infeasible_kind=kind, infeasible_reason=f.reason)
    return m, meta


def generate_benchmark(out_dir: str, meta_dir: str, per_difficulty: int = 20, n_infeasible: int = 6, base_seed: int = 2024, n_corridor: int = 10, n_per_combo: int = 3, maps_dir: str | None = None, n_vertical: int = 0) -> list[dict]:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    scenarios = []
    for d_i, diff in enumerate(["easy", "medium", "hard"]):
        for i in range(per_difficulty):
            sid = f"{diff}_{i:03d}"
            for attempt in range(100):
                rng = random.Random(f"{base_seed}|{sid}|{attempt}")
                r = _build(rng, sid, diff)
                if r:
                    break
            else:
                raise RuntimeError(f"could not generate {sid}")
            scenarios.append((sid, diff, r, base_seed, attempt))
    for i in range(n_corridor):
        sid = f"corridor_{i:03d}"
        for attempt in range(100):
            r = _build_corridor(random.Random(f"{base_seed}|{sid}|{attempt}"), sid)
            if r:
                break
        else:
            raise RuntimeError(f"could not generate {sid}")
        scenarios.append((sid, "corridor", r, base_seed, attempt))
    for i in range(n_vertical):   # optional 3D family; n_vertical = 0 keeps the published 118-scenario benchmark byte-identical
        sid = f"vertical_{i:03d}"
        for attempt in range(200):
            r = _build_vertical(random.Random(f"{base_seed}|{sid}|{attempt}"), sid)
            if r:
                break
        else:
            raise RuntimeError(f"could not generate {sid}")
        scenarios.append((sid, "vertical", r, base_seed, attempt))
    for ci, combo in enumerate(COMBOS if n_per_combo else []):
        for i in range(n_per_combo):
            sid = f"combo_{'-'.join(FAM_ABBR[f] for f in combo)}_{i:03d}"
            for attempt in range(100):
                r = _build(random.Random(f"{base_seed}|{sid}|{attempt}"), sid, "combo", combo)
                if r:
                    break
            else:
                raise RuntimeError(f"could not generate {sid}")
            scenarios.append((sid, "combo", r, base_seed, attempt))
    kinds = ["required_in_zone", "distance_below_bound", "cyclic_order", "energy_below_bound"]
    for i in range(n_infeasible):
        sid = f"infeasible_{i:03d}"
        rng = random.Random(f"{base_seed}|{sid}")
        scenarios.append((sid, "infeasible", _infeasible(rng, sid, kinds[i % 4]), base_seed, 0))
    out = []
    rows = []
    for sid, diff, (m, meta), seed, attempt in scenarios:
        rec = {"scenario_id": sid, "difficulty": diff, "base_seed": seed, "attempt": attempt, "mission": m.to_dict(), "metadata": meta}
        with open(os.path.join(out_dir, sid + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        out.append(rec)
        rows.append({"scenario_id": sid, "difficulty": diff, **meta})
    if maps_dir:
        os.makedirs(maps_dir, exist_ok=True)
        from src.simulator.visualize import plot_plan
        from src.verifier.verifier import verify
        for rec in out:
            m = Mission.from_dict(rec["mission"])
            f = assess(m)
            plot_plan(m, f.witness_plan, verify(m, f.witness_plan) if f.witness_plan else None, title=f"{rec['scenario_id']} ({rec['difficulty']}) - reference route",
                      path=os.path.join(maps_dir, rec["scenario_id"] + ".png"))
    import pandas as pd
    pd.DataFrame(rows).to_csv(os.path.join(meta_dir, "scenario_index.csv"), index=False)
    return out


def load_scenarios(path: str) -> list[dict]:
    out = []
    if not os.path.isdir(path):
        return out
    for fn in sorted(os.listdir(path)):
        if fn.endswith(".json"):
            with open(os.path.join(path, fn)) as f:
                out.append(json.load(f))
    return out

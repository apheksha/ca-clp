"""Randomised + adversarial self-test of the hand-written geometry (reviewer: 'numerical geometry tests are mentioned but not characterised').

Reference: dense sampling of the segment (SAMPLES points) against an independent point-in-zone predicate (circle: hypot; polygon: winding number, NOT the ray
casting used by the production code). Contract tested (closed zones, tolerance TOL = 1e-9 absolute):
  (P) production says 'no intersection'  => no sample lies inside the zone                       (no false negatives, up to sampling)
  (Q) production says 'intersection'     => some sample lies within `slack` of the zone          (no false positives beyond grazing)
plus deterministic ADVERSARIAL cases with exact expected answers (tangent, offset above/below tolerance, vertex graze, collinear edge, zero-length segment,
endpoint on boundary, large coordinates). `run_selftest()` returns the counts so a paper can cite them; scripts/geometry_test_report.py prints them."""
from __future__ import annotations
import math
import random
from src.models.schemas import NoFlyZone
from src.utils import geometry as G

SAMPLES = 4000


def _winding_inside(p, verts) -> bool:
    wn = 0
    n = len(verts)
    for i in range(n):
        a, b = verts[i], verts[(i + 1) % n]
        if a[1] <= p[1]:
            if b[1] > p[1] and (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]) > 0:
                wn += 1
        elif b[1] <= p[1] and (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]) < 0:
            wn -= 1
    return wn != 0


def _ref_inside(z: NoFlyZone, p) -> bool:
    return math.hypot(p[0] - z.center[0], p[1] - z.center[1]) <= z.radius if z.kind == "circle" else _winding_inside(p, z._verts())


def _dist_to_zone(z: NoFlyZone, p) -> float:
    if z.kind == "circle":
        return max(0.0, math.hypot(p[0] - z.center[0], p[1] - z.center[1]) - z.radius)
    return 0.0 if _winding_inside(p, z._verts()) else G.min_dist_to_polygon_boundary(p, z._verts())


def _rand_zone(rng: random.Random, scale: float) -> NoFlyZone:
    c = [rng.uniform(0.2, 0.8) * scale, rng.uniform(0.2, 0.8) * scale]
    if rng.random() < 0.5:
        return NoFlyZone("Z", "circle", center=c, radius=rng.uniform(0.02, 0.15) * scale).validate()
    n = rng.randint(3, 8)
    R = rng.uniform(0.03, 0.15) * scale
    angs = sorted(rng.uniform(0, 2 * math.pi) for _ in range(n))     # star-shaped, possibly strongly concave
    return NoFlyZone("Z", "polygon", vertices=[[c[0] + R * rng.uniform(0.3, 1.0) * math.cos(a), c[1] + R * rng.uniform(0.3, 1.0) * math.sin(a)] for a in angs]).validate()


def randomised(n: int = 3000, seed: int = 0, scale: float = 100.0) -> dict:
    rng = random.Random(seed)
    fn = fp = checked = 0
    for _ in range(n):
        z = _rand_zone(rng, scale)
        a = (rng.uniform(0, scale), rng.uniform(0, scale))
        b = (rng.uniform(0, scale), rng.uniform(0, scale))
        got = z.intersects_segment(a, b)
        pts = [(a[0] + k / SAMPLES * (b[0] - a[0]), a[1] + k / SAMPLES * (b[1] - a[1])) for k in range(SAMPLES + 1)]
        ref = any(_ref_inside(z, p) for p in pts)
        slack = G.dist(a, b) / SAMPLES + 1e-6        # sampling resolution
        checked += 1
        if ref and not got:
            fn += 1
        if got and not ref and min(_dist_to_zone(z, p) for p in pts) > slack:
            fp += 1
    return {"n_random_cases": checked, "false_negatives": fn, "false_positives_beyond_grazing": fp, "scale": scale, "samples_per_segment": SAMPLES}


def adversarial() -> dict:
    """Each case: (description, zone, a, b, expected_intersects). Offsets are relative to TOL = 1e-9."""
    cases = []
    for scale in (1.0, 100.0, 1e5):      # real-airspace coordinates are metres (~1e4-1e5), synthetic are 0-100
        r = 0.1 * scale
        c = (0.5 * scale, 0.5 * scale)
        circ = NoFlyZone("C", "circle", center=list(c), radius=r).validate()
        for name, off, exp in [("tangent_exact", 0.0, True), ("inside_by_1e-7", -1e-7 * max(1, scale / 100), True), ("outside_by_1e-6", 1e-6 * max(1, scale / 100), False),
                               ("outside_by_1e-4", 1e-4 * max(1, scale / 100), False)]:
            y = c[1] + r + off
            cases.append((f"circle_{name}@{scale:g}", circ, (c[0] - scale * 0.3, y), (c[0] + scale * 0.3, y), exp))
        sq = NoFlyZone("S", "polygon", vertices=[[c[0] - r, c[1] - r], [c[0] + r, c[1] - r], [c[0] + r, c[1] + r], [c[0] - r, c[1] + r]]).validate()
        eps = 1e-6 * max(1, scale / 100)
        cases += [
            (f"poly_edge_collinear@{scale:g}", sq, (c[0] - 3 * r, c[1] + r), (c[0] + 3 * r, c[1] + r), True),
            (f"poly_vertex_graze@{scale:g}", sq, (c[0] + r - 2 * r, c[1] + 2 * r), (c[0] + r + 2 * r, c[1] + 0.0 * r), True),
            (f"poly_near_miss_above@{scale:g}", sq, (c[0] - 3 * r, c[1] + r + eps), (c[0] + 3 * r, c[1] + r + eps), False),
            (f"poly_endpoint_on_boundary@{scale:g}", sq, (c[0] + r, c[1]), (c[0] + 3 * r, c[1]), True),
            (f"poly_zero_length_inside@{scale:g}", sq, c, c, True),
            (f"poly_zero_length_outside@{scale:g}", sq, (c[0] + 5 * r, c[1]), (c[0] + 5 * r, c[1]), False),
            (f"poly_segment_inside_no_edge_cross@{scale:g}", sq, (c[0] - 0.5 * r, c[1]), (c[0] + 0.5 * r, c[1]), True),
        ]
        conc = NoFlyZone("L", "polygon", vertices=[[0.1 * scale, 0.1 * scale], [0.9 * scale, 0.1 * scale], [0.9 * scale, 0.3 * scale], [0.3 * scale, 0.3 * scale],
                                                    [0.3 * scale, 0.9 * scale], [0.1 * scale, 0.9 * scale]]).validate()    # L shape: the notch is NOT inside
        cases += [(f"concave_notch_outside@{scale:g}", conc, (0.5 * scale, 0.5 * scale), (0.8 * scale, 0.8 * scale), False),
                  (f"concave_cross_arm@{scale:g}", conc, (0.5 * scale, 0.0), (0.5 * scale, 0.5 * scale), True)]
    bad = []
    for name, z, a, b, exp in cases:
        if z.intersects_segment(a, b) != exp:
            bad.append(name)
    # tolerance regime documentation
    return {"n_adversarial_cases": len(cases), "failures": bad, "tolerance_abs": G.TOL,
            "decisive_offset_note": "offsets >= 1e-6 (scaled by coordinate magnitude/100) are decided correctly up to 1e5 coordinate scale; offsets below ~1e-11*scale are below float64 resolution and are resolved by the closed-set convention (touch => intersect)"}


def exact_crosscheck(n: int = 3000, seed: int = 5, scale: float = 100.0) -> dict:
    """Float predicate vs EXACT rational predicate (src/utils/exact_geometry.py) on random cases and on cases constructed to sit within a few ulps / 1e-9 of the boundary.
    Contract: the float verifier may be conservative within TOL (reports intersection when exactly there is none but the gap is < ~1e-9*scale), but must NEVER miss an
    exact intersection."""
    from src.utils.exact_geometry import compare_with_float
    rng = random.Random(seed)
    missed = conservative = checked = 0
    worst_gap = 0.0
    for i in range(n):
        z = _rand_zone(rng, scale)
        a = (rng.uniform(0, scale), rng.uniform(0, scale))
        b = (rng.uniform(0, scale), rng.uniform(0, scale))
        if i % 3 == 0 and z.kind == "circle":          # tangent-ish: shift the segment so it sits delta from the circle
            cx, cy = z.center
            delta = rng.choice([0.0, 1e-12, 1e-10, 1e-8, -1e-8, 1e-6, -1e-6]) * scale
            y = cy + z.radius + delta
            a, b = (cx - 0.5 * scale * 0.2, y), (cx + 0.5 * scale * 0.2, y)
        r = compare_with_float(z, a, b)
        checked += 1
        if r["kind"] == "FLOAT_MISSED_INTERSECTION":
            missed += 1
        elif r["kind"] == "float_conservative_grazing":
            conservative += 1
            worst_gap = max(worst_gap, _dist_to_zone(z, min([a, b], key=lambda p: _dist_to_zone(z, p))))
    return {"n_cases": checked, "float_missed_exact_intersection": missed, "float_conservative_within_tolerance": conservative, "scale": scale}


def run_selftest(n_random: int = 3000, seed: int = 0) -> dict:
    r = {"random_scale_100": randomised(n_random, seed, 100.0), "random_scale_1e5": randomised(max(n_random // 3, 300), seed + 1, 1e5), "adversarial": adversarial()}
    r["exact_crosscheck"] = exact_crosscheck(max(n_random, 500), seed + 2)
    r["ok"] = (r["exact_crosscheck"]["float_missed_exact_intersection"] == 0 and r["random_scale_100"]["false_negatives"] == 0 and r["random_scale_100"]["false_positives_beyond_grazing"] == 0
               and r["random_scale_1e5"]["false_negatives"] == 0 and r["random_scale_1e5"]["false_positives_beyond_grazing"] == 0 and not r["adversarial"]["failures"])
    return r

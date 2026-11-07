"""EXACT (rational-arithmetic) geometric predicates, used as ground truth for the floating-point verifier and as an optional strict mode.

Floating-point inputs are converted to `fractions.Fraction` (exactly representable), so every predicate below is decided WITHOUT rounding error and WITHOUT a
tolerance: 'intersects' means the closed segment and the closed zone share at least one point in exact real arithmetic. The production verifier uses a 1e-9
tolerance on top of float64 (conservative: a grazing contact within 1e-9 counts as intersecting); `compare_with_float` quantifies where the two disagree.
Circle predicates are exact for squared distances (no square roots): dist(c, segment)^2 <= r^2."""
from __future__ import annotations
from fractions import Fraction as F


def _f(p):
    return (F(p[0]), F(p[1]))


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _on_segment(p, a, b) -> bool:
    return _cross(a, b, p) == 0 and min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def segments_intersect(a, b, c, d) -> bool:
    a, b, c, d = _f(a), _f(b), _f(c), _f(d)
    d1, d2, d3, d4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return _on_segment(c, a, b) or _on_segment(d, a, b) or _on_segment(a, c, d) or _on_segment(b, c, d)


def point_in_closed_polygon(p, verts) -> bool:
    """Exact: boundary counts as inside."""
    p = _f(p)
    v = [_f(q) for q in verts]
    n = len(v)
    for i in range(n):
        if _on_segment(p, v[i], v[(i + 1) % n]):
            return True
    inside = False
    for i in range(n):
        a, b = v[i], v[(i + 1) % n]
        if (a[1] > p[1]) != (b[1] > p[1]):
            x = a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if p[0] < x:
                inside = not inside
    return inside


def segment_intersects_polygon(a, b, verts) -> bool:
    if point_in_closed_polygon(a, verts) or point_in_closed_polygon(b, verts):
        return True
    n = len(verts)
    return any(segments_intersect(a, b, verts[i], verts[(i + 1) % n]) for i in range(n))


def segment_intersects_circle(a, b, c, r) -> bool:
    a, b, c, r = _f(a), _f(b), _f(c), F(r)
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    if L2 == 0:
        t = F(0)
    else:
        t = ((c[0] - a[0]) * dx + (c[1] - a[1]) * dy) / L2
        t = min(max(t, F(0)), F(1))
    px, py = a[0] + t * dx, a[1] + t * dy
    return (px - c[0]) ** 2 + (py - c[1]) ** 2 <= r * r


def zone_intersects_segment_exact(zone, a, b) -> bool:
    if zone.kind == "circle":
        return segment_intersects_circle(a, b, zone.center, zone.radius)
    return segment_intersects_polygon(a, b, zone.vertices)


def compare_with_float(zone, a, b) -> dict:
    """Disagreement classification between the production float predicate and the exact one."""
    fl, ex = zone.intersects_segment(a, b), zone_intersects_segment_exact(zone, a, b)
    return {"float": fl, "exact": ex, "agree": fl == ex,
            "kind": "agree" if fl == ex else ("float_conservative_grazing" if fl and not ex else "FLOAT_MISSED_INTERSECTION")}

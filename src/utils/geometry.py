"""Small, dependency-free 2D computational geometry with an explicit boundary policy.

BOUNDARY POLICY (documented in docs/VERIFIER_RULES.md): no-fly zones are CLOSED sets.
A point on the boundary is inside; a segment that merely touches the boundary
(tangent to a circle, grazing a polygon vertex/edge) intersects the zone.
All comparisons use an absolute tolerance TOL.
"""
from __future__ import annotations
import math

TOL = 1e-9
Point = tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def closest_on_segment(p: Point, a: Point, b: Point) -> tuple[Point, float, float]:
    """Return (closest point, parameter t in [0,1], distance) from p to segment ab."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    l2 = dx * dx + dy * dy
    if l2 <= 0.0:
        return (a[0], a[1]), 0.0, dist(p, a)
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2
    t = max(0.0, min(1.0, t))
    c = (a[0] + t * dx, a[1] + t * dy)
    return c, t, dist(p, c)


def on_segment(p: Point, a: Point, b: Point, tol: float = TOL) -> bool:
    return closest_on_segment(p, a, b)[2] <= tol


def _cross(o: Point, a: Point, b: Point) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def segments_intersect(a: Point, b: Point, c: Point, d: Point, tol: float = TOL) -> bool:
    """Closed segments ab and cd share at least one point (touching counts)."""
    if on_segment(a, c, d, tol) or on_segment(b, c, d, tol) or on_segment(c, a, b, tol) or on_segment(d, a, b, tol):
        return True
    d1, d2, d3, d4 = _cross(a, b, c), _cross(a, b, d), _cross(c, d, a), _cross(c, d, b)
    return d1 * d2 < 0 and d3 * d4 < 0


def segment_intersection_point(a: Point, b: Point, c: Point, d: Point):
    """Return (t along ab, point) for an intersection of non-parallel segments, else None."""
    rx, ry = b[0] - a[0], b[1] - a[1]
    sx, sy = d[0] - c[0], d[1] - c[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-14:
        return None
    t = ((c[0] - a[0]) * sy - (c[1] - a[1]) * sx) / den
    u = ((c[0] - a[0]) * ry - (c[1] - a[1]) * rx) / den
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return t, (a[0] + t * rx, a[1] + t * ry)
    return None


def point_in_polygon(p: Point, verts: list[Point], tol: float = TOL) -> str:
    """'inside' | 'boundary' | 'outside' (ray casting, boundary tested first)."""
    n = len(verts)
    for i in range(n):
        if on_segment(p, verts[i], verts[(i + 1) % n], tol):
            return "boundary"
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        if (yi > p[1]) != (yj > p[1]) and p[0] < (xj - xi) * (p[1] - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return "inside" if inside else "outside"


def polygon_centroid(verts: list[Point]) -> Point:
    return (sum(v[0] for v in verts) / len(verts), sum(v[1] for v in verts) / len(verts))


def min_dist_to_polygon_boundary(p: Point, verts: list[Point]) -> float:
    n = len(verts)
    return min(closest_on_segment(p, verts[i], verts[(i + 1) % n])[2] for i in range(n))


def circle_chord(a: Point, b: Point, c: Point, r: float):
    """Entry/exit points where segment ab crosses circle (c,r); None if no chord."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    fx, fy = a[0] - c[0], a[1] - c[1]
    A = dx * dx + dy * dy
    if A <= 0:
        return None
    B = 2 * (fx * dx + fy * dy)
    C = fx * fx + fy * fy - r * r
    disc = B * B - 4 * A * C
    if disc < 0:
        return None
    s = math.sqrt(disc)
    t1, t2 = max(0.0, (-B - s) / (2 * A)), min(1.0, (-B + s) / (2 * A))
    if t1 > t2:
        return None
    return (a[0] + t1 * dx, a[1] + t1 * dy), (a[0] + t2 * dx, a[1] + t2 * dy)


def circle_chord_t(a: Point, b: Point, c: Point, r: float, tol: float = TOL):
    """Parameter interval [t1, t2] (clamped to [0,1]) of closed segment ab that lies inside the closed disc (c, r); None if disjoint.
    A tangent / grazing contact (within tol) returns a degenerate interval (t, t)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    A = dx * dx + dy * dy
    if A <= 0.0:
        return (0.0, 0.0) if dist(a, c) <= r + tol else None
    cp, tc, dmin = closest_on_segment(c, a, b)
    if dmin > r + tol:
        return None
    fx, fy = a[0] - c[0], a[1] - c[1]
    B = 2 * (fx * dx + fy * dy)
    C = fx * fx + fy * fy - r * r
    disc = B * B - 4 * A * C
    if disc <= 0:
        return (tc, tc)
    s = math.sqrt(disc)
    t1, t2 = max(0.0, (-B - s) / (2 * A)), min(1.0, (-B + s) / (2 * A))
    if t1 > t2:
        return (tc, tc)
    return (t1, t2)


def polygon_inside_intervals(a: Point, b: Point, verts: list[Point], tol: float = TOL) -> list[tuple[float, float]]:
    """Parameter intervals of closed segment ab lying in the CLOSED polygon (boundary counts as inside). Degenerate (t, t) = touching."""
    n = len(verts)
    ts = {0.0, 1.0}
    for i in range(n):
        c, d = verts[i], verts[(i + 1) % n]
        r = segment_intersection_point(a, b, c, d)
        if r:
            ts.add(min(1.0, max(0.0, r[0])))
        L2 = (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
        if L2 > 0:   # collinear overlap: project the edge end points
            for q in (c, d):
                if on_segment(q, a, b, tol):
                    ts.add(min(1.0, max(0.0, ((q[0] - a[0]) * (b[0] - a[0]) + (q[1] - a[1]) * (b[1] - a[1])) / L2)))
    ts_sorted = sorted(ts)
    pt = lambda t: (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
    out: list[tuple[float, float]] = []
    for t0, t1 in zip(ts_sorted, ts_sorted[1:]):
        if t1 - t0 > 1e-12 and point_in_polygon(pt((t0 + t1) / 2), verts, tol) != "outside":
            out.append((t0, t1))
    for t in ts_sorted:   # isolated touching points
        if point_in_polygon(pt(t), verts, tol) != "outside" and not any(t0 - 1e-12 <= t <= t1 + 1e-12 for t0, t1 in out):
            out.append((t, t))
    out.sort()
    return out

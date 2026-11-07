"""Local tangent-plane (equirectangular) projection used for all geodata: x east, y north, in metres, around (lat0, lon0). Error << 1 % over tens of km; use a proper
UTM/geodesic library for large areas or surveying-grade work."""
from __future__ import annotations
import math

R = 6371008.8


class LocalFrame:
    def __init__(self, lat0: float, lon0: float):
        self.lat0, self.lon0 = lat0, lon0
        self._kx = math.radians(1.0) * R * math.cos(math.radians(lat0))
        self._ky = math.radians(1.0) * R

    def to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        return ((lon - self.lon0) * self._kx, (lat - self.lat0) * self._ky)

    def to_latlon(self, x: float, y: float) -> tuple[float, float]:
        return (self.lat0 + y / self._ky, self.lon0 + x / self._kx)


def douglas_peucker(pts: list, tol: float) -> list:
    """Simplify an OPEN polyline; keeps the end points. For a closed ring pass the ring without the repeated last point."""
    if len(pts) < 3 or tol <= 0:
        return list(pts)
    a, b = pts[0], pts[-1]
    dmax, idx = 0.0, 0
    for i in range(1, len(pts) - 1):
        d = _pt_seg(pts[i], a, b)
        if d > dmax:
            dmax, idx = d, i
    if dmax > tol:
        return douglas_peucker(pts[:idx + 1], tol)[:-1] + douglas_peucker(pts[idx:], tol)
    return [a, b]


def _pt_seg(p, a, b) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L2))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def simplify_ring(ring: list, tol: float) -> list:
    """Simplify a closed ring (no repeated closing vertex) while keeping at least 3 vertices."""
    if len(ring) <= 4 or tol <= 0:
        return list(ring)
    # split at the two most distant vertices so both halves are open polylines
    i0 = 0
    j0 = max(range(len(ring)), key=lambda j: math.dist(ring[0], ring[j]))
    h1 = douglas_peucker(ring[i0:j0 + 1], tol)
    h2 = douglas_peucker(ring[j0:] + [ring[0]], tol)
    out = h1[:-1] + h2[:-1]
    return out if len(out) >= 3 else list(ring)

"""Grid A* baseline (survey Objective 2): a conventional, non-LLM planner used as a feasibility reference.
8-connected grid, obstacle cells inflated by 0.75 cell, path smoothed by line-of-sight against the TRUE zones. It is deliberately a simple
baseline: it can fail where the visibility-graph oracle succeeds (narrow gaps vs. grid resolution) and returns shorter-than-grid paths after smoothing."""
from __future__ import annotations
import heapq
import math
from src.models.schemas import Mission, FlightPlan, Waypoint
from src.benchmark.feasibility import visiting_order, extent
from src.utils import geometry as G

SQ2 = math.sqrt(2.0)


def _blocked_grid(m: Mission, x0, y0, nx, ny, cell, pad):
    blocked = [[False] * ny for _ in range(nx)]
    for z in m.no_fly_zones:
        c, R = z.centroid(), z.bounding_radius() + pad + cell
        i0, i1 = max(0, int((c[0] - R - x0) / cell)), min(nx - 1, int((c[0] + R - x0) / cell) + 1)
        j0, j1 = max(0, int((c[1] - R - y0) / cell)), min(ny - 1, int((c[1] + R - y0) / cell) + 1)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                p = (x0 + (i + 0.5) * cell, y0 + (j + 0.5) * cell)
                if z.kind == "circle":
                    hit = G.dist(p, c) <= z.radius + pad
                else:
                    hit = z.contains(p) or G.min_dist_to_polygon_boundary(p, z._verts()) <= pad
                if hit:
                    blocked[i][j] = True
    return blocked


def _free(m, a, b):
    return not any(z.intersects_segment(a, b) for z in m.no_fly_zones)


def _search(blocked, nx, ny, s, t):
    openq = [(0.0, 0.0, s)]
    best = {s: 0.0}
    prev = {}
    while openq:
        f, g, u = heapq.heappop(openq)
        if u == t:
            path = [u]
            while u in prev:
                u = prev[u]
                path.append(u)
            return path[::-1]
        if g > best.get(u, 1e18) + 1e-12:
            continue
        ux, uy = u
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                v = (ux + dx, uy + dy)
                if not (0 <= v[0] < nx and 0 <= v[1] < ny) or (blocked[v[0]][v[1]] and v != t):
                    continue
                if dx and dy and (blocked[ux + dx][uy] or blocked[ux][uy + dy]):
                    continue   # no corner cutting
                ng = g + (SQ2 if dx and dy else 1.0)
                if ng < best.get(v, 1e18):
                    best[v] = ng
                    prev[v] = u
                    h = math.hypot(t[0] - v[0], t[1] - v[1])
                    heapq.heappush(openq, (ng + h, ng, v))
    return None


def astar_plan(m: Mission, resolution: int = 200) -> FlightPlan | None:
    order = visiting_order(m)
    if order is None:
        return None
    pts = [m.start_xy] + [m.required(i).xy for i in order] + [m.dest_xy]
    ex = extent(m)
    cell = max(ex / resolution, 1e-6)
    allx = [p[0] for p in pts] + [z.centroid()[0] + s * z.bounding_radius() for z in m.no_fly_zones for s in (-1, 1)]
    ally = [p[1] for p in pts] + [z.centroid()[1] + s * z.bounding_radius() for z in m.no_fly_zones for s in (-1, 1)]
    x0, y0 = min(allx) - 3 * cell, min(ally) - 3 * cell
    nx, ny = int((max(allx) - x0) / cell) + 4, int((max(ally) - y0) / cell) + 4
    blocked = _blocked_grid(m, x0, y0, nx, ny, cell, 0.75 * cell)
    cellof = lambda p: (min(nx - 1, max(0, int((p[0] - x0) / cell))), min(ny - 1, max(0, int((p[1] - y0) / cell))))
    center = lambda c: (x0 + (c[0] + 0.5) * cell, y0 + (c[1] + 0.5) * cell)
    route = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if _free(m, a, b):
            route.append(b)
            continue
        ca, cb = cellof(a), cellof(b)
        blocked[ca[0]][ca[1]] = False
        path = _search(blocked, nx, ny, ca, cb)
        if path is None:
            return None
        route.extend([center(c) for c in path[1:-1]] + [b])
    # line-of-sight smoothing that never skips a required point
    req_pts = {m.required(i).xy: i for i in order}
    out = [route[0]]
    i = 0
    while i < len(route) - 1:
        j = len(route) - 1
        while j > i + 1:
            if _free(m, route[i], route[j]) and not any(route[k] in req_pts for k in range(i + 1, j)):
                break
            j -= 1
        out.append(route[j])
        i = j
    wps, k = [], 0
    for p in out[1:-1]:
        if p in req_pts:
            wid = req_pts[p]
        else:
            k += 1
            wid = f"W{k}"
        wps.append(Waypoint(wid, round(p[0], 3), round(p[1], 3), m.min_altitude, len(wps) + 1))
    return FlightPlan(m.mission_id, wps, Waypoint("DEST", m.destination[0], m.destination[1], m.min_altitude, len(wps) + 1)).validate()

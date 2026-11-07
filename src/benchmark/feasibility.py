"""Deterministic feasibility oracle (a CONSERVATIVE decision procedure, not a complete planner).

Three-valued output with explicit guarantees:
  FEASIBLE   = a constructive witness plan passes the verifier. Sound BY CONSTRUCTION (the verifier is the judge); no optimality claim.
  INFEASIBLE = at least one of SIX exact necessary conditions fails (each is a mathematical argument, see docs/ORACLE_CORRECTNESS.md and `assess`):
               (1) altitude family active and min_altitude > max_altitude
               (2) ordering family active and the precedence pairs contain a cycle
               (3) the (fixed) start, or a required waypoint / the destination that cannot be moved out of a zone within
                   position_tolerance, lies inside a zone that covers the whole altitude band
               (4) straight-line lower bound on route length (reduced by the slack that position_tolerance allows) exceeds max_distance
               (5) the corresponding energy lower bound exceeds max_energy (proxy model, or the PHYSICAL model of src/models/energy.py for any airspeed policy)
               (6) the ROUTED lower bound (shortest path around the inscribed polygons of circular zones and around convex polygonal zones that cover the whole
                   altitude band, minimised over all precedence-feasible visiting orders) exceeds max_distance / max_energy. Only run when no witness was found.
               Only these conditions may produce INFEASIBLE. A failed witness search NEVER does.
  UNKNOWN    = no witness found and no necessary condition violated. This is the oracle's incompleteness region.

Problem class for which FEASIBLE/INFEASIBLE decisions are exact (modulo the numerical tolerance G.TOL): 2D map, circular or polygonal zones,
constant-altitude witness, fixed visiting order derived from the precedence pairs (first topological order), linear energy proxy.
What it does NOT guarantee:
  * route OPTIMALITY: circles are represented by a circumscribed regular n-gon (CIRCLE_SIDES, inflated by `clear`), so witness length is an UPPER bound on
    the shortest path around the inflated circle (wrapped part overestimated by <= tan(pi/n)/(pi/n) - 1 = 2.35 % for n = 12 [v12: was misstated as 1.15 %, which is the value for n = 17]; see scripts/oracle_validation.py);
  * polygon inflation is a vertex push-out along the centroid ray (exact for convex star-shaped zones, heuristic for strongly concave ones -> UNKNOWN, never INFEASIBLE);
  * joint optimisation of visiting order: ties among valid orders are broken by a fixed topological order, so a feasible mission with a better order
    may be reported UNKNOWN when the fixed order is too long (never INFEASIBLE, because the length bound uses the same order only to build the witness,
    see the note in `assess` on why the lower bound is order-independent);
  * altitude/energy co-optimisation: the witness flies at a single altitude (lowest one that clears vertically bounded zones).
"""
from __future__ import annotations
import heapq
import math
from dataclasses import dataclass
from src.models.schemas import Mission, FlightPlan, Waypoint, NoFlyZone
from src.utils import geometry as G
from src.verifier.verifier import verify


CIRCLE_SIDES = 12   # sides of the circumscribed polygon that represents an inflated circle in the visibility graph


@dataclass
class Feasibility:
    status: str
    reason: str
    lower_bound: float | None = None
    witness_distance: float | None = None
    witness_plan: FlightPlan | None = None
    guarantee: str = ""          # "verified_witness" | "necessary_condition_violated" | "inconclusive"
    proof_kind: str | None = None  # which necessary condition fired (INFEASIBLE only)
    route_bracket: tuple | None = None   # (lower, upper) bounds [map units] on the true shortest feasible chain length through the required points (needs certify=True)

    def to_dict(self):
        return {"status": self.status, "reason": self.reason, "lower_bound": self.lower_bound,
                "witness_distance": self.witness_distance, "guarantee": self.guarantee, "proof_kind": self.proof_kind, "route_bracket": list(self.route_bracket) if self.route_bracket else None}


def visiting_order(m: Mission) -> list[str] | None:
    ids = [r.id for r in m.required_waypoints]
    indeg = {i: 0 for i in ids}
    for b, a in m.ordering:
        indeg[a] += 1
    out, left = [], list(ids)
    pairs = [tuple(p) for p in m.ordering]
    while left:
        ready = [i for i in left if indeg[i] == 0]
        if not ready:
            return None
        n = ready[0]
        out.append(n)
        left.remove(n)
        for b, a in pairs:
            if b == n:
                indeg[a] -= 1
    return out


def _inflated(z: NoFlyZone, clear: float, n: int | None = None) -> list[G.Point]:
    n = n or CIRCLE_SIDES
    if z.kind == "circle":
        R = (z.radius + clear) / math.cos(math.pi / n)
        return [(z.center[0] + R * math.cos(2 * math.pi * k / n), z.center[1] + R * math.sin(2 * math.pi * k / n)) for k in range(n)]
    c = z.centroid()
    out = []
    for v in z._verts():
        d = G.dist(c, v) or 1.0
        out.append((v[0] + (v[0] - c[0]) / d * clear * 1.5, v[1] + (v[1] - c[1]) / d * clear * 1.5))
    return out


def _free(zones, a, b) -> bool:
    return not any(z.intersects_segment(a, b) for z in zones)


def _active_zones(m: Mission) -> list[NoFlyZone]:
    return list(m.no_fly_zones) if m.is_active("no_fly_zone") else []


def zones_at_altitude(m: Mission, alt: float) -> list[NoFlyZone]:
    """Active zones that block a constant-altitude flight at `alt`."""
    return [z for z in _active_zones(m) if z.vertical_overlap(alt, alt)]


def witness_altitudes(m: Mission) -> list[float]:
    """Candidate constant altitudes for the witness, lowest first (lowest = least climb energy). With only full-height zones this is just [min_altitude]."""
    lo, hi = m.min_altitude, m.max_altitude
    cands = [lo]
    for z in _active_zones(m):
        if z.is_vertically_bounded:
            if z.z_max is not None:
                cands.append(z.z_max + 0.5)     # fly over
            if z.z_min is not None:
                cands.append(z.z_min - 0.5)     # fly under
    t = m.terrain_grid
    if t is not None:
        chain = [m.start_xy] + [r.xy for r in m.required_waypoints] + [m.dest_xy]
        for gmax in (t.max_height_on(chain), max(max(r) for r in t.heights)):   # along the straight chain, and the worst case over the whole grid
            cands.append(gmax + m.min_agl + 0.5)
    if len(cands) > 1:
        cands.append(hi)     # only needed when vertically bounded zones / terrain make the altitude a real decision
    ok = sorted({round(c, 6) for c in cands if (lo - G.TOL <= c <= hi + G.TOL) or not m.is_active("altitude")})
    if t is not None and m.is_active("altitude"):
        ok = [c for c in ok if c >= lo - G.TOL]
    return ok or [lo]


def extent(m: Mission) -> float:
    pts = [m.start_xy, m.dest_xy] + [r.xy for r in m.required_waypoints] + [z.centroid() for z in m.no_fly_zones]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(max(xs) - min(xs), max(ys) - min(ys), 1.0)


def buffered(z: NoFlyZone, d: float) -> NoFlyZone:
    """A zone grown by roughly d map units (circle: exact radius + d; polygon: vertices pushed out along the centroid ray, as in _inflated)."""
    if d <= 0:
        return z
    if z.kind == "circle":
        return NoFlyZone(z.id, "circle", center=list(z.center), radius=z.radius + d, z_min=z.z_min, z_max=z.z_max)
    return NoFlyZone(z.id, "polygon", vertices=[list(v) for v in _inflated(z, d / 1.5)], z_min=z.z_min, z_max=z.z_max)


def shortest_route(m: Mission, pts: list[G.Point], clear: float | None = None, zones: list[NoFlyZone] | None = None,
                   n_sides: int | None = None, edge_clearance: float = 0.0) -> list[G.Point] | None:
    """Shortest path through `pts` (in order) avoiding `zones` (default: all active zones, i.e. flight at an altitude every zone blocks),
    in the visibility graph of the zones inflated by `clear`. Length is an upper bound of the true shortest path (see module docstring).
    edge_clearance (v12, default 0 = original behaviour): when > 0, every leg must also avoid the zones grown by this many map units, so the route keeps
    that clearance along its legs and not only at its corner points (with 0, `clear` only moves the candidate corners; legs are checked against the
    true zones and may graze them). Used for execution-aware planning (scripts/execution_clearance_study.py)."""
    clear = clear if clear is not None else max(0.5, extent(m) * 0.002)  # scale-aware (0.5 for the 0-100 synthetic map)
    zones = _active_zones(m) if zones is None else zones
    if edge_clearance > 0:      # legs and corners must clear the GROWN zones; `clear` stays the small corner margin around them
        zones = [buffered(z, edge_clearance) for z in zones]
    verts = [v for z in zones for v in _inflated(z, clear, n_sides) if not any(zz.contains(v) for zz in zones)]
    nv = len(verts)
    vis = [[False] * nv for _ in range(nv)]
    for i in range(nv):
        for j in range(i + 1, nv):
            vis[i][j] = vis[j][i] = _free(zones, verts[i], verts[j])
    path: list[G.Point] = [pts[0]]
    for a, b in zip(pts, pts[1:]):
        if _free(zones, a, b):
            path.append(b)
            continue
        nodes = [a, b] + verts  # 0=a, 1=b
        d = {0: 0.0}
        prev: dict[int, int] = {}
        pq = [(0.0, 0)]
        while pq:
            dd, u = heapq.heappop(pq)
            if dd > d.get(u, 1e18):
                continue
            if u == 1:
                break
            for w in range(1, len(nodes)):
                if w == u:
                    continue
                if u >= 2 and w >= 2:
                    ok = vis[u - 2][w - 2]
                else:
                    ok = _free(zones, nodes[u], nodes[w])
                if ok:
                    nd = dd + G.dist(nodes[u], nodes[w])
                    if nd < d.get(w, 1e18) - 1e-12:
                        d[w], prev[w] = nd, u
                        heapq.heappush(pq, (nd, w))
        if 1 not in d:
            return None
        seq, u = [], 1
        while u != 0:
            seq.append(u)
            u = prev[u]
        path.extend(nodes[i] for i in reversed(seq))
    return path


def _movable_out_of_zone(m: Mission, p: G.Point, z: NoFlyZone) -> bool:
    """True if a point that the plan may displace by position_tolerance could leave zone z (so 'inside z' is NOT a proof of infeasibility)."""
    if z.kind == "circle":
        depth = z.radius - G.dist(p, z.centroid())
    else:
        depth = G.min_dist_to_polygon_boundary(p, z._verts())
    return depth <= m.position_tolerance + G.TOL


def assess(m: Mission, certify: bool = False, clear: float | None = None, edge_clearance: float = 0.0) -> Feasibility:
    """clear = extra clearance (map units) around inflated zones for the witness's corner points (default: tiny, 0.2 % of the map extent).
    edge_clearance > 0 (v12) also keeps every witness LEG that far from the zones (execution-aware witness). Defaults reproduce the original oracle exactly."""
    f = _assess(m, clear, edge_clearance)
    if certify and f.status == "FEASIBLE" and f.witness_distance is not None:
        lb = routed_lower_bound(m)
        if lb is not None:
            f.route_bracket = (round(max(lb, 0.0), 6), round(chain_length(f.witness_plan, m), 6))
    return f


def chain_length(plan, m: Mission) -> float:
    pts = plan.route_points(m)
    return sum(G.dist(pts[i][1], pts[i + 1][1]) for i in range(len(pts) - 1))


def _assess(m: Mission, clear: float | None = None, edge_clearance: float = 0.0) -> Feasibility:
    alt_on = m.is_active("altitude")
    lo, hi = m.min_altitude, m.max_altitude
    if alt_on and lo > hi:
        return Feasibility("INFEASIBLE", "empty altitude band", guarantee="necessary_condition_violated", proof_kind="empty_altitude_band")
    order = visiting_order(m)
    if order is None:
        if m.is_active("ordering"):
            return Feasibility("INFEASIBLE", "contradictory (cyclic) ordering constraints", guarantee="necessary_condition_violated", proof_kind="cyclic_ordering")
        order = [r.id for r in m.required_waypoints]   # ordering family inactive: precedence pairs are not enforced
    zones = _active_zones(m)
    band_lo, band_hi = (lo, hi) if alt_on else (float("-inf"), float("inf"))
    # (3) a point that must be visited / cannot be displaced out of a zone covering the whole band. The start is fixed (no tolerance);
    #     required waypoints and the destination may move by position_tolerance, so a shallow intrusion is NOT a proof (-> witness search decides).
    for name, p, fixed in [("start", m.start_xy, True), ("destination", m.dest_xy, False)] + [(f"required waypoint {r.id}", r.xy, False) for r in m.required_waypoints]:
        for z in zones:
            if z.contains(p) and z.covers_all_altitudes(band_lo, band_hi) and (fixed or not _movable_out_of_zone(m, p, z)):
                return Feasibility("INFEASIBLE", f"{name} lies inside no-fly zone {z.id}", guarantee="necessary_condition_violated", proof_kind="point_in_zone")
    pts = [m.start_xy] + [m.required(i).xy for i in order] + [m.dest_xy]
    lb = sum(G.dist(a, b) for a, b in zip(pts, pts[1:]))
    # Order-independence note: with the order fixed by `order`, lb is the length of the polyline through the (displaceable) points. Every precedence-feasible
    # order is at least as long as the shortest one, so we only use `lb` as a proof when ordering is a total order; otherwise we use the minimum over
    # orders (n_req <= 5 in all benchmarks => at most 120 permutations).
    lb_sound = _min_length_over_orders(m) - m.position_tolerance * (2 * len(m.required_waypoints) + 1)
    lb_sound = max(0.0, lb_sound)
    if m.is_active("distance") and lb_sound > m.max_distance + G.TOL:
        return Feasibility("INFEASIBLE", f"straight-line lower bound {lb_sound:.2f} exceeds max_distance {m.max_distance}", lower_bound=lb_sound,
                           guarantee="necessary_condition_violated", proof_kind="distance_lower_bound")
    if m.is_active("energy") and m.max_energy is not None:
        asc = max(0.0, lo - m.start_altitude) if alt_on else 0.0
        e_lb = _energy_lb(m, lb_sound, asc, len(order))
        if e_lb > m.max_energy + G.TOL:
            return Feasibility("INFEASIBLE", f"energy lower bound {e_lb:.2f} exceeds max_energy {m.max_energy}", lower_bound=lb_sound,
                               guarantee="necessary_condition_violated", proof_kind="energy_lower_bound")
    req_at = {m.required(i).xy: i for i in order}
    last_dist = None
    best = None            # (cost, plan, distance): with several candidate altitudes the CHEAPEST verified witness is returned (energy if active, else distance)
    alts_try = witness_altitudes(m)
    use_energy = m.is_active("energy") and m.max_energy is not None
    for alt in alts_try:
        route = shortest_route(m, pts, clear=clear, zones=zones_at_altitude(m, alt), edge_clearance=edge_clearance)
        if route is None:
            continue
        wps, k = [], 0
        for p in route[1:-1]:
            if p in req_at:
                wid = req_at[p]
            else:
                k += 1
                while f"W{k}" in {r.id for r in m.required_waypoints}:   # never collide with a required waypoint id
                    k += 1
                wid = f"W{k}"
            wps.append(Waypoint(wid, p[0], p[1], alt, len(wps) + 1))
        plan = FlightPlan(m.mission_id, wps, Waypoint("DEST", m.destination[0], m.destination[1], alt, len(wps) + 1)).validate()
        res = verify(m, plan)
        last_dist = res.route_distance
        if res.success:
            cost = res.energy if use_energy else res.route_distance
            if len(alts_try) == 1:
                return Feasibility("FEASIBLE", "verified witness plan exists", lower_bound=lb_sound, witness_distance=res.route_distance, witness_plan=plan,
                                   guarantee="verified_witness")
            if best is None or cost < best[0] - 1e-9:
                best = (cost, plan, res.route_distance)
    if best is None and m.terrain_grid is not None:     # constant altitude cannot satisfy min AND max AGL over hilly terrain: terrain-following witness
        route = shortest_route(m, pts, clear=clear, zones=zones, edge_clearance=edge_clearance)
        if route is not None:
            plan = _terrain_following_plan(m, route, req_at)
            if plan is not None:
                res = verify(m, plan)
                last_dist = res.route_distance
                if res.success:
                    best = (0.0, plan, res.route_distance)
    if best is not None:
        return Feasibility("FEASIBLE", "verified witness plan exists", lower_bound=lb_sound, witness_distance=best[2], witness_plan=best[1], guarantee="verified_witness")
    # (6) routed lower bound: tried only when no witness exists (it upgrades some UNKNOWN verdicts to proofs)
    rlb = routed_lower_bound(m)
    if rlb is not None:
        rlb_sound = max(0.0, rlb - m.position_tolerance * (2 * len(m.required_waypoints) + 1))
        asc = max(0.0, lo - m.start_altitude) if alt_on else 0.0
        if m.is_active("distance") and rlb_sound > m.max_distance + G.TOL:
            return Feasibility("INFEASIBLE", f"routed lower bound {rlb_sound:.2f} exceeds max_distance {m.max_distance}", lower_bound=rlb_sound,
                               guarantee="necessary_condition_violated", proof_kind="routed_distance_lower_bound")
        if m.is_active("energy") and m.max_energy is not None and _energy_lb(m, rlb_sound, asc, len(order)) > m.max_energy + G.TOL:
            return Feasibility("INFEASIBLE", "routed energy lower bound exceeds max_energy", lower_bound=rlb_sound, guarantee="necessary_condition_violated",
                               proof_kind="routed_energy_lower_bound")
        lb_sound = max(lb_sound, rlb_sound)
    if last_dist is None:
        return Feasibility("UNKNOWN", "witness search found no route", lower_bound=lb_sound, guarantee="inconclusive")
    return Feasibility("UNKNOWN", "witness plan did not satisfy constraints (likely distance); no impossibility proof",
                       lower_bound=lb_sound, witness_distance=last_dist, guarantee="inconclusive")


def _min_length_over_orders(m: Mission) -> float:
    """Length of the shortest polyline START -> (all required points in any precedence-feasible order) -> DEST, ignoring zones (a valid lower bound)."""
    import itertools
    ids = [r.id for r in m.required_waypoints]
    pairs = [tuple(p) for p in m.ordering] if m.is_active("ordering") else []
    best = None
    if len(ids) > 7:   # factorial guard: fall back to the fixed topological order (still a valid length, but only an upper bound of the minimum)
        order = visiting_order(m) or ids
        pts = [m.start_xy] + [m.required(i).xy for i in order] + [m.dest_xy]
        return sum(G.dist(a, b) for a, b in zip(pts, pts[1:])) if not pairs else 0.0
    for perm in itertools.permutations(ids):
        pos = {i: k for k, i in enumerate(perm)}
        if any(pos[a] > pos[b] for a, b in pairs):
            continue
        pts = [m.start_xy] + [m.required(i).xy for i in perm] + [m.dest_xy]
        d = sum(G.dist(a, b) for a, b in zip(pts, pts[1:]))
        best = d if best is None else min(best, d)
    return best if best is not None else 0.0


def _energy_lb(m: Mission, length_lb: float, min_ascent_m: float, n_waypoints: int) -> float:
    if m.energy_model == "physical":
        from src.models.energy import energy_lower_bound_wh
        return energy_lower_bound_wh(m, length_lb, min_ascent_m)   # valid for any airspeed policy; waypoint hover omitted (>= 0)
    return length_lb + m.energy_k_v * min_ascent_m + m.energy_k_wp * n_waypoints


# ---------------------------------------------------------------------------------------------------------------------------------------------
#  Routed lower bound (necessary condition 6)
# ---------------------------------------------------------------------------------------------------------------------------------------------
LB_CIRCLE_SIDES = 24


def _is_convex(verts) -> bool:
    n, sign = len(verts), 0
    for i in range(n):
        a, b, c = verts[i], verts[(i + 1) % n], verts[(i + 2) % n]
        cr = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cr) < 1e-12:
            continue
        if sign == 0:
            sign = 1 if cr > 0 else -1
        elif (cr > 0) != (sign > 0):
            return False
    return True


def _lb_obstacles(m: Mission) -> list[list[G.Point]]:
    """Polygons that are SUBSETS of blocked regions (so shortest paths around them are lower bounds): inscribed polygons of circular zones and convex polygonal
    zones, restricted to zones that cover the whole altitude band (they cannot be overflown). Non-convex polygons are dropped (fewer obstacles => still a lower bound)."""
    lo, hi = (m.min_altitude, m.max_altitude) if m.is_active("altitude") else (float("-inf"), float("inf"))
    out = []
    for z in _active_zones(m):
        if not z.covers_all_altitudes(lo, hi):
            continue
        if z.kind == "circle":
            c = z.centroid()
            verts = [(c[0] + z.radius * math.cos(2 * math.pi * k / LB_CIRCLE_SIDES), c[1] + z.radius * math.sin(2 * math.pi * k / LB_CIRCLE_SIDES)) for k in range(LB_CIRCLE_SIDES)]
        else:
            verts = [tuple(v) for v in z._verts()]
            if not _is_convex(verts):
                continue
        out.append(verts)
    return out


def _routed_pairwise(m: Mission, keys: list[G.Point], polys: list[list[G.Point]]):
    """All-pairs shortest-path lengths between `keys` around the polygons (open interiors blocked); None if a key lies inside an obstacle."""
    import heapq
    shrunk = []
    for v in polys:
        cx, cy = G.polygon_centroid(v)
        sv = [[cx + (1 - 1e-9) * (x - cx), cy + (1 - 1e-9) * (y - cy)] for x, y in v]
        shrunk.append(NoFlyZone("S", "polygon", vertices=sv))
    bbox = [(min(x for x, _ in v), min(y for _, y in v), max(x for x, _ in v), max(y for _, y in v)) for v in polys]
    nodes = list(keys) + [p for v in polys for p in v]
    for k in keys:
        if any(z.contains(k) for z in shrunk):
            return None

    def free(a, b):
        x0, x1, y0, y1 = min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])
        for z, (bx0, by0, bx1, by1) in zip(shrunk, bbox):
            if x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1:
                continue
            if z.intersects_segment(a, b):
                return False
        return True

    N = len(nodes)
    adj = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if i < len(keys) and j < len(keys) or True:
                if free(nodes[i], nodes[j]):
                    d = G.dist(nodes[i], nodes[j])
                    adj[i].append((j, d))
                    adj[j].append((i, d))
    out = {}
    for s_i in range(len(keys)):
        dist = [float("inf")] * N
        dist[s_i] = 0.0
        pq = [(0.0, s_i)]
        while pq:
            d, u = heapq.heappop(pq)
            if d > dist[u]:
                continue
            for v, w in adj[u]:
                if d + w < dist[v]:
                    dist[v] = d + w
                    heapq.heappush(pq, (dist[v], v))
        for t_i in range(len(keys)):
            out[(s_i, t_i)] = dist[t_i]
    return out


def routed_lower_bound(m: Mission) -> float | None:
    """Lower bound on the length of ANY route that visits the required waypoints (at their coordinates) in some precedence-feasible order and avoids the zones that
    cannot be overflown. Returns None if not computable. Valid for all altitude profiles because only zones covering the whole band are used."""
    import itertools
    ids = [r.id for r in m.required_waypoints]
    if len(ids) > 6:
        return None
    polys = _lb_obstacles(m)
    keys = [m.start_xy] + [m.required(i).xy for i in ids] + [m.dest_xy]
    pw = _routed_pairwise(m, keys, polys)
    if pw is None:
        return None
    pairs = [tuple(p) for p in m.ordering] if m.is_active("ordering") else []
    best = None
    for perm in itertools.permutations(range(len(ids))):
        pos = {ids[k]: i for i, k in enumerate(perm)}
        if any(pos[a] > pos[b] for a, b in pairs):
            continue
        seq = [0] + [k + 1 for k in perm] + [len(keys) - 1]
        d = sum(pw[(a, b)] for a, b in zip(seq, seq[1:]))
        best = d if best is None else min(best, d)
    return best


def _terrain_following_plan(m: Mission, route: list, req_at: dict) -> FlightPlan | None:
    """Densify the horizontal route at half-cell spacing and set each altitude to the highest (ground + min_agl + margin) within one spacing, clamped to the altitude band.
    Verified afterwards by the verifier (never trusted)."""
    t = m.terrain_grid
    margin = max(2.0, 0.05 * (m.max_agl or 100.0))
    step = t.cell / 2.0
    pts, kinds = [route[0]], ["start"]
    for a, b in zip(route, route[1:]):
        L = G.dist(a, b)
        n = max(1, int(math.ceil(L / step)))
        for k in range(1, n + 1):
            q = (a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n)
            pts.append(q if k < n else b)
            kinds.append("route" if k < n else "vertex")
    lo, hi = (m.min_altitude, m.max_altitude) if m.is_active("altitude") else (float("-inf"), float("inf"))
    need = [t.height_at(*p) + m.min_agl + margin for p in pts]
    alt = [max(need[max(0, i - 1):i + 2]) for i in range(len(pts))]
    alt = [min(max(a, lo), hi) for a in alt]
    wps, k = [], 0
    ids = {r.id for r in m.required_waypoints}
    for i in range(1, len(pts) - 1):
        if pts[i] in req_at:
            wid = req_at[pts[i]]
        else:
            k += 1
            while f"T{k}" in ids:
                k += 1
            wid = f"T{k}"
        wps.append(Waypoint(wid, pts[i][0], pts[i][1], alt[i], len(wps) + 1))
    try:
        return FlightPlan(m.mission_id, wps, Waypoint("DEST", m.destination[0], m.destination[1], alt[-1], len(wps) + 1)).validate()
    except Exception:
        return None

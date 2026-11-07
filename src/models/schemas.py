"""Strict data models (stdlib dataclasses with explicit validation; pydantic-free by design
so the core runs with only numpy/pandas/matplotlib/scipy/pyyaml)."""
from __future__ import annotations
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Any

from src.utils import geometry as G

ALL_FAMILIES = ("altitude", "no_fly_zone", "distance", "ordering", "energy")
ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")
RESERVED_IDS = {"START", "DEST"}


class SchemaError(ValueError):
    """Raised when data does not conform to the schema."""


def _num(v: Any, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise SchemaError(f"{name} must be a number, got {type(v).__name__}: {v!r}")
    if not math.isfinite(v):
        raise SchemaError(f"{name} must be finite")
    return float(v)


def _keys(d: Any, required: set[str], optional: set[str], name: str) -> None:
    if not isinstance(d, dict):
        raise SchemaError(f"{name} must be an object")
    missing = required - set(d)
    if missing:
        raise SchemaError(f"{name} missing fields: {sorted(missing)}")
    extra = set(d) - required - optional
    if extra:
        raise SchemaError(f"{name} has unknown fields: {sorted(extra)}")


def _xy(v: Any, name: str) -> tuple[float, float]:
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise SchemaError(f"{name} must be [x, y]")
    return (_num(v[0], name + "[0]"), _num(v[1], name + "[1]"))


@dataclass
class Waypoint:
    id: str
    x: float
    y: float
    altitude: float
    order: int = 0

    def validate(self) -> "Waypoint":
        if not isinstance(self.id, str) or not ID_RE.match(self.id):
            raise SchemaError(f"invalid waypoint id {self.id!r}")
        self.x, self.y, self.altitude = _num(self.x, "x"), _num(self.y, "y"), _num(self.altitude, "altitude")
        if isinstance(self.order, bool) or not isinstance(self.order, int) or self.order < 0:
            raise SchemaError("order must be a non-negative integer")
        return self

    @property
    def xy(self) -> G.Point:
        return (self.x, self.y)


@dataclass
class RequiredWaypoint:
    """A named location the mission requires the plan to visit (coordinates fixed by the mission)."""
    id: str
    x: float
    y: float

    def validate(self) -> "RequiredWaypoint":
        if not isinstance(self.id, str) or not ID_RE.match(self.id) or self.id in RESERVED_IDS:
            raise SchemaError(f"invalid required waypoint id {self.id!r}")
        self.x, self.y = _num(self.x, "x"), _num(self.y, "y")
        return self

    @property
    def xy(self) -> G.Point:
        return (self.x, self.y)


@dataclass
class NoFlyZone:
    id: str
    kind: str  # "circle" | "polygon"
    center: list[float] | None = None
    radius: float | None = None
    vertices: list[list[float]] | None = None
    description: str = ""
    category: str = "nfz"   # "nfz" (restricted airspace) | "obstacle" (building / tower / terrain feature)
    # Optional VERTICAL extent (metres, same unit as altitude). None = unbounded on that side (a full-height 2D zone, the original behaviour).
    # A zone with z_max=60 is e.g. a 60 m building: flying above 60 m clears it. The zone only blocks where altitude is within [z_min, z_max] (closed).
    z_min: float | None = None
    z_max: float | None = None

    def validate(self) -> "NoFlyZone":
        if not isinstance(self.id, str) or not ID_RE.match(self.id):
            raise SchemaError(f"invalid zone id {self.id!r}")
        if self.category not in ("nfz", "obstacle"):
            raise SchemaError(f"unknown zone category {self.category!r}")
        if self.kind == "circle":
            if self.center is None or self.radius is None:
                raise SchemaError("circle zone needs center and radius")
            self.center = list(_xy(self.center, "center"))
            self.radius = _num(self.radius, "radius")
            if self.radius <= 0:
                raise SchemaError("radius must be > 0")
            self.vertices = None
        elif self.kind == "polygon":
            if not self.vertices or len(self.vertices) < 3:
                raise SchemaError("polygon zone needs >= 3 vertices")
            self.vertices = [list(_xy(v, "vertex")) for v in self.vertices]
            self.center = self.radius = None
        else:
            raise SchemaError(f"unknown zone kind {self.kind!r}")
        if self.z_min is not None:
            self.z_min = _num(self.z_min, "z_min")
        if self.z_max is not None:
            self.z_max = _num(self.z_max, "z_max")
        if self.z_min is not None and self.z_max is not None and self.z_min > self.z_max:
            raise SchemaError("zone z_min must be <= z_max")
        return self

    # ---- vertical extent (3D-aware zones) ----
    @property
    def is_vertically_bounded(self) -> bool:
        return self.z_min is not None or self.z_max is not None

    def vertical_overlap(self, lo: float, hi: float) -> bool:
        """True if the closed altitude interval [lo, hi] intersects the zone's closed vertical extent."""
        if lo > hi:
            lo, hi = hi, lo
        if self.z_min is not None and hi < self.z_min - G.TOL:
            return False
        if self.z_max is not None and lo > self.z_max + G.TOL:
            return False
        return True

    def covers_all_altitudes(self, lo: float, hi: float) -> bool:
        """True if every altitude in [lo, hi] lies inside the zone's vertical extent (the zone cannot be flown over/under within the band)."""
        return (self.z_min is None or self.z_min <= lo + G.TOL) and (self.z_max is None or self.z_max >= hi - G.TOL)

    def contains_at(self, p: G.Point, altitude: float) -> bool:
        return self.contains(p) and self.vertical_overlap(altitude, altitude)

    def inside_intervals(self, a: G.Point, b: G.Point) -> list[tuple[float, float]]:
        if self.kind == "circle":
            iv = G.circle_chord_t(a, b, self.centroid(), self.radius)
            return [iv] if iv else []
        return G.polygon_inside_intervals(a, b, self._verts())

    def intersects_segment_3d(self, a: G.Point, b: G.Point, alt_a: float, alt_b: float) -> bool:
        """Closed-set intersection of the 3D segment (a,alt_a)->(b,alt_b) (altitude linear in the horizontal parameter) with the zone prism."""
        if not self.is_vertically_bounded:
            return self.intersects_segment(a, b)
        for t0, t1 in self.inside_intervals(a, b):
            h0, h1 = alt_a + t0 * (alt_b - alt_a), alt_a + t1 * (alt_b - alt_a)
            if self.vertical_overlap(min(h0, h1), max(h0, h1)):
                return True
        return False

    # ---- geometry (closed-set semantics) ----
    def _verts(self) -> list[G.Point]:
        return [(v[0], v[1]) for v in self.vertices]

    def centroid(self) -> G.Point:
        return (self.center[0], self.center[1]) if self.kind == "circle" else G.polygon_centroid(self._verts())

    def bounding_radius(self) -> float:
        if self.kind == "circle":
            return self.radius
        c = self.centroid()
        return max(G.dist(c, v) for v in self._verts())

    def area(self) -> float:
        if self.kind == "circle":
            return math.pi * self.radius ** 2
        v = self._verts()
        return abs(sum(v[i][0] * v[(i + 1) % len(v)][1] - v[(i + 1) % len(v)][0] * v[i][1] for i in range(len(v)))) / 2

    def contains(self, p: G.Point) -> bool:
        if self.kind == "circle":
            return G.dist(p, self.centroid()) <= self.radius + G.TOL
        return G.point_in_polygon(p, self._verts()) != "outside"

    def intersects_segment(self, a: G.Point, b: G.Point) -> bool:
        if self.kind == "circle":
            return G.closest_on_segment(self.centroid(), a, b)[2] <= self.radius + G.TOL
        vs = self._verts()
        # exact fast reject (v12): any contact under the closed-set test (tolerance G.TOL) needs the bounding boxes to overlap within that tolerance,
        # so a gap larger than 1e-6 can never hide an intersection; this only skips work (large building sets made the oracle slow).
        e = 1e-6
        if (max(a[0], b[0]) < min(v[0] for v in vs) - e or min(a[0], b[0]) > max(v[0] for v in vs) + e
                or max(a[1], b[1]) < min(v[1] for v in vs) - e or min(a[1], b[1]) > max(v[1] for v in vs) + e):
            return False
        if G.point_in_polygon(a, vs) != "outside" or G.point_in_polygon(b, vs) != "outside":
            return True
        return any(G.segments_intersect(a, b, vs[i], vs[(i + 1) % len(vs)]) for i in range(len(vs)))

    def point_evidence(self, p: G.Point) -> dict:
        if self.kind == "circle":
            d = G.dist(p, self.centroid())
            return {"zone_kind": "circle", "center": self.center, "radius": self.radius,
                    "distance_to_center": round(d, 4), "penetration_depth": round(self.radius - d, 4)}
        return {"zone_kind": "polygon", "vertices": self.vertices,
                "distance_to_boundary": round(G.min_dist_to_polygon_boundary(p, self._verts()), 4)}

    def segment_evidence(self, a: G.Point, b: G.Point) -> dict:
        if self.kind == "circle":
            c = self.centroid()
            cp, t, d = G.closest_on_segment(c, a, b)
            ev = {"zone_kind": "circle", "center": self.center, "radius": self.radius,
                  "closest_point_on_segment": [round(cp[0], 4), round(cp[1], 4)],
                  "min_distance_to_center": round(d, 4), "penetration_depth": round(self.radius - d, 4)}
            ch = G.circle_chord(a, b, c, self.radius)
            if ch:
                ev["entry_point"] = [round(ch[0][0], 4), round(ch[0][1], 4)]
                ev["exit_point"] = [round(ch[1][0], 4), round(ch[1][1], 4)]
            return ev
        vs = self._verts()
        hits = []
        for i in range(len(vs)):
            r = G.segment_intersection_point(a, b, vs[i], vs[(i + 1) % len(vs)])
            if r:
                hits.append(r)
        hits.sort()
        ev: dict = {"zone_kind": "polygon", "vertices": self.vertices, "n_edge_crossings": len(hits),
                    "centroid": [round(x, 4) for x in self.centroid()]}
        if hits:
            ev["entry_point"] = [round(hits[0][1][0], 4), round(hits[0][1][1], 4)]
            ev["exit_point"] = [round(hits[-1][1][0], 4), round(hits[-1][1][1], 4)]
        return ev


@dataclass
class Mission:
    mission_id: str
    description: str
    start: list[float]
    destination: list[float]
    required_waypoints: list[RequiredWaypoint]
    min_altitude: float
    max_altitude: float
    max_distance: float
    no_fly_zones: list[NoFlyZone] = field(default_factory=list)
    ordering: list[list[str]] = field(default_factory=list)  # pairs [before, after]
    position_tolerance: float = 1.0
    # --- resource model (survey: "distance, energy, time") ---
    start_altitude: float = 0.0            # >0 when the UAV is already airborne (dynamic replanning)
    max_energy: float | None = None        # battery budget in energy units; None = no energy constraint
    energy_k_v: float = 0.5                # energy units per unit of climb
    energy_k_wp: float = 0.0               # energy units per waypoint (hover / turn overhead)
    speed_h: float = 10.0                  # horizontal speed (units/s) for the flight-time estimate
    speed_v: float = 2.0                   # climb speed (units/s)
    active: list = field(default_factory=lambda: list(ALL_FAMILIES))   # constraint families enforced (controlled single/combined tests)
    # --- v10: physical models (all optional; defaults reproduce the earlier behaviour exactly) ---
    unit_m: float = 1.0                    # metres per map unit (needed by the physical energy model, terrain and dynamics)
    energy_model: str = "proxy"            # "proxy": E = L + k_v A + k_wp n (route-length units) | "physical": battery energy in Wh (src/models/energy.py)
    vehicle: Any = None                    # vehicle profile name or parameter dict (config/vehicles.yaml); required for energy_model="physical"
    payload_kg: float = 0.0
    wind: list | None = None               # steady wind [east, north] m/s used by the physical energy model (and as the mean wind of the simulator)
    terrain: dict | None = None            # TerrainGrid.to_dict(); when present the terrain-clearance family is enforced
    min_agl: float = 0.0                   # minimum clearance above terrain (m); only with terrain
    max_agl: float | None = None           # maximum height above terrain (m), e.g. a 120 m AGL ceiling; only with terrain
    terrain_exempt_m: float = 0.0          # radius (m) around START and DEST where min_agl is not enforced (take-off / landing happen at ground level)

    def is_active(self, fam: str) -> bool:
        return fam in self.active

    @property
    def terrain_grid(self):
        if self.terrain is None:
            return None
        if getattr(self, "_terrain_cache", None) is None:
            from src.models.terrain import TerrainGrid
            object.__setattr__(self, "_terrain_cache", TerrainGrid.from_dict(self.terrain))
        return self._terrain_cache

    def validate(self) -> "Mission":
        if not isinstance(self.mission_id, str) or not re.match(r"^[A-Za-z0-9_\-]{1,80}$", self.mission_id):
            raise SchemaError("invalid mission_id")
        self.start, self.destination = list(_xy(self.start, "start")), list(_xy(self.destination, "destination"))
        for r in self.required_waypoints:
            r.validate()
        ids = [r.id for r in self.required_waypoints]
        if len(set(ids)) != len(ids):
            raise SchemaError("duplicate required waypoint ids")
        self.min_altitude, self.max_altitude = _num(self.min_altitude, "min_altitude"), _num(self.max_altitude, "max_altitude")
        self.max_distance = _num(self.max_distance, "max_distance")
        if self.max_distance <= 0:
            raise SchemaError("max_distance must be > 0")
        self.position_tolerance = _num(self.position_tolerance, "position_tolerance")
        self.start_altitude = _num(self.start_altitude, "start_altitude")
        if self.max_energy is not None:
            self.max_energy = _num(self.max_energy, "max_energy")
            if self.max_energy <= 0:
                raise SchemaError("max_energy must be > 0")
        for nm in ("energy_k_v", "energy_k_wp", "speed_h", "speed_v"):
            setattr(self, nm, _num(getattr(self, nm), nm))
        if self.speed_h <= 0 or self.speed_v <= 0:
            raise SchemaError("speeds must be > 0")
        if not set(self.active) <= set(ALL_FAMILIES):
            raise SchemaError(f"active must be a subset of {ALL_FAMILIES}")
        self.unit_m = _num(self.unit_m, "unit_m")
        self.payload_kg = _num(self.payload_kg, "payload_kg")
        self.min_agl = _num(self.min_agl, "min_agl")
        if self.unit_m <= 0 or self.payload_kg < 0 or self.min_agl < 0:
            raise SchemaError("unit_m must be > 0; payload_kg and min_agl must be >= 0")
        if self.max_agl is not None:
            self.max_agl = _num(self.max_agl, "max_agl")
        if self.energy_model not in ("proxy", "physical"):
            raise SchemaError("energy_model must be 'proxy' or 'physical'")
        if self.energy_model == "physical":
            if self.vehicle is None:
                raise SchemaError("energy_model 'physical' requires a vehicle profile")
            from src.models.vehicle import get_vehicle
            v = get_vehicle(self.vehicle)
            if self.payload_kg > v.payload_max_kg + 1e-12:
                raise SchemaError(f"payload {self.payload_kg} kg exceeds vehicle payload capacity {v.payload_max_kg} kg")
        if self.wind is not None:
            if not isinstance(self.wind, (list, tuple)) or len(self.wind) != 2:
                raise SchemaError("wind must be [east, north] in m/s")
            self.wind = [_num(self.wind[0], "wind[0]"), _num(self.wind[1], "wind[1]")]
        if self.terrain is not None:
            from src.models.terrain import TerrainGrid
            try:
                TerrainGrid.from_dict(self.terrain)
            except (KeyError, ValueError, TypeError) as e:
                raise SchemaError(f"invalid terrain: {e}")
        zids = [z.id for z in self.no_fly_zones]
        if len(set(zids)) != len(zids):
            raise SchemaError("duplicate zone ids")
        for z in self.no_fly_zones:
            z.validate()
        for p in self.ordering:
            if not isinstance(p, (list, tuple)) or len(p) != 2 or p[0] not in ids or p[1] not in ids:
                raise SchemaError(f"ordering pair {p!r} must reference required waypoint ids")
        return self

    @property
    def start_xy(self) -> G.Point:
        return (self.start[0], self.start[1])

    @property
    def dest_xy(self) -> G.Point:
        return (self.destination[0], self.destination[1])

    def zone(self, zid: str) -> NoFlyZone:
        for z in self.no_fly_zones:
            if z.id == zid:
                return z
        raise KeyError(zid)

    def required(self, rid: str) -> RequiredWaypoint:
        for r in self.required_waypoints:
            if r.id == rid:
                return r
        raise KeyError(rid)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Mission":
        _keys(d, {"mission_id", "description", "start", "destination", "required_waypoints", "min_altitude",
                  "max_altitude", "max_distance"},
              {"no_fly_zones", "ordering", "position_tolerance", "start_altitude", "max_energy", "energy_k_v", "energy_k_wp", "speed_h", "speed_v", "active",
                                                  "unit_m", "energy_model", "vehicle", "payload_kg", "wind", "terrain", "min_agl", "max_agl", "terrain_exempt_m"}, "mission")
        return Mission(
            mission_id=d["mission_id"], description=d["description"], start=d["start"], destination=d["destination"],
            required_waypoints=[RequiredWaypoint(**r) for r in d["required_waypoints"]],
            min_altitude=d["min_altitude"], max_altitude=d["max_altitude"], max_distance=d["max_distance"],
            no_fly_zones=[NoFlyZone(**z) for z in d.get("no_fly_zones", [])],
            ordering=[list(p) for p in d.get("ordering", [])],
            position_tolerance=d.get("position_tolerance", 1.0), start_altitude=d.get("start_altitude", 0.0), max_energy=d.get("max_energy"),
            energy_k_v=d.get("energy_k_v", 0.5), energy_k_wp=d.get("energy_k_wp", 0.0), speed_h=d.get("speed_h", 10.0), speed_v=d.get("speed_v", 2.0),
            active=list(d.get("active", ALL_FAMILIES)), unit_m=d.get("unit_m", 1.0), energy_model=d.get("energy_model", "proxy"), vehicle=d.get("vehicle"),
            payload_kg=d.get("payload_kg", 0.0), wind=d.get("wind"), terrain=d.get("terrain"), min_agl=d.get("min_agl", 0.0), max_agl=d.get("max_agl"), terrain_exempt_m=d.get("terrain_exempt_m", 0.0)).validate()


@dataclass
class FlightPlan:
    mission_id: str
    waypoints: list[Waypoint]
    destination: Waypoint

    def validate(self) -> "FlightPlan":
        ids = set()
        orders = set()
        for w in self.waypoints:
            w.validate()
            if w.id in RESERVED_IDS:
                raise SchemaError(f"waypoint id {w.id!r} is reserved")
            if w.id in ids:
                raise SchemaError(f"duplicate waypoint id {w.id!r}")
            if w.order in orders:
                raise SchemaError(f"duplicate waypoint order {w.order}")
            ids.add(w.id)
            orders.add(w.order)
        self.destination.validate()
        return self

    def ordered(self) -> list[Waypoint]:
        return sorted(self.waypoints, key=lambda w: w.order)

    def route_points(self, mission: Mission) -> list[tuple[str, G.Point]]:
        return [("START", mission.start_xy)] + [(w.id, w.xy) for w in self.ordered()] + [("DEST", self.destination.xy)]

    def to_dict(self) -> dict:
        return {"mission_id": self.mission_id,
                "waypoints": [asdict(w) for w in self.ordered()],
                "destination": asdict(self.destination)}

    @staticmethod
    def from_dict(d: dict) -> "FlightPlan":
        """Strict parse of a stored/internal plan (order fields required)."""
        _keys(d, {"mission_id", "waypoints", "destination"}, set(), "plan")
        return FlightPlan(d["mission_id"], [Waypoint(**w) for w in d["waypoints"]], Waypoint(**d["destination"])).validate()

    @staticmethod
    def from_llm_json(d: Any, mission_id: str) -> "FlightPlan":
        """Strict parse of LLM output: {"waypoints":[{id,x,y,altitude}...], "destination":{x,y,altitude}}.
        Order is the list order (1..n). Unknown fields, strings-as-numbers, missing fields are rejected."""
        _keys(d, {"waypoints", "destination"}, set(), "plan")
        if not isinstance(d["waypoints"], list):
            raise SchemaError("waypoints must be a list")
        wps = []
        for i, w in enumerate(d["waypoints"]):
            _keys(w, {"id", "x", "y", "altitude"}, set(), f"waypoints[{i}]")
            wps.append(Waypoint(w["id"], w["x"], w["y"], w["altitude"], i + 1))
        dst = d["destination"]
        _keys(dst, {"x", "y", "altitude"}, {"id"}, "destination")
        return FlightPlan(mission_id, wps, Waypoint("DEST", dst["x"], dst["y"], dst["altitude"], len(wps) + 1)).validate()


# ---- verification / repair / experiment records ----
@dataclass
class Violation:
    type: str
    family: str  # altitude | no_fly_zone | distance | ordering | mission_preservation
    location: str
    actual: Any
    required: Any
    details: str
    waypoint_id: str | None = None
    segment: list[str] | None = None
    zone_id: str | None = None
    evidence: dict = field(default_factory=dict)
    severity: str = "medium"   # low | medium | high (deterministic rule in verifier)
    hint: str = ""             # generic remedy for this constraint type (never a solution)

    def cid(self) -> tuple:
        """Constraint identity used to follow a violation across repair rounds (stable when detour waypoints are renamed)."""
        if self.type == "NFZ_SEGMENT":
            return (self.type, self.zone_id)
        if self.type == "NFZ_WAYPOINT":
            return (self.type, self.waypoint_id, self.zone_id)
        if self.type == "ORDER_VIOLATION":
            return (self.type, tuple(self.evidence.get("expected", [])))
        if self.family in ("distance", "energy"):
            return (self.type,)
        return (self.type, self.waypoint_id)

    def key(self) -> tuple:
        return (self.type, self.waypoint_id, tuple(self.segment) if self.segment else None, self.zone_id)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    valid: bool                      # all four hard constraints satisfied
    mission_preserved: bool          # all explicit mission requirements retained
    violations: list[Violation]
    preservation_violations: list[Violation]
    route_distance: float
    per_constraint: dict
    verification_time_s: float
    energy: float = 0.0
    flight_time_s: float = 0.0
    ascent: float = 0.0

    @property
    def success(self) -> bool:
        return self.valid and self.mission_preserved

    @property
    def score(self) -> int:
        return len(self.violations) + len(self.preservation_violations)

    def all_violations(self) -> list[Violation]:
        return self.violations + self.preservation_violations

    def to_dict(self) -> dict:
        return {"valid": self.valid, "mission_preserved": self.mission_preserved, "success": self.success,
                "violations": [v.to_dict() for v in self.violations],
                "preservation_violations": [v.to_dict() for v in self.preservation_violations],
                "route_distance": self.route_distance, "per_constraint": self.per_constraint,
                "verification_time_s": self.verification_time_s, "score": self.score,
                "energy": self.energy, "flight_time_s": self.flight_time_s, "ascent": self.ascent}


@dataclass
class RepairIteration:
    iteration: int
    plan: dict | None
    verification: dict | None
    feedback: dict | None
    outcome: str
    score: int | None
    prev_score: int | None
    regression: bool
    new_violations: list
    diff: dict
    calls: list
    error: str | None = None
    base: int | None = None      # index of the earlier iteration whose plan this round repaired (differs from k-1 after a guard rollback)


@dataclass
class ExperimentResult:
    run_id: str
    experiment_id: str
    scenario_id: str
    difficulty: str
    scenario_meta: dict
    condition: str
    condition_config: dict
    model: str
    provider: str
    is_mock: bool
    seed: int
    initial_plan: dict | None
    iterations: list
    final_plan: dict | None
    final_verification: dict | None
    final_status: str
    metrics: dict
    timing: dict

    def to_dict(self) -> dict:
        return asdict(self)

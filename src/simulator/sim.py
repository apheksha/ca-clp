"""Built-in UAV flight simulator (pure Python, no external dependencies): executes a verified plan with a vehicle profile, wind, localization error, a lookahead
path-following controller with first-order velocity response and acceleration limits, battery discharge from the physical power model, and continuous 3D
collision checking against zones and terrain.

SCOPE (read this): a point-mass kinematic model with a simplified autopilot. It reproduces the EFFECTS the verifier ignores (wind drift, tracking error, corner
cutting, speed scheduling, localization error, battery consumption vs. prediction) but it is NOT a substitute for PX4 SITL / Gazebo, which add attitude dynamics,
real estimator/controller code and sensor plugins (see src/simulator/px4_sitl.py and docs/PX4_GAZEBO.md). Parameters are representative, not calibrated.

Units: simulator state in metres/seconds; missions in map units (Mission.unit_m metres per unit); altitudes in metres on the plan's datum."""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from src.models.energy import power_w, hover_power_w
from src.models.schemas import Mission, FlightPlan, NoFlyZone
from src.models.vehicle import VehicleProfile, get_vehicle
from src.simulator.wind import WindModel
from src.simulator.sensors import LocalizationModel, AlphaBetaEstimator
from src.utils import geometry as G


@dataclass
class SimConfig:
    dt: float = 0.1
    max_time_s: float | None = None
    collision_margin_m: float = 1.0       # vehicle radius + safety margin used for zone checks
    ground_contact_m: float = 0.3         # below this height above terrain the vehicle has hit the ground (except near take-off / landing)
    gust_sigma: float = 0.0               # m/s, OU gust standard deviation
    gust_tau_s: float = 8.0
    shear_alpha: float = 0.0
    wind_compensation: float = 0.8        # fraction of the STEADY wind that the autopilot feed-forwards (gusts are never compensated)
    use_estimates: bool = True            # controller uses the estimated (noisy) position; False = perfect localization
    loc_rate_hz: float = 5.0
    record_every: int = 5                 # store every n-th step
    stop_on_collision: bool = True
    seed: int = 0
    # v12 flight-dynamics options (default off = earlier behaviour)
    turn_limit: bool = False              # cap the lateral acceleration at min(max_lateral_accel, V^2 / min_turn_radius_m): an explicit turning-radius limit
    smoothing_radius_m: float = 0.0       # replace every corner at a NON-required waypoint by a circular fillet of this radius (fly-by); required sites stay fly-over
    loc_delay_s: float = 0.0              # state-estimation delay: position fixes describe the vehicle `loc_delay_s` seconds earlier
    compensate_loc_delay: bool = True     # the estimator propagates each delayed fix by v_est * delay (what an autopilot does with a known latency)


@dataclass
class TrajectoryResult:
    t: list = field(default_factory=list)
    pos: list = field(default_factory=list)        # true (x, y, z) metres
    est: list = field(default_factory=list)        # estimated position
    vel: list = field(default_factory=list)        # true ground velocity
    power_w: list = field(default_factory=list)
    energy_wh: list = field(default_factory=list)  # cumulative
    cross_track_m: list = field(default_factory=list)
    status: str = "running"                         # arrived | collision | battery_depleted | timeout
    collision: dict | None = None
    reached: list = field(default_factory=list)    # (waypoint id, time)
    events: list = field(default_factory=list)
    unit_m: float = 1.0
    vehicle: str = ""
    planned_energy_wh: float | None = None

    @property
    def energy_used_wh(self) -> float:
        return self.energy_wh[-1] if self.energy_wh else 0.0

    @property
    def duration_s(self) -> float:
        return self.t[-1] if self.t else 0.0

    def to_dict(self) -> dict:
        ct = self.cross_track_m
        return {"status": self.status, "duration_s": round(self.duration_s, 2), "energy_used_wh": round(self.energy_used_wh, 4), "planned_energy_wh": self.planned_energy_wh,
                "max_cross_track_m": round(max(ct), 3) if ct else None, "mean_cross_track_m": round(sum(ct) / len(ct), 3) if ct else None,
                "collision": self.collision, "reached": self.reached, "n_events": len(self.events), "vehicle": self.vehicle}


def _inflate_zone(z: NoFlyZone, margin_map: float) -> NoFlyZone:
    if margin_map <= 0:
        return z
    if z.kind == "circle":
        return NoFlyZone(z.id, "circle", center=list(z.center), radius=z.radius + margin_map, category=z.category, z_min=z.z_min, z_max=z.z_max)
    from src.benchmark.feasibility import _inflated
    return NoFlyZone(z.id, "polygon", vertices=[list(v) for v in _inflated(z, margin_map)], category=z.category, z_min=z.z_min, z_max=z.z_max)


class FlightSimulator:
    def __init__(self, mission: Mission, plan: FlightPlan, vehicle=None, cfg: SimConfig | None = None, truth_zones_fn=None, wind: WindModel | None = None,
                 loc: LocalizationModel | None = None):
        self.m = mission
        self.cfg = cfg or SimConfig()
        self.veh: VehicleProfile = get_vehicle(vehicle if vehicle is not None else (mission.vehicle or "generic_quad_small"))
        self.unit = mission.unit_m
        self.wind = wind or WindModel(mission.wind or (0.0, 0.0), self.cfg.gust_sigma, self.cfg.gust_tau_s, self.cfg.shear_alpha, seed=self.cfg.seed)
        v = self.veh
        self.loc = loc or LocalizationModel(v.gps_sigma_m, v.gps_bias_sigma_m, v.gps_bias_tau_s, self.cfg.loc_rate_hz, v.gps_dropout_p, self.cfg.seed,
                                            delay_s=self.cfg.loc_delay_s)
        self.truth_zones_fn = truth_zones_fn or (lambda t: mission.no_fly_zones if mission.is_active("no_fly_zone") else [])
        p0 = (mission.start[0] * self.unit, mission.start[1] * self.unit, mission.start_altitude)
        self.p = list(p0)
        self.va = [0.0, 0.0, 0.0]                  # air-relative velocity
        self.v = [0.0, 0.0, 0.0]                   # ground velocity
        self.est = AlphaBetaEstimator(p0, meas_sigma=v.gps_sigma_m + v.gps_bias_sigma_m)
        self.t = 0.0
        self.energy = 0.0
        self.step_i = 0
        self.dwell_until = -1.0
        self.visited: set[str] = set()
        self.res = TrajectoryResult(unit_m=self.unit, vehicle=v.name)
        self._infl: dict = {}
        self._set_path(plan, mission, from_pos=None)
        from src.models.energy import plan_energy
        try:
            self.res.planned_energy_wh = round(plan_energy(mission, plan)["energy_wh"], 4) if mission.energy_model == "physical" else None
        except Exception:
            self.res.planned_energy_wh = None
        self.max_time = self.cfg.max_time_s or self._default_max_time()
        self.done = False

    # ---------------- path handling ----------------
    def _set_path(self, plan: FlightPlan, mission: Mission, from_pos):
        u = self.unit
        nodes = [("START" if from_pos is None else "ESTIMATE", tuple(from_pos) if from_pos else (mission.start[0] * u, mission.start[1] * u, mission.start_altitude))]
        nodes += [(w.id, (w.x * u, w.y * u, w.altitude)) for w in plan.ordered()]
        nodes.append(("DEST", (plan.destination.x * u, plan.destination.y * u, plan.destination.altitude)))
        if self.cfg.smoothing_radius_m > 0:
            nodes = _fillet_path(nodes, self.cfg.smoothing_radius_m, {r.id for r in mission.required_waypoints}, self.res.events, self.t)
        self.path, self.k = nodes, 0
        self.res.events.append({"t": round(self.t, 2), "event": "path_set", "n_waypoints": len(nodes) - 2})
        self.plan_mission = mission

    def replace_plan(self, plan: FlightPlan, mission: Mission | None = None, via=None) -> None:
        """Online re-planning: continue from the current ESTIMATED position along a new plan (waypoints already visited stay visited).
        via: optional 3D point (metres) the new plan was computed to START from (the predicted position at activation). The vehicle first flies to it along its old
        path so the verified first leg of the new plan is the one actually flown; skipped when the vehicle is already past it."""
        self._set_path(plan, mission or self.m, from_pos=tuple(self.est.p))
        if via is not None:
            ahead = sum((via[i] - self.est.p[i]) * self.v[i] for i in range(2)) > 0 or math.hypot(self.v[0], self.v[1]) < 0.5
            if ahead and math.dist(self.est.p, via) > self.veh.acceptance_radius_m:
                self.path.insert(1, ("VIA", tuple(via)))
        self.dwell_until = -1.0

    def _default_max_time(self) -> float:
        L = sum(math.dist(a[1], b[1]) for a, b in zip(self.path, self.path[1:]))
        return 4.0 * L / max(self.veh.v_cruise * 0.5, 1.0) + 300.0

    # ---------------- helpers ----------------
    def _unit_vec(self, a, b):
        d = math.dist(a, b)
        return ((b[0] - a[0]) / d, (b[1] - a[1]) / d, (b[2] - a[2]) / d) if d > 1e-9 else (0.0, 0.0, 0.0)

    def _zones(self):
        out = []
        mm = self.cfg.collision_margin_m / self.unit
        for z in self.truth_zones_fn(self.t):
            key = (z.id, z.kind, mm, tuple(z.center or ()), z.radius, tuple(map(tuple, z.vertices or ())), z.z_min, z.z_max)
            if key not in self._infl:
                self._infl[key] = _inflate_zone(z, mm)
            out.append(self._infl[key])
        return out

    def _corner_speed(self, k: int) -> float:
        v = self.veh
        if k + 2 >= len(self.path):
            return 0.0
        a, b, c = self.path[k][1], self.path[k + 1][1], self.path[k + 2][1]
        u1, u2 = self._unit_vec(a, b), self._unit_vec(b, c)
        cosd = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        theta = math.acos(cosd)
        if v.kind == "fixed_wing":
            return v.v_cruise
        return max(1.0, v.v_cruise * math.cos(theta / 2) ** 2) if theta > 0.09 else v.v_cruise

    # ---------------- one step ----------------
    def step(self) -> bool:
        """Advance one time step. Returns False when the run is finished (see result.status)."""
        if self.done:
            return False
        cfg, v, dt = self.cfg, self.veh, self.cfg.dt
        gnd = self.m.terrain_grid
        z_agl = self.p[2] - (gnd.height_at(self.p[0] / self.unit, self.p[1] / self.unit) if gnd else 0.0)
        self.wind.step(dt)
        w = self.wind.at(z_agl)
        # --- estimation ---
        self.est.predict(dt, self.t)
        z_ = self.loc.measure(self.t, self.p)
        if z_ is not None and cfg.loc_delay_s > 0 and cfg.compensate_loc_delay:   # standard latency compensation: propagate the delayed fix with the velocity estimate
            z_ = tuple(z_[i] + self.est.v[i] * cfg.loc_delay_s for i in range(3))
        self.est.update(z_)
        ep = tuple(self.p) if not cfg.use_estimates else tuple(self.est.p)
        # --- guidance ---
        hovering = self.t < self.dwell_until
        A, B = self.path[self.k][1], self.path[self.k + 1][1]
        AB = math.dist(A, B)
        d_to_B = math.dist(ep, B)
        if hovering:
            vcmd = (0.0, 0.0, 0.0)
            tgt_speed = 0.0
        else:
            if AB > 1e-6:
                u = self._unit_vec(A, B)
                s = (ep[0] - A[0]) * u[0] + (ep[1] - A[1]) * u[1] + (ep[2] - A[2]) * u[2]
                look = max(4.0, 1.2 * math.hypot(self.v[0], self.v[1]))
                tt = min(AB, max(0.0, s) + look)
                target = (A[0] + u[0] * tt, A[1] + u[1] * tt, A[2] + u[2] * tt)
            else:
                target = B
            last = self.k + 1 == len(self.path) - 1
            v_end = 0.0 if (last or v.hover_time_per_waypoint_s > 0) else self._corner_speed(self.k)
            a_dec = max(v.max_accel * 0.7, 0.3)
            tgt_speed = min(v.v_cruise, math.sqrt(max(0.0, v_end * v_end + 2 * a_dec * max(d_to_B - 0.5 * v.acceptance_radius_m, 0.0))))
            if v.kind == "fixed_wing":
                tgt_speed = max(tgt_speed, cfg_stall(v))
            dh = math.hypot(target[0] - ep[0], target[1] - ep[1])
            ux, uy = ((target[0] - ep[0]) / dh, (target[1] - ep[1]) / dh) if dh > 1e-9 else (0.0, 0.0)
            vz_des = max(-v.v_descend_max, min(v.v_climb_max, 0.8 * (target[2] - ep[2])))
            vcmd = (tgt_speed * ux, tgt_speed * uy, vz_des)
        comp = cfg.wind_compensation
        wm = self.wind.steady
        va_cmd = (vcmd[0] - comp * wm[0], vcmd[1] - comp * wm[1], vcmd[2])
        sp = math.hypot(va_cmd[0], va_cmd[1])
        if sp > v.v_max:
            va_cmd = (va_cmd[0] * v.v_max / sp, va_cmd[1] * v.v_max / sp, va_cmd[2])
        gain = v.tracking_gain
        dv = [gain * (va_cmd[i] - self.va[i]) * dt for i in range(3)]
        if cfg.turn_limit:
            V = math.hypot(self.va[0], self.va[1])
            if V > 1.0:
                ux_, uy_ = self.va[0] / V, self.va[1] / V
                along = dv[0] * ux_ + dv[1] * uy_
                px_, py_ = dv[0] - along * ux_, dv[1] - along * uy_
                a_lat = min(v.max_lateral_accel, V * V / max(v.min_turn_radius_m, 1e-6))
                pm = math.hypot(px_, py_)
                if pm > a_lat * dt:
                    k_ = a_lat * dt / pm
                    dv[0], dv[1] = along * ux_ + px_ * k_, along * uy_ + py_ * k_
        mag = math.sqrt(sum(x * x for x in dv))
        lim = v.max_accel * dt
        if mag > lim:
            dv = [x * lim / mag for x in dv]
        self.va = [self.va[i] + dv[i] for i in range(3)]
        self.v = [self.va[0] + w[0], self.va[1] + w[1], self.va[2]]
        prev = tuple(self.p)
        self.p = [self.p[i] + self.v[i] * dt for i in range(3)]
        # --- power / battery ---
        airspeed = math.hypot(self.va[0], self.va[1])
        P = power_w(v, airspeed, self.va[2], self.m.payload_kg) if not hovering else hover_power_w(v, self.m.payload_kg) + v.mass_kg * 0.0
        self.energy += P * dt / 3600.0
        self.t += dt
        self.step_i += 1
        # --- waypoint progress (uses the estimate, like a real autopilot) ---
        if not hovering and d_to_B <= v.acceptance_radius_m:
            nid = self.path[self.k + 1][0]
            self.res.reached.append((nid, round(self.t, 2)))
            self.visited.add(nid)
            if nid == "DEST":
                self._finish("arrived")
            else:
                self.k += 1
                if v.hover_time_per_waypoint_s > 0:
                    self.dwell_until = self.t + v.hover_time_per_waypoint_s
        # --- collision (continuous swept check on the TRUE motion) ---
        col = self._collision(prev, tuple(self.p), z_agl)
        if col and not self.done:
            self.res.collision = col
            if cfg.stop_on_collision:
                self._finish("collision")
        if self.energy >= v.usable_wh and not self.done:
            self._finish("battery_depleted")
        if self.t >= self.max_time and not self.done:
            self._finish("timeout")
        # --- record ---
        if self.step_i % cfg.record_every == 0 or self.done:
            self.res.t.append(round(self.t, 3))
            self.res.pos.append(tuple(self.p))
            self.res.est.append(tuple(self.est.p))
            self.res.vel.append(tuple(self.v))
            self.res.power_w.append(P)
            self.res.energy_wh.append(self.energy)
            self.res.cross_track_m.append(self._cross_track())
        return not self.done

    def _finish(self, status: str):
        self.done = True
        self.res.status = status

    def _cross_track(self) -> float:
        best = float("inf")
        for k in (max(0, self.k - 1), self.k):
            a, b = self.path[k][1], self.path[k + 1][1]
            ab = [b[i] - a[i] for i in range(3)]
            L2 = sum(x * x for x in ab)
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, sum((self.p[i] - a[i]) * ab[i] for i in range(3)) / L2))
            best = min(best, math.dist(self.p, [a[i] + t * ab[i] for i in range(3)]))
        return best

    def _collision(self, p0, p1, z_agl_prev):
        u = self.unit
        a, b = (p0[0] / u, p0[1] / u), (p1[0] / u, p1[1] / u)
        for z in self._zones():
            if z.intersects_segment_3d(a, b, p0[2], p1[2]):
                return {"kind": "zone", "zone": z.id, "t": round(self.t, 2), "position_m": [round(x, 2) for x in p1]}
        g = self.m.terrain_grid
        if g is not None and self.t > 3.0 and math.dist(p1, self.path[-1][1]) > 2 * self.veh.acceptance_radius_m:
            agl = p1[2] - g.height_at(p1[0] / u, p1[1] / u)
            if agl < self.cfg.ground_contact_m:
                return {"kind": "terrain", "t": round(self.t, 2), "position_m": [round(x, 2) for x in p1], "agl_m": round(agl, 2)}
        return None

    # ---------------- convenience ----------------
    def run(self, on_step=None) -> TrajectoryResult:
        while self.step():
            if on_step is not None:
                on_step(self)
        return self.res


def cfg_stall(v: VehicleProfile) -> float:
    return 0.6 * v.v_cruise


def _fillet_path(nodes, R, required_ids, events, t):
    """Fly-by smoothing: each interior corner at a waypoint that is not a required site becomes a circular arc of radius R (metres), sampled every ~10 degrees.
    When a leg is too short for the tangent length R*tan(theta/2), the radius is reduced for that corner (logged as a 'fillet_reduced' event)."""
    out = [nodes[0]]
    for i in range(1, len(nodes) - 1):
        nid, P = nodes[i]
        A, B = out[-1][1], nodes[i + 1][1]
        if nid in required_ids or nid in ("START", "DEST", "ESTIMATE", "VIA"):
            out.append(nodes[i])
            continue
        l1, l2 = math.dist(A[:2], P[:2]), math.dist(P[:2], B[:2])
        if l1 < 1e-6 or l2 < 1e-6:
            out.append(nodes[i])
            continue
        u1 = ((P[0] - A[0]) / l1, (P[1] - A[1]) / l1)
        u2 = ((B[0] - P[0]) / l2, (B[1] - P[1]) / l2)
        cosd = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        theta = math.acos(cosd)
        if theta < math.radians(3):
            out.append(nodes[i])
            continue
        tl = R * math.tan(theta / 2)
        r_eff = R
        lim = 0.45 * min(l1, l2)
        if tl > lim:
            tl, r_eff = lim, lim / math.tan(theta / 2)
            events.append({"t": round(t, 2), "event": "fillet_reduced", "waypoint": nid, "radius_m": round(r_eff, 1)})
        T1 = (P[0] - u1[0] * tl, P[1] - u1[1] * tl)
        cross = u1[0] * u2[1] - u1[1] * u2[0]
        sgn = 1.0 if cross > 0 else -1.0
        n1 = (-u1[1] * sgn, u1[0] * sgn)                       # towards the centre
        C = (T1[0] + n1[0] * r_eff, T1[1] + n1[1] * r_eff)
        a0 = math.atan2(T1[1] - C[1], T1[0] - C[0])
        k = max(2, int(math.degrees(theta) / 10) + 1)
        for j in range(k + 1):
            ang = a0 + sgn * theta * j / k
            out.append((f"{nid}~{j}", (C[0] + r_eff * math.cos(ang), C[1] + r_eff * math.sin(ang), P[2])))
    out.append(nodes[-1])
    return out


def simulate(mission: Mission, plan: FlightPlan, vehicle=None, cfg: SimConfig | None = None, **kw) -> TrajectoryResult:
    return FlightSimulator(mission, plan, vehicle, cfg, **kw).run()

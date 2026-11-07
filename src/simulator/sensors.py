"""Sensor models: localization (GNSS with Gauss-Markov bias, white noise, dropouts) + alpha-beta state estimator, and an obstacle / airspace-event sensor with
detection range, detection probability and latency. These are SIMULATED sensors (statistical models), not drivers for real hardware."""
from __future__ import annotations
import math
import random


class LocalizationModel:
    """GNSS-like position measurement: z = p + b + n; b is a first-order Gauss-Markov bias (tau, sigma_b), n white noise (sigma), updated at `rate_hz`; with probability
    `dropout_p` per update the measurement is missing (the estimator then coasts on its velocity estimate)."""

    def __init__(self, sigma_m: float = 1.2, bias_sigma_m: float = 1.5, bias_tau_s: float = 60.0, rate_hz: float = 5.0, dropout_p: float = 0.0, seed: int = 0,
                 delay_s: float = 0.0):
        self.sigma, self.bsig, self.tau, self.period, self.p_drop = sigma_m, bias_sigma_m, max(bias_tau_s, 1e-3), 1.0 / rate_hz, dropout_p
        self.rng = random.Random(f"loc|{seed}")
        self.b = [self.rng.gauss(0, bias_sigma_m) for _ in range(3)]
        self._t_next = 0.0
        self.delay = max(0.0, delay_s)     # v12: state-estimation delay - every fix describes where the vehicle was `delay_s` earlier
        self._hist: list = []

    def measure(self, t: float, pos):
        """Return a measurement (x, y, z) when an update is due and not dropped, else None."""
        if self.delay > 0:
            self._hist.append((t, tuple(pos)))
            while len(self._hist) > 2 and self._hist[1][0] <= t - self.delay:
                self._hist.pop(0)
            pos = self._hist[0][1]          # newest sample at or before t - delay (the oldest one while less than `delay` has passed)
        if t + 1e-12 < self._t_next:
            return None
        dt = self.period
        self._t_next = t + self.period
        a = math.exp(-dt / self.tau)
        s = self.bsig * math.sqrt(1 - a * a)
        self.b = [a * b + s * self.rng.gauss(0, 1) for b in self.b]
        if self.rng.random() < self.p_drop:
            return None
        return tuple(pos[i] + self.b[i] + self.rng.gauss(0, self.sigma) for i in range(3))


class AlphaBetaEstimator:
    """Position/velocity estimator: predict with the velocity estimate, correct with measurements (gains alpha, beta). `sigma()` is a rough 1-sigma position
    uncertainty that grows while no measurement arrives."""

    def __init__(self, p0, alpha: float = 0.35, beta: float = 0.05, meas_sigma: float = 1.5):
        self.p = list(p0)
        self.v = [0.0, 0.0, 0.0]
        self.alpha, self.beta, self.meas_sigma = alpha, beta, meas_sigma
        self.t_last_meas = None      # time of the last measurement (None until the first one: the first measurement only corrects the position)
        self._t = 0.0

    def predict(self, dt: float, t: float) -> None:
        self._t = t
        for i in range(3):
            self.p[i] += self.v[i] * dt

    def update(self, z) -> None:
        if z is None:
            return
        dt = None if self.t_last_meas is None else max(self._t - self.t_last_meas, 0.02)
        for i in range(3):
            r = z[i] - self.p[i]
            self.p[i] += self.alpha * r
            if dt is not None:
                self.v[i] += self.beta * r / dt
        self.t_last_meas = self._t

    def sigma(self) -> float:
        return self.meas_sigma + 0.5 * max(0.0, self._t - (self.t_last_meas if self.t_last_meas is not None else 0.0))


class ZoneEventSensor:
    """Detects restricted-zone events that are NOT in the UAV's current map.
    * onboard: a zone is 'seen' when the ESTIMATED position is within `range_m` of its (metre-space) boundary and the zone exists; detection is reported after
      `onboard_delay_s` (processing latency) with probability `p_detect` per second in range.
    * broadcast: an externally announced zone (NOTAM / UTM message) is received `broadcast_latency_s` after it appears, independent of position.
    Whichever comes first defines the detection time of the event."""

    def __init__(self, range_m: float = 60.0, onboard_delay_s: float = 1.0, p_detect: float = 0.9, broadcast_latency_s: float | None = None, seed: int = 0):
        self.range, self.delay, self.p, self.bcast = range_m, onboard_delay_s, p_detect, broadcast_latency_s
        self.rng = random.Random(f"zsens|{seed}")
        self._first_seen: dict[str, float] = {}
        self.known: set[str] = set()

    def poll(self, t: float, est_pos_m, zones_truth: list, appear_t: dict[str, float], unit_m: float, dt: float) -> list[str]:
        """Return ids of zones newly DETECTED at time t. `zones_truth` are NoFlyZone objects (map units) currently existing; `appear_t` maps zone id -> appearance time."""
        new = []
        for z in zones_truth:
            if z.id in self.known:
                continue
            ta = appear_t.get(z.id, 0.0)
            if self.bcast is not None and t >= ta + self.bcast:
                self.known.add(z.id)
                new.append(z.id)
                continue
            d = _dist_to_zone_m(z, est_pos_m, unit_m)
            if d <= self.range:
                self._first_seen.setdefault(z.id, t)
                if t >= self._first_seen[z.id] + self.delay and self.rng.random() < min(1.0, self.p * dt + (1.0 if t >= self._first_seen[z.id] + self.delay + 2.0 else 0.0)):
                    self.known.add(z.id)
                    new.append(z.id)
        return new


def _dist_to_zone_m(z, p_m, unit_m: float) -> float:
    from src.utils import geometry as G
    p = (p_m[0] / unit_m, p_m[1] / unit_m)
    if z.kind == "circle":
        return max(0.0, G.dist(p, z.centroid()) - z.radius) * unit_m
    return 0.0 if z.contains(p) else G.min_dist_to_polygon_boundary(p, z._verts()) * unit_m

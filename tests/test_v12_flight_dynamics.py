"""v12.1: flight-dynamics options of the simulator (turning-radius limit, fillet smoothing, state-estimation delay) and the smoothed-plan re-verification
used by scripts/flight_dynamics_study.py. Defaults must keep the earlier behaviour."""
import dataclasses
import importlib.util
import math
import os
import unittest

from tests.helpers import mk_mission, mk_plan
from src.models.schemas import Mission
from src.models.vehicle import get_vehicle
from src.simulator.batch import prepare_mission
from src.simulator.sensors import LocalizationModel
from src.simulator.sim import SimConfig, simulate, _fillet_path
from src.verifier.verifier import verify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def corner_mission():
    """Start (0,0) -> required A (100,0) -> corner C (100,100), not required -> destination (0,100); 10 m per unit."""
    d = mk_mission(destination=[0, 100], required_waypoints=[], ordering=[], max_distance=400, min_altitude=40.0, max_altitude=100.0,
                   start_altitude=60.0).to_dict()
    return Mission.from_dict(d)


class Fillet(unittest.TestCase):
    def test_right_angle_arc_is_tangent_and_has_radius_r(self):
        ev = []
        nodes = [("START", (0.0, 0.0, 50.0)), ("W1", (100.0, 0.0, 50.0)), ("DEST", (100.0, 100.0, 50.0))]
        out = _fillet_path(nodes, 20.0, set(), ev, 0.0)
        arc = [p for nid, p in out if nid.startswith("W1~")]
        self.assertGreaterEqual(len(arc), 3)
        for x, y, h in arc:                                          # left turn: centre (80, 20)
            self.assertAlmostEqual(math.hypot(x - 80.0, y - 20.0), 20.0, places=6)
            self.assertEqual(h, 50.0)
        self.assertAlmostEqual(arc[0][0], 80.0, places=6)
        self.assertAlmostEqual(arc[0][1], 0.0, places=6)
        self.assertAlmostEqual(arc[-1][0], 100.0, places=6)
        self.assertAlmostEqual(arc[-1][1], 20.0, places=6)
        self.assertEqual(ev, [])

    def test_required_site_stays_fly_over(self):
        nodes = [("START", (0.0, 0.0, 50.0)), ("A", (100.0, 0.0, 50.0)), ("DEST", (100.0, 100.0, 50.0))]
        out = _fillet_path(nodes, 20.0, {"A"}, [], 0.0)
        self.assertEqual([nid for nid, _ in out], ["START", "A", "DEST"])

    def test_short_leg_reduces_the_radius_and_logs_it(self):
        ev = []
        nodes = [("START", (0.0, 0.0, 50.0)), ("W1", (10.0, 0.0, 50.0)), ("DEST", (10.0, 100.0, 50.0))]
        out = _fillet_path(nodes, 40.0, set(), ev, 0.0)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["event"], "fillet_reduced")
        self.assertLess(ev[0]["radius_m"], 40.0)
        self.assertGreaterEqual(min(x for _, (x, _, _) in out), 0.0)    # never starts the arc before the previous node


class TurnLimitAndDelay(unittest.TestCase):
    def setUp(self):
        self.m = prepare_mission(corner_mission(), "generic_quad_small", 10.0)
        self.plan = mk_plan([("C", 100, 0, 60), ("D", 100, 100, 60)], dest=(0, 100, 60))

    def test_defaults_are_unchanged(self):
        c = SimConfig()
        self.assertFalse(c.turn_limit)
        self.assertEqual(c.smoothing_radius_m, 0.0)
        self.assertEqual(c.loc_delay_s, 0.0)

    def test_turn_limit_bounds_lateral_acceleration(self):
        veh = dataclasses.replace(get_vehicle("generic_quad_small"), max_lateral_accel=0.8)
        dt = 0.2
        tr = simulate(self.m, self.plan, vehicle=veh, cfg=SimConfig(dt=dt, record_every=1, use_estimates=False, turn_limit=True))
        self.assertEqual(tr.status, "arrived")
        worst = 0.0
        for v0, v1 in zip(tr.vel, tr.vel[1:]):
            V = math.hypot(v0[0], v0[1])
            if V > 2.0:
                dvx, dvy = v1[0] - v0[0], v1[1] - v0[1]
                worst = max(worst, abs(v0[0] * dvy - v0[1] * dvx) / V / dt)
        self.assertLessEqual(worst, 0.8 * 1.05)
        free = simulate(self.m, self.plan, vehicle=veh, cfg=SimConfig(dt=dt, record_every=1, use_estimates=False))
        self.assertGreater(max(tr.cross_track_m), max(free.cross_track_m))     # a wider turn deviates more from the planned corner

    def test_smoothing_cuts_the_corner(self):
        tr = simulate(self.m, self.plan, cfg=SimConfig(dt=0.2, record_every=1, use_estimates=False, smoothing_radius_m=150.0))
        self.assertEqual(tr.status, "arrived")
        corner = (1000.0, 0.0)                                          # waypoint C in metres; the fillet keeps the vehicle away from it
        self.assertGreater(min(math.dist(p[:2], corner) for p in tr.pos), 30.0)

    def test_localization_delay_reports_the_old_position(self):
        loc = LocalizationModel(sigma_m=0.0, bias_sigma_m=0.0, rate_hz=10.0, delay_s=0.5)
        z = None
        for k in range(21):
            t = k * 0.1
            z = loc.measure(t, (t, 0.0, 0.0)) or z
        self.assertAlmostEqual(z[0], 1.5, places=6)                     # at t = 2.0 s the fix describes t = 1.5 s
        nodelay = LocalizationModel(sigma_m=0.0, bias_sigma_m=0.0, rate_hz=10.0)
        self.assertAlmostEqual(nodelay.measure(2.0, (2.0, 0.0, 0.0))[0], 2.0, places=6)

    def test_compensated_delay_still_arrives(self):
        tr = simulate(self.m, self.plan, cfg=SimConfig(dt=0.1, loc_delay_s=0.5, compensate_loc_delay=True))
        self.assertEqual(tr.status, "arrived")


class SmoothedPlanReverification(unittest.TestCase):
    def test_smoothed_plan_of_the_study_verifies(self):
        spec = importlib.util.spec_from_file_location("fds", os.path.join(ROOT, "scripts", "flight_dynamics_study.py"))
        fds = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fds)
        m = corner_mission()
        plan = mk_plan([("C", 100, 0, 60), ("D", 100, 100, 60)], dest=(0, 100, 60))
        self.assertTrue(verify(m, plan).success)
        sp, n_red = fds.smoothed_plan(m, plan, 5.0)
        self.assertGreater(len(sp.waypoints), len(plan.waypoints))
        self.assertEqual(n_red, 0)
        self.assertTrue(verify(m, sp).success)


if __name__ == "__main__":
    unittest.main()

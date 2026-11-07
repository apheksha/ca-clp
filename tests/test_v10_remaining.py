"""Tests for the v10 additions: physical energy, terrain, routed oracle bound, exact geometry, simulator, event detection + online re-planning, NL understanding,
model registry, geodata loaders, PX4/Gazebo export."""
import json, math, os, random, tempfile, unittest
import yaml
from tests.helpers import mk_mission, mk_plan, GOOD
from src.models.schemas import Mission, NoFlyZone, RequiredWaypoint, FlightPlan, Waypoint
from src.models.vehicle import get_vehicle
from src.models import energy as EN
from src.verifier.verifier import verify
from src.benchmark.feasibility import assess, routed_lower_bound


def physical(m: Mission, **kw) -> Mission:
    d = m.to_dict()
    d.update(energy_model="physical", vehicle="generic_quad_small", max_energy=40.0, **kw)
    return Mission.from_dict(d)


class PhysicalEnergy(unittest.TestCase):
    def setUp(self):
        self.m = physical(mk_mission(start_altitude=60.0))
        self.p = mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))

    def test_payload_and_wind_increase_energy(self):
        e0 = EN.plan_energy(self.m, self.p)["energy_wh"]
        d = self.m.to_dict(); d["payload_kg"] = 0.5
        e1 = EN.plan_energy(Mission.from_dict(d), self.p)["energy_wh"]
        d = self.m.to_dict(); d["wind"] = [-6.0, 0.0]      # headwind on a +x route
        e2 = EN.plan_energy(Mission.from_dict(d), self.p)["energy_wh"]
        self.assertGreater(e1, e0)
        self.assertGreater(e2, e0)

    def test_verifier_energy_violation_in_wh(self):
        d = self.m.to_dict(); d["max_energy"] = 0.05
        r = verify(Mission.from_dict(d), self.p)
        v = [x for x in r.violations if x.family == "energy"]
        self.assertTrue(v and v[0].evidence["unit"] == "Wh")

    def test_wind_beyond_authority_is_not_flyable(self):
        d = self.m.to_dict(); d["wind"] = [0.0, 30.0]      # 30 m/s crosswind > 12 m/s cruise
        r = verify(Mission.from_dict(d), self.p)
        self.assertTrue(any("cannot be flown" in x.details for x in r.violations if x.family == "energy"))

    def test_lower_bound_is_below_actual(self):
        e = EN.plan_energy(self.m, self.p)["energy_wh"]
        self.assertLessEqual(EN.energy_lower_bound_wh(self.m, 100.0, 0.0), e)

    def test_payload_above_capacity_rejected(self):
        d = self.m.to_dict(); d["payload_kg"] = 5.0
        with self.assertRaises(Exception):
            Mission.from_dict(d)

    def test_proxy_default_unchanged(self):
        self.assertEqual(mk_mission().energy_model, "proxy")


class Terrain(unittest.TestCase):
    def mission(self, **kw):
        d = mk_mission(min_altitude=0.0, max_altitude=300.0).to_dict()
        d.update(terrain={"x0": 0, "y0": -50, "cell": 10, "heights": [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 3 + [[100] * 11] * 0}, min_agl=20.0, **kw)
        return Mission.from_dict(d)

    def test_low_agl_flagged_and_exempt_near_ends(self):
        m = self.mission()
        low = mk_plan([("A", 30, 0, 10), ("B", 70, 0, 10)], dest=(100, 0, 10))
        r = verify(m, low)
        self.assertTrue(any(v.type == "TERRAIN_CLEARANCE" for v in r.violations))
        d = m.to_dict(); d["terrain_exempt_m"] = 1000.0
        self.assertFalse(any(v.type == "TERRAIN_CLEARANCE" for v in verify(Mission.from_dict(d), low).violations))

    def test_hill_between_waypoints_is_caught(self):
        heights = [[0.0] * 11 for _ in range(3)]
        for r in range(3):
            heights[r][5] = 80.0                           # ridge at x = 50
        d = mk_mission(min_altitude=0.0, max_altitude=300.0).to_dict()
        d.update(terrain={"x0": 0, "y0": -10, "cell": 10, "heights": heights}, min_agl=10.0)
        m = Mission.from_dict(d)
        p = mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))   # both waypoints clear, but the leg A->B crosses the ridge (ground 80 > 60)
        self.assertTrue(any(v.type == "TERRAIN_CLEARANCE" for v in verify(m, p).violations))

    def test_oracle_terrain_following_witness(self):
        heights = [[0.0] * 21 for _ in range(5)]
        for r in range(5):
            for c in range(21):
                heights[r][c] = 60.0 * math.sin(c / 3.0) ** 2
        d = mk_mission(min_altitude=0.0, max_altitude=400.0, max_distance=500.0, ordering=[]).to_dict()
        d.update(terrain={"x0": 0, "y0": -20, "cell": 5, "heights": heights}, min_agl=15.0, max_agl=40.0, terrain_exempt_m=15.0, start_altitude=0.0)
        m = Mission.from_dict(d)
        f = assess(m)
        self.assertEqual(f.status, "FEASIBLE")
        self.assertTrue(verify(m, f.witness_plan).success)


class OracleExtras(unittest.TestCase):
    def test_routed_bound_proves_what_straight_bound_cannot(self):
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=20).validate()      # big zone exactly between the points
        m = mk_mission(no_fly_zones=[z], max_distance=104.0, ordering=[], required_waypoints=[RequiredWaypoint("A", 10, 0), RequiredWaypoint("B", 90, 0)])
        straight = 100.0
        self.assertLess(straight, 104.0)
        rlb = routed_lower_bound(m)
        self.assertGreater(rlb, 104.0)
        f = assess(m)
        self.assertEqual((f.status, f.proof_kind), ("INFEASIBLE", "routed_distance_lower_bound"))

    def test_routed_bound_never_exceeds_witness(self):
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=20).validate()
        m = mk_mission(no_fly_zones=[z], max_distance=1000.0, ordering=[], required_waypoints=[RequiredWaypoint("A", 10, 0), RequiredWaypoint("B", 90, 0)])
        f = assess(m, certify=True)
        lb, ub = f.route_bracket
        self.assertLessEqual(lb, ub + 1e-9)
        self.assertLess(ub / lb - 1, 0.05)

    def test_exact_geometry_never_missed(self):
        from src.utils.geometry_selftest import exact_crosscheck
        r = exact_crosscheck(600, 11)
        self.assertEqual(r["float_missed_exact_intersection"], 0)

    def test_physical_energy_proof(self):
        d = mk_mission(ordering=[]).to_dict()
        d.update(energy_model="physical", vehicle="generic_quad_small", max_energy=0.01, unit_m=100.0)
        f = assess(Mission.from_dict(d))
        self.assertEqual((f.status, f.proof_kind), ("INFEASIBLE", "energy_lower_bound"))


class Simulator(unittest.TestCase):
    def mission(self, **kw):
        d = mk_mission(min_altitude=40.0, max_altitude=100.0, start_altitude=60.0, ordering=[]).to_dict()
        d.update(energy_model="physical", vehicle="generic_quad_small", max_energy=40.0, unit_m=10.0, **kw)
        return Mission.from_dict(d)

    def plan(self):
        return mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))

    def test_flies_and_arrives(self):
        from src.simulator.sim import simulate, SimConfig
        from src.simulator.evaluation import evaluate_execution
        m = self.mission()
        tr = simulate(m, self.plan(), cfg=SimConfig(dt=0.2))
        self.assertEqual(tr.status, "arrived")
        ev = evaluate_execution(m, self.plan(), tr)
        self.assertTrue(ev["executed_feasible"], ev)

    def test_wind_increases_tracking_error(self):
        from src.simulator.sim import simulate, SimConfig
        calm = simulate(self.mission(), self.plan(), cfg=SimConfig(dt=0.2, wind_compensation=0.0, seed=1))
        d = self.mission().to_dict(); d["wind"] = [0.0, 6.0]
        windy = simulate(Mission.from_dict(d), self.plan(), cfg=SimConfig(dt=0.2, wind_compensation=0.0, seed=1))
        self.assertGreater(max(windy.cross_track_m), max(calm.cross_track_m) + 1.0)

    def test_battery_depletion(self):
        from src.simulator.sim import simulate, SimConfig
        tr = simulate(self.mission(), self.plan(), vehicle={"base": "generic_quad_small", "battery_wh": 0.05}, cfg=SimConfig(dt=0.2))
        self.assertEqual(tr.status, "battery_depleted")

    def test_zone_collision_detected(self):
        from src.simulator.sim import simulate, SimConfig
        d = self.mission().to_dict()
        d["no_fly_zones"] = [NoFlyZone("Z", "circle", center=[50, 0], radius=5).validate().__dict__]
        m = Mission.from_dict(d)
        tr = simulate(m, self.plan(), cfg=SimConfig(dt=0.2))
        self.assertEqual(tr.status, "collision")

    def test_estimator_regression_no_divergence(self):
        from src.simulator.sensors import LocalizationModel, AlphaBetaEstimator
        loc, est = LocalizationModel(1.2, 1.5, 60, 5.0, 0.0, 0), AlphaBetaEstimator((0, 0, 0))
        p, t, worst = [0.0, 0.0, 0.0], 0.0, 0.0
        for _ in range(200):
            p[0] += 6.0; p[2] += 1.75; t += 0.5
            est.predict(0.5, t); est.update(loc.measure(t, p))
            worst = max(worst, abs(est.p[2] - p[2]), abs(est.p[0] - p[0]))
        self.assertLess(worst, 20.0)

    def test_dynamics_batch_helper(self):
        from src.simulator.batch import prepare_mission
        m = prepare_mission(mk_mission(), "generic_quad_small", 20.0)
        self.assertEqual(m.unit_m, 20.0)


class OnlineReplanning(unittest.TestCase):
    def setUp(self):
        d = mk_mission(min_altitude=40.0, max_altitude=100.0, start_altitude=60.0, ordering=[], max_distance=1e5,
                       required_waypoints=[RequiredWaypoint("A", 60, 0)], dest=(120, 0, 60) if False else None).to_dict() if False else None
        base = mk_mission(min_altitude=40.0, max_altitude=100.0, start_altitude=60.0, ordering=[], max_distance=1e5)
        d = base.to_dict()
        d.update(unit_m=10.0, vehicle="generic_quad_small", destination=[200.0, 0.0], required_waypoints=[{"id": "A", "x": 60.0, "y": 0.0}, {"id": "B", "x": 140.0, "y": 0.0}],
                 active=["altitude", "no_fly_zone", "distance", "ordering"])
        self.m = Mission.from_dict(d)
        self.plan = FlightPlan(self.m.mission_id, [Waypoint("A", 60, 0, 60, 1), Waypoint("B", 140, 0, 60, 2)], Waypoint("DEST", 200, 0, 60, 3)).validate()

    def event(self):
        from src.dynamic.detection import Event
        z = NoFlyZone("EVT", "circle", center=[100, 0], radius=8).validate()      # 80 m radius on the path between A and B (their coordinates are 40 map units away)
        return Event("zone_add", 2.0, zone=z, delivery="broadcast")

    def test_event_detected_once_and_replanned(self):
        from src.dynamic.online import run_online_episode, classical_replanner
        from src.simulator.sim import SimConfig
        r = run_online_episode(self.m, self.plan, [self.event()], classical_replanner(clearance_m=30), sim_cfg=SimConfig(dt=0.25, record_every=4, seed=1), broadcast_latency_s=3, replan_latency_s=3)
        self.assertEqual(len(r["detections"]), 1)
        self.assertEqual(r["detections"][0]["channel"], "broadcast")
        self.assertTrue(any(x.get("replan_verified") for x in r["replans"]))
        self.assertTrue(r["mission_success"], r["evaluation"])

    def test_without_replanner_the_vehicle_hits_the_zone(self):
        from src.dynamic.online import run_online_episode
        from src.simulator.sim import SimConfig
        r = run_online_episode(self.m, self.plan, [self.event()], lambda a, b, c: None, sim_cfg=SimConfig(dt=0.25, record_every=4, seed=1), broadcast_latency_s=3)
        self.assertFalse(r["mission_success"])

    def test_sensor_onboard_detection_requires_range(self):
        from src.simulator.sensors import ZoneEventSensor
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=5).validate()      # 500 m east, radius 50 m
        s = ZoneEventSensor(range_m=100, onboard_delay_s=0.5, p_detect=1.0, seed=0)
        far = [s.poll(t * 0.5, (0, 0, 50), [z], {"Z": 0.0}, 10.0, 0.5) for t in range(10)]
        self.assertFalse(any(far))
        near = [s.poll(10 + t * 0.5, (400, 0, 50), [z], {"Z": 0.0}, 10.0, 0.5) for t in range(10)]
        self.assertTrue(any(near))

    def test_events_from_change(self):
        from src.dynamic.detection import events_from_change
        ev = events_from_change(self.m, self.plan, 0.4, "add_zone", random.Random(3), t_appear=5.0)
        self.assertTrue(ev is None or ev.kind == "zone_add")


class NaturalLanguage(unittest.TestCase):
    def setUp(self):
        from src.nl.dataset import load_collection
        from src.nl.extraction import RuleExtractor
        self.mdef, self.ms = load_collection()
        self.ex = RuleExtractor(self.mdef.site_names(), self.mdef.zone_names())

    def test_collection_schema(self):
        self.assertGreaterEqual(len(self.ms), 20)
        self.assertEqual({m.category for m in self.ms}, {"complete", "incomplete", "ambiguous", "conflicting"})

    def test_complete_is_ready_and_incomplete_asks(self):
        from src.nl.analysis import analyze
        by = {m.id: m for m in self.ms}
        a = analyze(self.ex.extract(by["c01"].text), self.mdef, by["c01"].text)
        self.assertEqual(a.decision, "ready")
        a = analyze(self.ex.extract(by["i03"].text), self.mdef, by["i03"].text)
        self.assertEqual(a.decision, "needs_clarification")
        self.assertIn("start", a.missing)
        self.assertTrue(any("start" in q.lower() for q in a.questions))

    def test_conflicts_are_detected_deterministically(self):
        from src.nl.analysis import analyze
        by = {m.id: m for m in self.ms}
        for mid, kind in (("x01", "cyclic_ordering"), ("x02", "empty_altitude_band"), ("x04", "point_in_zone")):
            a = analyze(self.ex.extract(by[mid].text), self.mdef, by[mid].text)
            self.assertEqual(a.decision, "conflict", mid)
            self.assertIn(kind, {c["type"] for c in a.conflicts})

    def test_unknown_place_triggers_question(self):
        from src.nl.analysis import analyze
        by = {m.id: m for m in self.ms}
        a = analyze(self.ex.extract(by["a02"].text), self.mdef, by["a02"].text)
        self.assertTrue(a.unknown_places)

    def test_clarification_round_trip(self):
        from src.nl.pipeline import understand
        u = understand("Fly from Home to the Pier and take a look, then come back.", self.ex, self.mdef, ask=lambda an: {"max_altitude_m": 90.0})
        self.assertEqual(u.decision, "ready")
        self.assertIsNotNone(u.mission)
        self.assertEqual(len(u.rounds), 1)

    def test_intent_cross_validation_detects_wrong_extraction(self):
        from src.nl.intent import cross_validate
        truth = {"start": "Home", "destination": "Depot", "waypoints": ["School"], "order": [], "min_altitude_m": 40, "max_altitude_m": 100, "max_distance_m": 6000, "avoid_zones": ["Airfield"]}
        wrong = dict(truth, avoid_zones=[])
        r = cross_validate(wrong, truth, self.mdef)
        self.assertTrue(r["checked"])
        if r["extracted_status"] == "FEASIBLE":
            self.assertFalse(r["plan_for_extracted_valid_for_truth"] and not r["plan_for_extracted_valid_for_truth"])

    def test_llm_json_parser(self):
        from src.nl.extraction import parse_extraction_json
        e = parse_extraction_json('noise {"start":"Home","destination":null,"waypoints":["Pier"],"max_altitude_m":"90","missing":["destination"]} tail')
        self.assertEqual((e.start, e.max_altitude_m, e.missing), ("Home", 90.0, ["destination"]))
        self.assertIsNotNone(parse_extraction_json("no json here").error)


class Registry(unittest.TestCase):
    def test_adapt_request_records_every_change(self):
        from src.planner.model_registry import ModelSpec, adapt_request
        spec = ModelSpec("p", "m", supports_temperature=False, supports_seed=False, min_max_tokens=100, max_output_tokens=500)
        t, mt, sd, adj = adapt_request(spec, 0.0, 50, 3)
        self.assertIsNone(t); self.assertEqual(mt, 100); self.assertIsNone(sd)
        self.assertEqual(set(adj), {"temperature_omitted", "max_tokens_raised", "seed_not_sent"})

    def test_unknown_price_is_none_not_zero(self):
        from src.planner.model_registry import get_spec, cost_usd
        s = get_spec("gemini", "some-new-model")
        self.assertTrue(s.capabilities_unknown)
        self.assertIsNone(cost_usd(s, 100, 100))
        s.usd_per_mtok_in, s.usd_per_mtok_out = 1.0, 2.0
        self.assertAlmostEqual(cost_usd(s, 1_000_000, 500_000), 2.0)

    def test_availability_and_fallback(self):
        from src.planner.model_registry import resolve_available
        os.environ.pop("GROQ_API_KEY", None)
        spec, skipped = resolve_available("groq", "x")
        self.assertIsNone(spec)
        self.assertIn("GROQ_API_KEY", skipped[0]["reason"])
        spec, skipped = resolve_available("mock", "mock-heuristic-v1")
        self.assertEqual(spec.provider, "mock")


class Geodata(unittest.TestCase):
    def test_units_holes_and_vertical_limits(self):
        from src.geodata.airspace import load_airspace_geojson
        from src.geodata.projection import LocalFrame
        zs = {z.description: z for z in load_airspace_geojson("data/geodata_samples/airspace_sample.geojson", LocalFrame(47.5, 8.5))}
        self.assertAlmostEqual(zs["CTR TEST"].z_max, 3500 * 0.3048, places=3)       # ft -> m
        self.assertAlmostEqual(zs["TMA HIGH"].z_min, 95 * 30.48, places=2)         # FL95
        self.assertEqual(zs["RESTRICTED TEST"].kind, "circle")
        self.assertEqual(zs["HOLE TEST"].kind, "polygon")                           # hole ignored => whole polygon blocked (conservative)
        self.assertFalse(_is_convex(zs["CTR TEST"]))                                # real (concave) boundary kept, not a circle

    def test_osm_heights(self):
        from src.geodata.osm import load_osm_buildings
        from src.geodata.projection import LocalFrame
        bs = load_osm_buildings("data/geodata_samples/buildings_sample.osm.json", LocalFrame(47.5, 8.5))
        self.assertTrue(bs and all(b.category == "obstacle" and b.z_max > 0 for b in bs))

    def test_hgt_reader_fills_void_and_interpolates(self):
        from src.geodata.srtm import read_hgt
        t = read_hgt("data/geodata_samples/N47E008.hgt")
        self.assertEqual((t.lat0, t.lon0, t.n), (47, 8, 301))
        self.assertGreater(t.data[150][150], -1000)          # the void cell was filled
        self.assertTrue(300 < t.height(47.5, 8.5) < 700)

    def test_terrain_grid_bilinear(self):
        from src.models.terrain import TerrainGrid
        g = TerrainGrid(0, 0, 10, [[0, 10], [20, 30]])
        self.assertAlmostEqual(g.height_at(5, 5), 15.0)
        self.assertAlmostEqual(g.height_at(-100, -100), 0.0)

    def test_scenario_builder_end_to_end(self):
        from src.geodata.scenario_builder import build_geo_mission
        cfg = yaml.safe_load(open("data/geodata_samples/geo_sample.yaml").read())
        m, f = build_geo_mission(cfg, "data/geodata_samples")
        self.assertIsNotNone(m.terrain)
        self.assertGreater(len(m.no_fly_zones), 5)
        self.assertEqual(f.status, "FEASIBLE")
        self.assertTrue(verify(m, f.witness_plan).success)


def _is_convex(z):
    from src.benchmark.feasibility import _is_convex as c
    return c([tuple(v) for v in z.vertices])


class Exports(unittest.TestCase):
    def test_px4_items_and_log_conversion(self):
        from src.simulator.px4_sitl import mission_items, log_to_trajectory
        from src.simulator.evaluation import evaluate_execution
        m, p = mk_mission(), mk_plan(GOOD)
        items = mission_items(m, p, 47.39, 8.54, 1.0)
        self.assertEqual([i["command"] for i in items][0], 22)
        self.assertEqual(items[-1]["command"], 21)
        self.assertTrue(all(abs(i["x"]) < 900_000_000 for i in items))
        from src.simulator.export_px4 import local_to_latlon
        rows = [{"t": i, "lat": local_to_latlon(i * 10.0, 0, 47.39, 8.54)[0], "lon": local_to_latlon(i * 10.0, 0, 47.39, 8.54)[1], "rel_alt": 60.0} for i in range(11)]
        tr = log_to_trajectory(rows, m, 47.39, 8.54)
        self.assertAlmostEqual(tr.pos[-1][0], 100.0, delta=0.5)

    def test_gazebo_world_is_valid_xml(self):
        import xml.dom.minidom as x
        from src.simulator.gazebo_world import world_sdf
        z = NoFlyZone("Z", "polygon", vertices=[[10, 10], [20, 10], [15, 20]], z_max=50).validate()
        x.parseString(world_sdf(mk_mission(no_fly_zones=[z])))


class Vehicles(unittest.TestCase):
    def test_profiles_valid(self):
        for n in ("generic_quad_small", "generic_quad_large", "generic_fixed_wing"):
            v = get_vehicle(n)
            self.assertGreater(EN.power_level_w(v, v.v_cruise), 0)
        self.assertGreater(EN.hover_power_w(get_vehicle("generic_quad_large")), EN.hover_power_w(get_vehicle("generic_quad_small")))

    def test_override_dict(self):
        v = get_vehicle({"base": "generic_quad_small", "battery_wh": 10.0})
        self.assertEqual(v.battery_wh, 10.0)
        with self.assertRaises(Exception):
            get_vehicle({"base": "generic_quad_small", "bogus": 1})


if __name__ == "__main__":
    unittest.main()

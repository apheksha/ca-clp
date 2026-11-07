import unittest
from src.verifier.verifier import verify
from src.models.schemas import NoFlyZone, SchemaError, FlightPlan, Waypoint
from tests.helpers import mk_mission, mk_plan, GOOD


def types(r):
    return sorted(v.type for v in r.violations)


class TestVerifier(unittest.TestCase):
    def test_valid(self):
        r = verify(mk_mission(), mk_plan(GOOD))
        self.assertTrue(r.valid and r.success)
        self.assertAlmostEqual(r.route_distance, 100.0)

    def test_altitude_exact_limits_ok_and_violations(self):
        m = mk_mission()
        self.assertTrue(verify(m, mk_plan([("A", 30, 0, 40), ("B", 70, 0, 80)])).valid)
        r = verify(m, mk_plan([("A", 30, 0, 39.9), ("B", 70, 0, 80.1)]))
        self.assertEqual(types(r), ["ALTITUDE_ABOVE_MAX", "ALTITUDE_BELOW_MIN"])
        self.assertEqual({v.waypoint_id for v in r.violations}, {"A", "B"})
        r = verify(m, mk_plan(GOOD, dest=(100, 0, 200)))
        self.assertEqual(r.violations[0].waypoint_id, "DEST")

    def test_waypoint_in_circle_and_boundary(self):
        z = NoFlyZone("Z1", "circle", center=[30, 10], radius=10)
        m = mk_mission(no_fly_zones=[z])
        r = verify(m, mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)]))  # A exactly on boundary
        self.assertIn("NFZ_WAYPOINT", types(r))
        self.assertEqual(r.violations[0].zone_id, "Z1")

    def test_segment_crossing_with_legal_endpoints(self):
        z = NoFlyZone("Z1", "circle", center=[50, 0], radius=5)
        m = mk_mission(no_fly_zones=[z])
        r = verify(m, mk_plan(GOOD))
        self.assertEqual(types(r), ["NFZ_SEGMENT"])
        self.assertEqual(r.violations[0].segment, ["A", "B"])
        self.assertIn("entry_point", r.violations[0].evidence)

    def test_tangent_touch_counts_and_clear_passes(self):
        m = mk_mission(no_fly_zones=[NoFlyZone("Z1", "circle", center=[50, 5], radius=5)])
        self.assertEqual(types(verify(m, mk_plan(GOOD))), ["NFZ_SEGMENT"])  # tangent
        m = mk_mission(no_fly_zones=[NoFlyZone("Z1", "circle", center=[50, 5.01], radius=5)])
        self.assertTrue(verify(m, mk_plan(GOOD)).valid)

    def test_polygon_cases(self):
        sq = [[45, -5], [55, -5], [55, 5], [45, 5]]
        m = mk_mission(no_fly_zones=[NoFlyZone("P", "polygon", vertices=sq)])
        r = verify(m, mk_plan(GOOD))
        self.assertEqual(types(r), ["NFZ_SEGMENT"])
        self.assertEqual(r.violations[0].evidence["n_edge_crossings"], 2)
        # edge-grazing
        m = mk_mission(no_fly_zones=[NoFlyZone("P", "polygon", vertices=[[45, 0], [55, 0], [55, 5], [45, 5]])])
        self.assertEqual(types(verify(m, mk_plan(GOOD))), ["NFZ_SEGMENT"])
        # waypoint inside polygon
        m = mk_mission(no_fly_zones=[NoFlyZone("P", "polygon", vertices=[[25, -5], [35, -5], [35, 5], [25, 5]])])
        self.assertIn("NFZ_WAYPOINT", types(verify(m, mk_plan(GOOD))))
        # clear
        m = mk_mission(no_fly_zones=[NoFlyZone("P", "polygon", vertices=[[45, 1], [55, 1], [55, 5], [45, 5]])])
        self.assertTrue(verify(m, mk_plan(GOOD)).valid)

    def test_multiple_zones(self):
        zs = [NoFlyZone("Z1", "circle", center=[45, 0], radius=3), NoFlyZone("Z2", "circle", center=[60, 0], radius=3)]
        r = verify(mk_mission(no_fly_zones=zs), mk_plan(GOOD))
        self.assertEqual({v.zone_id for v in r.violations}, {"Z1", "Z2"})

    def test_distance(self):
        m = mk_mission(max_distance=90)
        r = verify(m, mk_plan(GOOD))
        self.assertEqual(types(r), ["DISTANCE_EXCEEDED"])
        self.assertAlmostEqual(r.violations[0].evidence["excess"], 10.0)
        self.assertTrue(verify(mk_mission(max_distance=100), mk_plan(GOOD)).valid)  # exact limit ok

    def test_ordering(self):
        m = mk_mission(max_distance=500)
        r = verify(m, mk_plan([("B", 70, 0, 60), ("A", 30, 0, 60)]))
        self.assertEqual(types(r), ["ORDER_VIOLATION"])
        self.assertEqual(r.violations[0].evidence["actual_required_sequence"], ["B", "A"])

    def test_preservation_separate_from_validity(self):
        r = verify(mk_mission(), mk_plan([("A", 30, 0, 60)]))  # B dropped
        self.assertTrue(r.valid)
        self.assertFalse(r.mission_preserved)
        self.assertFalse(r.success)
        self.assertEqual(r.preservation_violations[0].type, "MISSING_REQUIRED_WAYPOINT")
        r = verify(mk_mission(), mk_plan([("A", 30, 20, 60), ("B", 70, 0, 60)]))
        self.assertEqual(r.preservation_violations[0].type, "REQUIRED_WAYPOINT_MOVED")
        r = verify(mk_mission(), mk_plan(GOOD, dest=(90, 0, 60)))
        self.assertEqual(r.preservation_violations[0].type, "DESTINATION_CHANGED")

    def test_combined(self):
        m = mk_mission(max_distance=95, no_fly_zones=[NoFlyZone("Z1", "circle", center=[50, 0], radius=5)])
        r = verify(m, mk_plan([("B", 70, 0, 90), ("A", 30, 0, 60)]))
        self.assertEqual(set(types(r)), {"ALTITUDE_ABOVE_MAX", "ORDER_VIOLATION", "DISTANCE_EXCEEDED", "NFZ_SEGMENT"})
        self.assertFalse(r.valid)
        self.assertEqual(sum(v["n_violations"] for k, v in r.per_constraint.items() if k != "mission_preservation"), len(r.violations))

    def test_schema_errors(self):
        with self.assertRaises(SchemaError):
            FlightPlan("M1", [Waypoint("A", 0, 0, 1, 1), Waypoint("A", 1, 1, 1, 2)], Waypoint("DEST", 1, 1, 1, 3)).validate()
        with self.assertRaises(SchemaError):
            FlightPlan.from_llm_json({"waypoints": [{"id": "A", "x": "1", "y": 0, "altitude": 5}], "destination": {"x": 1, "y": 1, "altitude": 5}}, "M1")
        with self.assertRaises(SchemaError):
            FlightPlan.from_llm_json({"waypoints": [{"id": "A", "x": 1, "y": 0}], "destination": {"x": 1, "y": 1, "altitude": 5}}, "M1")
        with self.assertRaises(SchemaError):
            mk_mission(ordering=[["A", "Z"]])

    def test_duplicate_coordinates_zero_length_segment(self):
        r = verify(mk_mission(), mk_plan([("A", 30, 0, 60), ("A2", 30, 0, 60), ("B", 70, 0, 60)]))
        self.assertTrue(r.success)


if __name__ == "__main__":
    unittest.main()

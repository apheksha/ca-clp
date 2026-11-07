import unittest, tempfile, json, os
from src.benchmark.generator import generate_benchmark
from src.benchmark.feasibility import assess
from src.models.schemas import Mission, NoFlyZone
from src.verifier.verifier import verify
from tests.helpers import mk_mission


class TestBenchmark(unittest.TestCase):
    def test_deterministic_and_labels(self):
        with tempfile.TemporaryDirectory() as t:
            a = generate_benchmark(os.path.join(t, "a"), os.path.join(t, "ma"), per_difficulty=3, n_infeasible=3, n_corridor=0, n_per_combo=0)
            b = generate_benchmark(os.path.join(t, "b"), os.path.join(t, "mb"), per_difficulty=3, n_infeasible=3, n_corridor=0, n_per_combo=0)
            self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
            self.assertEqual(len(a), 12)
            for s in a:
                m = Mission.from_dict(s["mission"])
                f = assess(m)
                self.assertEqual(f.status, s["metadata"]["feasibility"])
                if f.status == "FEASIBLE":
                    self.assertTrue(verify(m, f.witness_plan).success)
            diffs = {s["difficulty"] for s in a}
            self.assertEqual(diffs, {"easy", "medium", "hard", "infeasible"})
            hard = [s for s in a if s["difficulty"] == "hard"]
            easy = [s for s in a if s["difficulty"] == "easy"]
            self.assertGreater(min(h["metadata"]["n_zones"] for h in hard), max(e["metadata"]["n_zones"] for e in easy) - 1)

    def test_feasibility_proofs(self):
        self.assertEqual(assess(mk_mission()).status, "FEASIBLE")
        self.assertEqual(assess(mk_mission(max_distance=50)).status, "INFEASIBLE")
        self.assertEqual(assess(mk_mission(ordering=[["A", "B"], ["B", "A"]])).status, "INFEASIBLE")
        self.assertEqual(assess(mk_mission(no_fly_zones=[NoFlyZone("Z", "circle", center=[30, 0], radius=3)])).status, "INFEASIBLE")
        self.assertEqual(assess(mk_mission(min_altitude=90, max_altitude=80)).status, "INFEASIBLE")

    def test_combo_family_isolates_constraints(self):
        from src.benchmark.generator import COMBOS
        with tempfile.TemporaryDirectory() as t:
            sc = generate_benchmark(os.path.join(t, "a"), os.path.join(t, "m"), per_difficulty=0, n_infeasible=0, n_corridor=0, n_per_combo=1)
            self.assertEqual(len(sc), len(COMBOS))
            for s in sc:
                m = Mission.from_dict(s["mission"])
                act = set(s["metadata"]["active"])
                self.assertEqual(set(m.active), act)
                self.assertEqual(assess(m).status, "FEASIBLE")
                if "no_fly_zone" not in act:
                    self.assertEqual(m.no_fly_zones, [])
                if "energy" in act:
                    self.assertIsNotNone(m.max_energy)
                else:
                    self.assertIsNone(m.max_energy)
                self.assertNotIn("battery", m.description) if "energy" not in act else self.assertIn("energy", m.description)
                # a plan violating an INACTIVE family is not reported as invalid for it
                w = assess(m).witness_plan
                self.assertTrue(verify(m, w).success)

    def test_levels_have_battery_and_obstacles(self):
        with tempfile.TemporaryDirectory() as t:
            sc = generate_benchmark(os.path.join(t, "a"), os.path.join(t, "m"), per_difficulty=6, n_infeasible=0, n_corridor=0, n_per_combo=0)
            by = {d: [s for s in sc if s["difficulty"] == d] for d in ("easy", "medium", "hard")}
            self.assertTrue(all("energy" not in s["metadata"]["active"] for s in by["easy"]))
            self.assertTrue(all("energy" in s["metadata"]["active"] for s in by["medium"] + by["hard"]))
            self.assertGreater(sum(s["metadata"]["n_obstacles"] for s in by["hard"] + by["medium"]), 0)
            self.assertLess(min(s["metadata"]["energy_slack"] for s in by["hard"]), min(s["metadata"]["energy_slack"] for s in by["medium"]) + 0.2)

    def test_corridor_family_needs_gap_route(self):
        with tempfile.TemporaryDirectory() as t:
            sc = generate_benchmark(os.path.join(t, "a"), os.path.join(t, "m"), per_difficulty=0, n_infeasible=0, n_corridor=4, n_per_combo=0)
            self.assertEqual(len(sc), 4)
            for s in sc:
                m = Mission.from_dict(s["mission"])
                f = assess(m)
                self.assertEqual(f.status, "FEASIBLE")
                self.assertGreater(f.witness_distance, f.lower_bound * 1.02)   # straight line is blocked by the wall
                self.assertTrue(any(z.kind == "polygon" for z in m.no_fly_zones))

    def test_nl_styles_contain_all_facts(self):
        from src.benchmark.generator import nl_description, STYLES
        m = mk_mission()
        for st in STYLES:
            d = nl_description(m, st)
            for token in ("40", "80", "150", "A", "B"):
                self.assertIn(token, d)

    def test_detour_witness(self):
        m = mk_mission(max_distance=200, no_fly_zones=[NoFlyZone("Z", "circle", center=[50, 0], radius=8)])
        f = assess(m)
        self.assertEqual(f.status, "FEASIBLE")
        self.assertGreater(f.witness_distance, 100)


if __name__ == "__main__":
    unittest.main()

import csv, os, random, tempfile, unittest
from src.benchmark.real_airspace import build_real_scenarios, project, region_data, load_airports
from src.benchmark.feasibility import assess
from src.models.schemas import Mission
from src.verifier.verifier import verify

HEAD = ["id", "ident", "type", "name", "latitude_deg", "longitude_deg", "elevation_ft", "continent", "iso_country", "iso_region", "municipality", "scheduled_service"]


def fixture(path):
    """FIXTURE_NOT_REAL: invented rows in the OurAirports schema, used only to test the pipeline offline."""
    rng = random.Random(1)
    rows = []
    for i, (t, sched) in enumerate([("large_airport", "yes"), ("medium_airport", "yes"), ("medium_airport", "no"), ("small_airport", "yes"), ("large_airport", "yes")]):
        rows.append([i, f"FZ{i}", t, f"Fixture Airport {i}", 40.72 + rng.uniform(-0.12, 0.12), -73.95 + rng.uniform(-0.15, 0.15), 10, "NA", "XX", "XX-A", "Nowhere", sched])
    for j in range(60):
        rows.append([100 + j, f"FH{j}", "heliport" if j % 4 else "small_airport", f"Fixture Site {j}", 40.72 + rng.uniform(-0.2, 0.2), -73.95 + rng.uniform(-0.25, 0.25), 5, "NA", "XX", "XX-A", "Nowhere", "no"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        w.writerows(rows)


class TestRealAirspace(unittest.TestCase):
    def test_projection(self):
        x, y = project(40.82, -73.95, 40.72, -73.95)
        self.assertAlmostEqual(x, 0, places=6)
        self.assertAlmostEqual(y, 11119.5, delta=5)          # 0.1 deg latitude ~ 11.12 km
        x, _ = project(40.72, -73.85, 40.72, -73.95)
        self.assertAlmostEqual(x, 8429, delta=30)            # 0.1 deg longitude at 40.72N ~ 8.43 km

    def test_pipeline_on_fixture(self):
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "FIXTURE_NOT_REAL_airports.csv")
            fixture(p)
            rd = region_data(load_airports(p), (40.72, -73.95), 30)
            self.assertGreaterEqual(len(rd["zones"]), 3)
            sc = build_real_scenarios(p, os.path.join(t, "o"), os.path.join(t, "m"), {"fixture_city": (40.72, -73.95, 30)}, per_region=3, log=lambda *_: None)
            self.assertGreaterEqual(len(sc), 1)
            for s in sc:
                m = Mission.from_dict(s["mission"])
                f = assess(m)
                self.assertEqual(f.status, "FEASIBLE")
                self.assertTrue(verify(m, f.witness_plan).success)
                self.assertEqual(s["metadata"]["source"], "ourairports")
                self.assertGreaterEqual(s["metadata"]["n_blocked_legs"], 1)
                self.assertIn("meters", m.description)
                # sites are never inside a zone
                for r in m.required_waypoints:
                    self.assertFalse(any(z.contains(r.xy) for z in m.no_fly_zones))


if __name__ == "__main__":
    unittest.main()

"""Regression tests for the merged final package (v9 additions kept in the V11 code base + fixes)."""
import json, os, subprocess, sys, tempfile, unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class Merge(unittest.TestCase):
    def test_nonce_changes_prompt_only_when_requested(self):
        from src.planner.prompts import initial_prompt
        from src.benchmark.generator import load_scenarios
        from src.models.schemas import Mission
        m = Mission.from_dict(load_scenarios(os.path.join(ROOT, "benchmark/real"))[0]["mission"])
        base = initial_prompt(m)
        self.assertEqual(base, initial_prompt(m, "spec", None))
        self.assertNotEqual(initial_prompt(m, "spec", 1), initial_prompt(m, "spec", 2))
        self.assertIn("[SAMPLE_ID]1[/SAMPLE_ID]", initial_prompt(m, "spec", 1))

    def test_nonce_condition_registered_and_paired(self):
        import yaml
        c = yaml.safe_load(open(os.path.join(ROOT, "config/experiments.yaml")))
        self.assertTrue(c["conditions"]["D_resample_nonce"]["resample_nonce"])
        self.assertIn(["D_resample_nonce", "A_llm_only"], c["pairs"])

    def test_real_scenarios_present(self):
        from src.benchmark.generator import load_scenarios
        self.assertEqual(len(load_scenarios(os.path.join(ROOT, "benchmark/real"))), 53)

    def test_empty_scenario_folder_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([sys.executable, "run.py", "experiment", "--provider", "mock", "--model", "mock-heuristic-v1", "--scenario-dir", d, "--seeds", "0"],
                               cwd=ROOT, capture_output=True, text=True, timeout=120)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("no scenarios found", r.stdout + r.stderr)

    def test_long_range_profile_loads_and_is_flyable_at_airport_scale(self):
        from src.models.vehicle import get_vehicle
        v = get_vehicle("long_range_fixed_wing")
        self.assertGreaterEqual(v.battery_wh * v.usable_fraction, 300)

    def test_llm_value_output_dir_for_run_outside_standard_layout(self):
        import inspect
        from src.evaluation import llm_value
        src = inspect.getsource(llm_value)
        self.assertIn('os.path.basename(par) == "raw"', src)


if __name__ == "__main__":
    unittest.main()

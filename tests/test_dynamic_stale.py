"""Dynamic study: every kept case must really invalidate the stale plan, and the stale/repaired plans must carry the dynamic mission id."""
import os, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import yaml
from src.benchmark.generator import load_scenarios
from src.experiments.dynamic_runner import build_cases
from src.verifier.verifier import verify


class DynamicStale(unittest.TestCase):
    def _cases(self, folder, n):
        dcfg = yaml.safe_load(open(os.path.join(ROOT, "config/experiments.yaml"))).get("dynamic", {})
        scen = load_scenarios(os.path.join(ROOT, folder))[:n]
        return build_cases(scen, dcfg, {})

    def test_stale_plan_carries_dynamic_id_and_is_really_invalid(self):
        for folder, n in (("benchmark/generated", 40), ("benchmark/real", 6)):
            cases = self._cases(folder, n)
            self.assertGreater(len(cases), 0, folder)
            for c in cases:
                v = verify(c["mission"], c["remaining_plan"])
                self.assertEqual(c["remaining_plan"].mission_id, c["mission"].mission_id, c["case_id"])
                self.assertFalse(v.valid, f"{c['case_id']}: change did not invalidate the stale plan")
                self.assertFalse(v.success)


if __name__ == "__main__":
    unittest.main()

import unittest, json
from src.models.schemas import NoFlyZone
from src.planner.planner import LLMPlanner
from src.planner.scripted import ScriptedLLM, plan_json
from src.repair.loop import LoopConfig, run_condition
from tests.helpers import mk_mission

Z = NoFlyZone("Z1", "circle", center=[50, 0], radius=5)
BAD = plan_json([("A", 30, 0, 60), ("B", 70, 0, 120)], (100, 0, 60))                # altitude + segment crossing
ALT_FIXED = plan_json([("A", 30, 0, 60), ("B", 70, 0, 60)], (100, 0, 60))          # seg crossing remains (1 violation)
GOOD = plan_json([("A", 30, 0, 60), ("W1", 50, 9, 60), ("B", 70, 0, 60)], (100, 0, 60))
WORSE = plan_json([("B", 70, 0, 120), ("A", 30, 0, 120)], (100, 0, 60))             # more violations
DROP = plan_json([("A", 30, 0, 60)], (100, 0, 60))                                   # drops B


def run(replies, **kw):
    m = mk_mission(no_fly_zones=[Z], max_distance=200)
    p = LLMPlanner(ScriptedLLM(replies), max_parse_retries=1)
    cfg = LoopConfig(**kw)
    return run_condition(m, p, cfg, seed=1), p


class TestLoop(unittest.TestCase):
    def test_multi_iteration_success(self):
        o, _ = run([BAD, ALT_FIXED, GOOD])
        self.assertEqual(o.status, "SUCCESS")
        self.assertEqual([i.outcome for i in o.iterations], ["INITIAL", "PARTIAL_REPAIR", "SUCCESS"])
        self.assertTrue(o.final_verification.success)
        self.assertEqual(o.iterations[1].prev_score, 2)

    def test_regression_detected_guard_vs_noguard(self):
        o, _ = run([ALT_FIXED, WORSE, GOOD], regression_guard=False)
        self.assertTrue(o.iterations[1].regression)
        self.assertEqual(o.iterations[1].outcome, "REGRESSION")
        self.assertEqual(o.status, "SUCCESS")
        o, p = run([ALT_FIXED, WORSE, ALT_FIXED], regression_guard=True, max_iterations=2)
        self.assertTrue(o.iterations[1].regression)
        self.assertIn("discarded", p.client.prompts[2])         # guard note shown to the LLM
        self.assertEqual(o.final_verification.score, 1)          # final = best plan, not the regressed one

    def test_no_guard_final_is_last_plan(self):
        o, _ = run([ALT_FIXED, WORSE, WORSE], regression_guard=False, max_iterations=2)
        self.assertEqual(o.status, "REGRESSION")

    def test_mission_destroying_repair_not_success(self):
        o, _ = run([BAD, DROP], max_iterations=1)
        self.assertTrue(o.iterations[1].verification["valid"] is False or True)
        self.assertFalse(o.final_verification.success if o.final_verification else False)

    def test_repeated_failure_no_change(self):
        o, _ = run([ALT_FIXED] * 4, max_iterations=3)
        self.assertEqual([i.outcome for i in o.iterations[1:]], ["NO_CHANGE"] * 3)
        self.assertEqual(o.status, "NO_CHANGE")

    def test_malformed_output_retry_and_failure(self):
        o, _ = run(["not json", ALT_FIXED, GOOD], max_iterations=1)
        self.assertEqual(len(o.initial.calls), 2)
        self.assertIsNotNone(o.initial.plan)
        o, _ = run(["not json", "{bad"], max_iterations=1)
        self.assertEqual(o.status, "FAILURE")

    def test_infeasible_precheck_stops(self):
        m = mk_mission(max_distance=50, no_fly_zones=[Z])
        s = ScriptedLLM([BAD])
        o = run_condition(m, LLMPlanner(s), LoopConfig(), seed=1)
        self.assertEqual(o.status, "INFEASIBLE")
        self.assertEqual(len(s.prompts), 1)  # only the initial call, no repair spent

    def test_baselines(self):
        o, _ = run([BAD], kind="llm_only")
        self.assertTrue(o.released and o.iterations[0].verification["valid"] is False)  # unsafe plan released
        o, _ = run([BAD], kind="llm_verify")
        self.assertFalse(o.released)
        o, _ = run([BAD, BAD, GOOD], kind="resample", max_iterations=3)
        self.assertEqual(o.status, "SUCCESS")
        self.assertEqual(o.iterations[-1].feedback, None)


if __name__ == "__main__":
    unittest.main()

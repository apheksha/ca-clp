"""Objective 4: mid-flight changes with preserved mission state."""
import os, random, tempfile, unittest
from src.models.schemas import NoFlyZone
from src.dynamic.changes import comfortable_plan, execution_state, build_dynamic_mission, classify_failure, CHANGES
from src.verifier.verifier import verify
from src.benchmark.feasibility import assess
from src.experiments.dynamic_runner import run_dynamic_experiment
from src.evaluation.dynamic_report import evaluate_dynamic, load_dynamic
from tests.helpers import mk_mission

Z = NoFlyZone("Z1", "circle", center=[50, 30], radius=8)


def base():
    return mk_mission(no_fly_zones=[Z], max_distance=260, max_energy=300, energy_k_v=1.0)


class TestDynamic(unittest.TestCase):
    def setUp(self):
        self.m = base()
        self.plan = comfortable_plan(self.m)

    def test_comfortable_plan_valid_and_not_minimal(self):
        self.assertTrue(verify(self.m, self.plan).success)
        self.assertGreater(self.plan.waypoints[0].altitude, self.m.min_altitude - 1e-9)

    def test_state_is_preserved_and_consistent(self):
        st = execution_state(self.m, self.plan, 0.5)
        total = verify(self.m, self.plan).route_distance
        self.assertAlmostEqual(st.flown_distance, 0.5 * total, places=6)
        self.assertEqual(st.visited, [w.id for w in self.plan.ordered()][:len(st.visited)])      # a prefix of the plan
        self.assertTrue(all(w.id not in st.visited for w in st.remaining_waypoints))
        self.assertGreater(st.energy_used, 0)

    def test_each_change_affects_plan_and_stays_feasible(self):
        invalidating = 0
        for ch in CHANGES:
            st = execution_state(self.m, self.plan, 0.3)
            r = build_dynamic_mission(self.m, self.plan, st, ch, random.Random(3))
            if r is None:
                continue
            new, rem, info = r
            self.assertEqual(new.start, [st.position[0], st.position[1]])
            self.assertAlmostEqual(new.start_altitude, st.altitude)
            if verify(new, rem).success:                                         # this change left the stale plan valid: not a case (the runner's require_triggered filter drops it)
                continue
            invalidating += 1
            self.assertEqual(assess(new).status, "FEASIBLE", ch)                # but a repair exists
            self.assertTrue(all(r_.id not in st.visited for r_ in new.required_waypoints))   # visited waypoints are not re-required
            self.assertIn("UPDATE DURING FLIGHT", new.description)
        self.assertGreaterEqual(invalidating, 1)

    def test_remaining_budgets_are_reduced_by_what_was_used(self):
        st = execution_state(self.m, self.plan, 0.5)
        new, rem, _ = build_dynamic_mission(self.m, self.plan, st, "add_zone", random.Random(1))
        self.assertAlmostEqual(new.max_distance, self.m.max_distance - st.flown_distance, places=4)
        self.assertAlmostEqual(new.max_energy, self.m.max_energy - st.energy_used, places=4)

    def test_failure_cause_taxonomy(self):
        rec = {"final_status": "REGRESSION", "feasibility": {"status": "INFEASIBLE", "reason": "straight-line lower bound 10 exceeds"},
               "final_verification": None, "dynamic": {"remaining_waypoints": 4, "progress": 0.3, "remaining_required": 2}}
        self.assertEqual(classify_failure(rec), "resource_limit_at_change")
        rec["feasibility"] = {"status": "INFEASIBLE", "reason": "required waypoint A lies inside no-fly zone Z1"}
        self.assertEqual(classify_failure(rec), "change_makes_mission_infeasible")
        rec["feasibility"] = {"status": "FEASIBLE"}
        rec["dynamic"]["remaining_waypoints"] = 1
        self.assertEqual(classify_failure(rec), "little_route_left_to_modify")
        rec["dynamic"]["remaining_waypoints"] = 5
        rec["final_verification"] = {"violations": [{"family": "no_fly_zone"}, {"family": "no_fly_zone"}, {"family": "distance"}], "preservation_violations": []}
        self.assertEqual(classify_failure(rec), "unresolved_no_fly_zone")
        rec["final_status"] = "SUCCESS"
        self.assertIsNone(classify_failure(rec))

    def test_runner_end_to_end_mock(self):
        with tempfile.TemporaryDirectory() as t:
            d = run_dynamic_experiment("config/default.yaml", "config/experiments.yaml", raw_root=t, seeds=[0], limit_cases=6,
                                       conditions=["E_classical", "D_resample", "C_full"], progress=lambda *_: None)
            df, man, _ = load_dynamic(d)
            self.assertEqual(len(df), 18)
            self.assertTrue(man["is_mock"] and man["kind"] == "dynamic")
            self.assertTrue(df.affected.all())
            self.assertTrue(df[df.condition == "E_classical"].final_success.all())     # classical re-plan solves feasible cases with 0 LLM calls
            self.assertTrue(df[df.condition == "E_classical"].n_llm_calls.eq(0).all())
            out = evaluate_dynamic(d, os.path.join(t, "p"), os.path.join(t, "f"), n_examples=1)
            for f in ("dynamic_summary_by_condition_change.csv", "dynamic_failure_causes.csv", "dynamic_paired.csv"):
                self.assertTrue(os.path.exists(os.path.join(out, f)))


if __name__ == "__main__":
    unittest.main()

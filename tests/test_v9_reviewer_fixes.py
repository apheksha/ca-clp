"""Tests for the reviewer-driven changes (v9): 3D zones, sound oracle, direct-patch baseline, prompt modes, temperature audit, dynamics, latency, reporting."""
import json, os, random, tempfile, unittest
from tests.helpers import mk_mission, mk_plan, GOOD
from src.models.schemas import NoFlyZone, RequiredWaypoint, FlightPlan, Waypoint
from src.verifier.verifier import verify
from src.benchmark.feasibility import assess
from src.repair.direct_patch import apply_prescriptive
from src.planner.prompts import mission_block, initial_prompt, PROMPT_MODES
from src.planner.temperature_check import analyse
from src.verifier.dynamics import check_dynamics, VehicleLimits
from src.simulator.export_px4 import to_qgc_plan
from src.utils.geometry_selftest import run_selftest


class Vertical(unittest.TestCase):
    def test_overflight_clears_finite_height_zone(self):
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=10, z_max=50).validate()
        m = mk_mission(no_fly_zones=[z], min_altitude=40, max_altitude=100)
        low = mk_plan([("A", 30, 0, 45), ("B", 70, 0, 45)], dest=(100, 0, 45))
        high = mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))
        self.assertFalse(verify(m, low).valid)
        self.assertTrue(verify(m, high).valid)

    def test_full_height_zone_unchanged(self):
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=10).validate()
        m = mk_mission(no_fly_zones=[z], min_altitude=40, max_altitude=100)
        self.assertFalse(verify(m, mk_plan(GOOD, dest=(100, 0, 90))).valid)

    def test_descending_leg_crosses_zone_prism(self):
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=10, z_min=0, z_max=30).validate()
        self.assertTrue(z.intersects_segment_3d((30, 0), (70, 0), 20, 20))
        self.assertFalse(z.intersects_segment_3d((30, 0), (70, 0), 40, 40))
        self.assertTrue(z.intersects_segment_3d((30, 0), (70, 0), 60, 10))   # descends into the prism while inside the circle

    def test_oracle_uses_overflight(self):
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=10, z_max=50).validate()
        m = mk_mission(no_fly_zones=[z], min_altitude=40, max_altitude=100, max_distance=102.0, ordering=[])
        f = assess(m)
        self.assertEqual(f.status, "FEASIBLE")
        self.assertGreater(f.witness_plan.destination.altitude, 50)


class OracleSoundness(unittest.TestCase):
    def test_required_point_shallowly_inside_zone_is_not_a_proof(self):
        # point 0.5 inside the boundary, tolerance 1.0 -> it may be displaced out; must NOT be INFEASIBLE
        z = NoFlyZone("Z", "circle", center=[30, 10], radius=10.5).validate()
        m = mk_mission(no_fly_zones=[z], max_distance=400)
        self.assertNotEqual(assess(m).status, "INFEASIBLE")

    def test_deep_point_is_infeasible_with_proof_kind(self):
        z = NoFlyZone("Z", "circle", center=[30, 0], radius=10).validate()
        f = assess(mk_mission(no_fly_zones=[z]))
        self.assertEqual((f.status, f.proof_kind, f.guarantee), ("INFEASIBLE", "point_in_zone", "necessary_condition_violated"))

    def test_inactive_families_do_not_prove_infeasibility(self):
        m = mk_mission(min_altitude=90, max_altitude=10, active=["distance", "ordering", "no_fly_zone"])
        self.assertNotEqual(assess(m).proof_kind, "empty_altitude_band")

    def test_distance_bound_accounts_for_tolerance(self):
        m = mk_mission(max_distance=99.5, ordering=[])     # straight chain = 100; tolerance lets points move -> not provable
        self.assertNotEqual(assess(m).status, "INFEASIBLE")
        self.assertEqual(assess(mk_mission(max_distance=80, ordering=[])).proof_kind, "distance_lower_bound")

    def test_unknown_never_from_failed_search(self):
        f = assess(mk_mission(max_distance=101.0))
        self.assertIn(f.status, ("FEASIBLE", "UNKNOWN"))


class DirectPatch(unittest.TestCase):
    def test_patch_fixes_zone_crossing_without_llm(self):
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=10).validate()
        m = mk_mission(no_fly_zones=[z], max_distance=400)
        p = mk_plan(GOOD)
        res = verify(m, p)
        self.assertFalse(res.valid)
        p2, log = apply_prescriptive(m, p, res)
        self.assertTrue(verify(m, p2).success)
        self.assertGreaterEqual(log["inserts"], 1)

    def test_patch_cannot_fix_order(self):
        m = mk_mission()
        p = mk_plan([("B", 70, 0, 60), ("A", 30, 0, 60)])
        p2, log = apply_prescriptive(m, p, verify(m, p))
        self.assertFalse(verify(m, p2).success)
        self.assertIn("ORDER_VIOLATION", log["unfixable"])

    def test_condition_runs_in_loop_with_zero_calls(self):
        from src.repair.loop import LoopConfig, run_condition
        from src.planner.planner import PlannerOutput
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=10).validate()
        m = mk_mission(no_fly_zones=[z], max_distance=400)
        o = run_condition(m, None, LoopConfig(name="G", kind="direct_patch", feedback_level="prescriptive"), 0, initial=PlannerOutput(mk_plan(GOOD), [], None))
        self.assertEqual((o.status, o.released), ("SUCCESS", True))
        self.assertEqual(sum(len(it.calls) for it in o.iterations[1:]), 0)


class PromptModes(unittest.TestCase):
    def test_modes(self):
        z = NoFlyZone("Z1", "circle", center=[50, 0], radius=10).validate()
        m = mk_mission(no_fly_zones=[z])
        m.description = "Fly from (0,0) to (100,0) via A and B avoiding Z1."
        self.assertIn("MISSION_SPEC", mission_block(m, "spec"))
        nl = mission_block(m, "nl_only")
        self.assertNotIn("MISSION_SPEC", nl)
        self.assertIn("circle centred at (50, 0) with radius 10", nl)
        self.assertNotIn("MISSION DESCRIPTION", mission_block(m, "spec_only"))
        with self.assertRaises(ValueError):
            mission_block(m, "bogus")
        self.assertEqual(initial_prompt(m), initial_prompt(m, "spec"))   # default text unchanged


class TemperatureCheck(unittest.TestCase):
    def test_verdicts(self):
        same = [{"text": "a", "request_temperature": 0}] * 4
        self.assertIn("LIKELY-IGNORED", analyse(same, [{"text": "a", "request_temperature": 0.7}] * 4)["verdict"])
        div = [{"text": str(i), "request_temperature": 0.7} for i in range(4)]
        self.assertIn("HONOURED", analyse(same, div)["verdict"])


class Dynamics(unittest.TestCase):
    def test_sharp_turn_and_climb(self):
        m = mk_mission(start_altitude=60)
        p = mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))
        ok = check_dynamics(m, p, VehicleLimits(min_turn_radius=10))
        self.assertTrue(ok["dynamically_feasible"])
        zig = mk_plan([("A", 30, 0, 60), ("X", 31, 30, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))
        self.assertGreater(check_dynamics(m, zig, VehicleLimits(min_turn_radius=50))["turn_violations"], 0)

    def test_qgc_export(self):
        d = to_qgc_plan(mk_mission(), mk_plan(GOOD), 13.0, 77.6, 1.0)
        self.assertEqual(d["fileType"], "Plan")
        self.assertEqual(d["mission"]["items"][0]["command"], 22)


class Latency(unittest.TestCase):
    def test_delayed_case_has_exposure_and_shorter_budget(self):
        import json as j
        from src.dynamic.changes import comfortable_plan, execution_state, build_delayed_dynamic_mission, build_dynamic_mission
        rec = j.load(open("benchmark/generated/easy_000.json"))
        from src.models.schemas import Mission
        m = Mission.from_dict(rec["mission"])
        plan = comfortable_plan(m)
        st, sd = execution_state(m, plan, 0.35), execution_state(m, plan, 0.5)
        r = build_delayed_dynamic_mission(m, plan, st, sd, "tighten_distance", random.Random(1))
        self.assertIsNotNone(r)
        self.assertIn("exposure", r[2])
        self.assertAlmostEqual(r[2]["detection_delay_progress"], 0.15, places=3)
        self.assertEqual(r[0].start[0], sd.position[0])

    def test_selection_is_seeded_and_not_file_order(self):
        from src.experiments.dynamic_runner import selection_order
        from src.benchmark.generator import load_scenarios
        sc = load_scenarios("benchmark/generated")
        a = [s["scenario_id"] for s in selection_order(sc, {"selection_seed": 1})]
        b = [s["scenario_id"] for s in selection_order(sc, {"selection_seed": 1})]
        f = [s["scenario_id"] for s in selection_order(sc, {"selection": "first"})]
        self.assertEqual(a, b)
        self.assertNotEqual(a, f)


class Geometry(unittest.TestCase):
    def test_selftest(self):
        r = run_selftest(400)
        self.assertTrue(r["ok"], r)
        self.assertGreaterEqual(r["adversarial"]["n_adversarial_cases"], 30)


class Reporting(unittest.TestCase):
    def test_metrics_triple_sums_to_one(self):
        from src.evaluation.report import summarize, cluster_bootstrap_ci
        import pandas as pd
        df = pd.DataFrame({"condition": ["A"] * 4, "scenario_id": list("wxyz"), "initial_valid": [0] * 4, "final_valid": [1, 0, 1, 0], "initial_success": [0] * 4,
                           "final_success": [True, False, False, False], "released": [True, True, False, False], "any_regression": [False] * 4, "failure": [False] * 4,
                           "repair_needed": [True] * 4, "repair_success": [True, False, False, False], "init_extraction_error": [False] * 4})
        r = summarize(df, ["condition"]).iloc[0]
        self.assertAlmostEqual(r.completion_rate + r.withheld_rate + r.unsafe_release_rate, 1.0)
        lo, hi = cluster_bootstrap_ci(df, "final_success")
        self.assertLessEqual(lo, hi)


if __name__ == "__main__":
    unittest.main()

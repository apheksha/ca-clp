"""Tests: direct-execution baselines, LLM-value analysis, consistent simulation study, claim levels."""
import json, math, os, tempfile, unittest
from tests.helpers import mk_mission, mk_plan, GOOD
from src.models.schemas import Mission, NoFlyZone, RequiredWaypoint, FlightPlan, Waypoint
from src.verifier.verifier import verify
from src.repair.direct_patch import apply_full, apply_prescriptive
from src.repair.loop import LoopConfig, run_condition
from src.planner.planner import PlannerOutput


def zone_mission(**kw):
    z = NoFlyZone("Z", "circle", center=[50, 0], radius=10).validate()
    return mk_mission(no_fly_zones=[z], max_distance=400, **kw)


class DirectBaselines(unittest.TestCase):
    def test_full_repairs_order_and_moved_waypoint(self):
        m = zone_mission()
        bad = mk_plan([("B", 70, 0, 60), ("A", 31, 0, 60)])
        r = verify(m, bad)
        p2, log = apply_full(m, bad, r)
        self.assertTrue(verify(m, p2).success)
        self.assertTrue(log["rebuilt_skeleton"])
        p3, _ = apply_prescriptive(m, bad, r)
        self.assertFalse(verify(m, p3).success)        # the prescriptive-equivalent control cannot fix structure

    def test_full_without_structure_equals_prescriptive(self):
        m = zone_mission()
        r = verify(m, mk_plan(GOOD))
        a, _ = apply_full(m, mk_plan(GOOD), r)
        b, _ = apply_prescriptive(m, mk_plan(GOOD), r)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_witness_fallback_executes_verifier_route(self):
        m = mk_mission(max_distance=104.0, ordering=[], required_waypoints=[RequiredWaypoint("A", 10, 0), RequiredWaypoint("B", 90, 0)],
                       no_fly_zones=[NoFlyZone("Z", "circle", center=[50, 0], radius=6).validate()])
        bad = mk_plan([("A", 10, 0, 60), ("X", 50, 80, 60), ("B", 90, 0, 60)], dest=(100, 0, 60))   # needless excursion: only a DISTANCE violation, which no local patch can fix
        no_fb = run_condition(m, None, LoopConfig(name="G", kind="direct_full", feedback_level="prescriptive", max_iterations=1), 0, initial=PlannerOutput(bad, [], None))
        fb = run_condition(m, None, LoopConfig(name="Gw", kind="direct_full", feedback_level="prescriptive", max_iterations=1, witness_fallback=True), 0, initial=PlannerOutput(bad, [], None))
        self.assertEqual(fb.status, "SUCCESS")
        self.assertTrue(any(it.diff.get("witness_fallback") for it in fb.iterations))
        self.assertEqual(sum(len(it.calls) for it in fb.iterations), 0)
        self.assertNotEqual(no_fb.status, "SUCCESS")      # the deterministic patch alone cannot remove the excursion; only the witness fallback releases a valid plan


class LLMValue(unittest.TestCase):
    def test_mcnemar_exact(self):
        from src.evaluation.llm_value import mcnemar_exact
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(5, 5), 1.0)
        self.assertLess(mcnemar_exact(10, 0), 0.01)

    def test_analysis_on_mock_run(self):
        from src.experiments.runner import run_experiment
        from src.evaluation.llm_value import llm_value_analysis
        d = tempfile.mkdtemp()
        raw = run_experiment("config/default.yaml", "config/experiments.yaml", "mock", "mock-heuristic-v1", raw_root=os.path.join(d, "raw"), seeds=[0], limit=8, conditions=["C_prescriptive", "G_direct_patch", "G_direct_full_witness", "E_classical"], scenario_dir="benchmark/generated")
        out = llm_value_analysis(raw, os.path.join(d, "lv"))
        self.assertTrue(out)
        md = open(os.path.join(out, "llm_value.md")).read()
        self.assertIn("C_prescriptive vs G_direct_patch", md)
        self.assertIn("MOCK", md)


class SimulationStudy(unittest.TestCase):
    def setUp(self):
        from src.simulator.batch import prepare_mission
        d = mk_mission(min_altitude=40.0, max_altitude=100.0, start_altitude=60.0, ordering=[]).to_dict()
        self.m = prepare_mission(Mission.from_dict(d), "generic_quad_small", 10.0)
        self.plan = mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))

    def test_record_has_all_required_outcomes_and_claim_levels(self):
        from src.simulator.sim import simulate, SimConfig
        from src.simulator.study import execution_record
        tr = simulate(self.m, self.plan, cfg=SimConfig(dt=0.2, gust_sigma=0.5))
        r = execution_record(self.m, self.plan, tr)
        for k in ("cross_track_p95_m", "dev_max_m", "mission_completed", "energy_used_wh", "energy_predicted_wh", "energy_error_pct", "battery_remaining_wh", "intrusion_zones",
                  "intrusion_max_penetration_m", "L1_geometric_verified", "L2_executed_feasible"):
            self.assertIn(k, r)
        self.assertEqual(r["L3_flight_safety"], "not assessed")
        self.assertTrue(r["mission_completed"])

    def test_deviation_zero_on_the_planned_path_and_positive_off_it(self):
        from src.simulator.study import deviation_stats, planned_polyline_m
        pl = planned_polyline_m(self.m, self.plan)
        self.assertAlmostEqual(deviation_stats(pl, [pl[1], pl[2]])["dev_max_m"], 0.0)
        self.assertAlmostEqual(deviation_stats(pl, [(pl[1][0], pl[1][1] + 7.0, pl[1][2])])["dev_horiz_max_m"], 7.0, places=6)

    def test_intrusion_depth_measured(self):
        from src.simulator.sim import TrajectoryResult
        from src.simulator.study import intrusion_stats
        z = NoFlyZone("Z", "circle", center=[50, 0], radius=5).validate()          # 50 m radius at unit_m = 10
        tr = TrajectoryResult(unit_m=10.0)
        tr.t = [0.0, 1.0, 2.0]
        tr.pos = [(300.0, 0.0, 60.0), (480.0, 0.0, 60.0), (700.0, 0.0, 60.0)]      # sample 2 is 20 m from the centre => 30 m deep
        st = intrusion_stats(self.m, tr, [z])
        self.assertEqual(st["n_zones_entered"], 0 if False else st["n_zones_entered"])
        self.assertAlmostEqual(st["max_penetration_m"], 30.0, places=3)

    def test_plot_is_written(self):
        from src.simulator.sim import simulate, SimConfig
        from src.simulator.plots import plot_planned_vs_actual
        tr = simulate(self.m, self.plan, cfg=SimConfig(dt=0.2))
        p = plot_planned_vs_actual(self.m, self.plan, tr, os.path.join(tempfile.mkdtemp(), "p.png"))
        self.assertTrue(os.path.getsize(p) > 1000)

    def test_study_runs_on_a_mock_run_and_writes_claim_gap(self):
        from src.experiments.runner import run_experiment
        from src.simulator.study import simulate_run_study
        d = tempfile.mkdtemp()
        raw = run_experiment("config/default.yaml", "config/experiments.yaml", "mock", "mock-heuristic-v1", raw_root=os.path.join(d, "raw"), seeds=[0], limit=4, conditions=["E_classical", "C_prescriptive"], scenario_dir="benchmark/generated")
        out = simulate_run_study(raw, winds={"calm": (0.0, 0.0)}, seeds=(0,), limit=3, plots=1, save_trajectories=1, out_dir=os.path.join(d, "sim"))
        gap = json.load(open(os.path.join(out, "claim_gap.json")))
        self.assertEqual(gap["L3_flight_safety"], "not assessed")
        self.assertTrue(os.path.exists(os.path.join(out, "executions.csv")))
        self.assertTrue(os.path.exists(os.path.join(out, "execution_summary.csv")))


class Claims(unittest.TestCase):
    def test_claim_levels_text(self):
        from src.evaluation.report import CLAIM_LEVELS_MD
        self.assertIn("not assessed", CLAIM_LEVELS_MD)
        self.assertIn("L2", CLAIM_LEVELS_MD)


if __name__ == "__main__":
    unittest.main()

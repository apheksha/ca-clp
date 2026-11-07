"""Regression tests for the issues found in the first REAL-model pilots (llama3.1:8b via Ollama, command-a via Cohere)."""
import json, os, unittest
from src.models.schemas import NoFlyZone
from src.planner.planner import LLMPlanner, normalize_plan_dict, extract_json
from src.planner.scripted import ScriptedLLM, plan_json
from src.planner.prompts import SYSTEM_PROMPT, PROMPT_VERSION
from src.repair.loop import LoopConfig, run_condition, RESAMPLE_TEMPERATURE
from tests.helpers import mk_mission

Z = NoFlyZone("Z1", "circle", center=[50, 0], radius=5)
BAD = plan_json([("A", 30, 0, 60), ("B", 70, 0, 120)], (100, 0, 60))


class Recording(ScriptedLLM):
    def __init__(self, replies):
        super().__init__(replies); self.temps = []

    def generate(self, system, user, temperature, max_tokens, seed):
        self.temps.append(temperature)
        return super().generate(system, user, temperature, max_tokens, seed)


class TestFormatNormalization(unittest.TestCase):
    def test_start_and_dest_entries_are_dropped_and_flagged(self):
        d = {"waypoints": [{"id": "START", "x": 0, "y": 0, "altitude": 30}, {"id": "A", "x": 30, "y": 0, "altitude": 60},
                           {"id": "B", "x": 70, "y": 0, "altitude": 60}, {"id": "DEST", "x": 100, "y": 0, "altitude": 60}],
             "destination": {"x": 100, "y": 0, "altitude": 60}}
        d2, notes = normalize_plan_dict(d)
        self.assertEqual([w["id"] for w in d2["waypoints"]], ["A", "B"])
        self.assertEqual(len(notes), 2)
        self.assertEqual(len(d["waypoints"]), 4)                              # input not mutated

    def test_unknown_fields_ignored_and_flagged(self):
        d = {"waypoints": [{"id": "A", "x": 1, "y": 2, "altitude": 3, "radius": 5}], "destination": {"x": 9, "y": 9, "altitude": 3, "note": "x"}}
        d2, notes = normalize_plan_dict(d)
        self.assertEqual(set(d2["waypoints"][0]), {"id", "x", "y", "altitude"})
        self.assertEqual(set(d2["destination"]), {"x", "y", "altitude"})
        self.assertEqual(len(notes), 2)

    def test_geometry_is_never_touched(self):
        d = {"waypoints": [{"id": "A", "x": 30.5, "y": 1.25, "altitude": 61}], "destination": {"x": 100, "y": 0, "altitude": 60}}
        d2, notes = normalize_plan_dict(d)
        self.assertEqual(d2, d); self.assertEqual(notes, [])

    def test_disabled_is_strict(self):
        d = {"waypoints": [{"id": "START", "x": 0, "y": 0, "altitude": 30}], "destination": {"x": 1, "y": 1, "altitude": 3}}
        self.assertEqual(normalize_plan_dict(d, enabled=False), (d, []))

    def test_planner_accepts_slip_when_enabled_rejects_when_strict(self):
        m = mk_mission(no_fly_zones=[Z], max_distance=200)
        slip = json.dumps({"waypoints": [{"id": "START", "x": 0, "y": 0, "altitude": 60}] + BAD["waypoints"], "destination": BAD["destination"]})
        ok = LLMPlanner(ScriptedLLM([slip]), max_parse_retries=0, normalize_format=True).initial_plan(m, 0)
        self.assertIsNotNone(ok.plan); self.assertTrue(ok.calls[0]["normalized"])
        strict = LLMPlanner(ScriptedLLM([slip]), max_parse_retries=0, normalize_format=False).initial_plan(m, 0)
        self.assertIsNone(strict.plan); self.assertIn("reserved", strict.error)

    def test_think_block_stripped(self):
        self.assertEqual(extract_json('<think>maybe {"a":0}</think>{"a": 1}'), {"a": 1})

    def test_prompt_forbids_start_dest(self):
        self.assertIn("do NOT list START or DEST", SYSTEM_PROMPT)
        self.assertEqual(PROMPT_VERSION, "plan-v3")


class TestResampleTemperature(unittest.TestCase):
    def test_resample_samples_above_zero_and_records_it(self):
        m = mk_mission(no_fly_zones=[Z], max_distance=200)
        llm = Recording([BAD]); p = LLMPlanner(llm, temperature=0.0, max_parse_retries=0)
        o = run_condition(m, p, LoopConfig(kind="resample", max_iterations=3), seed=1)
        self.assertEqual(llm.temps[0], 0.0)
        self.assertTrue(all(t == RESAMPLE_TEMPERATURE for t in llm.temps[1:]))
        self.assertTrue(all(c["temperature"] == RESAMPLE_TEMPERATURE for it in o.iterations[1:] for c in it.calls))

    def test_closed_loop_keeps_planner_temperature(self):
        m = mk_mission(no_fly_zones=[Z], max_distance=200)
        llm = Recording([BAD]); p = LLMPlanner(llm, temperature=0.0, max_parse_retries=0)
        run_condition(m, p, LoopConfig(kind="closed_loop", max_iterations=2), seed=1)
        self.assertTrue(all(t == 0.0 for t in llm.temps))

    def test_runner_refuses_resample_at_zero_for_real_provider(self):
        from src.experiments import runner
        src = open(runner.__file__).read()
        self.assertIn("resample at temperature 0", src)


class TestDynamicUntriggered(unittest.TestCase):
    def test_every_built_case_actually_invalidates_the_stale_plan(self):
        from src.benchmark.generator import load_scenarios
        from src.experiments.dynamic_runner import build_cases
        from src.verifier.verifier import verify
        scen = [s for s in load_scenarios("benchmark/reduced")]
        cases = build_cases(scen, {"n_per_difficulty": 2, "progress": [0.35], "changes": ["add_zone", "enlarge_zone", "tighten_distance"], "require_triggered": True})
        self.assertGreater(len(cases), 3)
        for c in cases:
            self.assertFalse(verify(c["mission"], c["remaining_plan"]).success, c["case_id"])
        loose = build_cases(scen, {"n_per_difficulty": 2, "progress": [0.35], "changes": ["add_zone", "enlarge_zone", "tighten_distance"], "require_triggered": False})
        self.assertGreaterEqual(len(loose), len(cases))


class TestReducedBenchmark(unittest.TestCase):
    def test_subset_is_stratified_and_deterministic(self):
        files = sorted(os.listdir("benchmark/reduced"))
        self.assertEqual(len([f for f in files if f.startswith("easy_")]), 5)
        self.assertEqual(len([f for f in files if f.startswith("infeasible_")]), 2)
        self.assertEqual(len([f for f in files if f.startswith("combo_")]), 14)


class TestPilotPack(unittest.TestCase):
    def test_pilot_pack_is_certified_and_binds(self):
        from src.benchmark.generator import load_scenarios
        from src.models.schemas import Mission, FlightPlan, Waypoint
        from src.verifier.verifier import verify
        sc = load_scenarios("benchmark/pilot")
        self.assertEqual(len(sc), 10)
        st = {s["scenario_id"]: s["metadata"]["feasibility"] for s in sc}
        self.assertEqual(sorted(k for k, v in st.items() if v == "INFEASIBLE"), ["pilot_09_inf_wp_in_zone", "pilot_10_inf_budget"])
        for s in sc:
            m = Mission.from_dict(s["mission"])
            alt = (m.min_altitude + m.max_altitude) / 2
            w = [Waypoint(r.id, r.x, r.y, alt, i + 1) for i, r in enumerate(m.required_waypoints)]
            naive = FlightPlan(m.mission_id, w, Waypoint("DEST", m.destination[0], m.destination[1], alt, len(w) + 1))
            ok = verify(m, naive).success
            if s["scenario_id"] not in ("pilot_01_control", "pilot_02_alt_narrow"):
                self.assertFalse(ok, s["scenario_id"])          # every non-control scenario defeats a naive straight-line plan


class TestPrescriptiveFeedback(unittest.TestCase):
    def test_detour_is_attached_and_removes_the_zone_violation(self):
        from src.benchmark.generator import load_scenarios
        from src.models.schemas import Mission, FlightPlan, Waypoint
        from src.verifier.verifier import verify
        from src.feedback.feedback import build_feedback, LEVELS, repair_prompt
        self.assertIn("prescriptive", LEVELS)
        s = [x for x in load_scenarios("benchmark/reduced") if x["scenario_id"] == "easy_000"][0]
        m = Mission.from_dict(s["mission"]); alt = (m.min_altitude + m.max_altitude) / 2
        w = [Waypoint(r.id, r.x, r.y, alt, i + 1) for i, r in enumerate(m.required_waypoints)]
        plan = FlightPlan(m.mission_id, w, Waypoint("DEST", m.destination[0], m.destination[1], alt, len(w) + 1))
        res = verify(m, plan); self.assertFalse(res.success)
        fb = build_feedback(m, plan, res, "prescriptive", "targeted")
        dets = [v["suggested_detour"] for v in fb.payload["violations"] if "suggested_detour" in v]
        self.assertTrue(dets)
        self.assertIn("suggested_detour", repair_prompt(m, plan, fb))
        self.assertIn("apply it exactly", repair_prompt(m, plan, fb))
        ev = build_feedback(m, plan, res, "evidence", "targeted")
        self.assertFalse(any("suggested_detour" in v for v in ev.payload["violations"]))     # evidence level must NOT leak coordinates

    def test_upper_bound_script_runs(self):
        import subprocess, sys
        out = subprocess.run([sys.executable, "scripts/prescriptive_upper_bound.py", "benchmark/reduced"], capture_output=True, text=True, timeout=300).stdout
        self.assertIn("NFZ violations fully removed", out)


if __name__ == "__main__":
    unittest.main()

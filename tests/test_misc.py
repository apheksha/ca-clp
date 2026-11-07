import unittest, json, tempfile, os
from src.utils import geometry as G
from src.models.schemas import FlightPlan, NoFlyZone, SchemaError, Mission
from src.verifier.verifier import verify
from src.feedback.feedback import build_feedback, repair_prompt
from src.planner.planner import LLMPlanner
from src.planner.scripted import ScriptedLLM, plan_json
from src.planner.llm import AnthropicClient, OpenAICompatClient, LLMError, make_client
from src.planner.mock_llm import MockLLM
from src.repair.loop import LoopConfig, run_condition
from src.repair.diff import plan_diff
from src.app_logic import run_session
from tests.helpers import mk_mission, mk_plan, GOOD

Z = NoFlyZone("Z1", "circle", center=[50, 0], radius=5)


class TestGeometry(unittest.TestCase):
    def test_segments(self):
        self.assertTrue(G.segments_intersect((0, 0), (10, 10), (0, 10), (10, 0)))
        self.assertTrue(G.segments_intersect((0, 0), (5, 5), (5, 5), (9, 0)))      # touching endpoint
        self.assertFalse(G.segments_intersect((0, 0), (1, 1), (2, 2), (3, 3)))     # collinear disjoint
        self.assertTrue(G.segments_intersect((0, 0), (4, 0), (2, 0), (6, 0)))      # collinear overlap

    def test_polygon_points(self):
        sq = [(0, 0), (4, 0), (4, 4), (0, 4)]
        self.assertEqual(G.point_in_polygon((2, 2), sq), "inside")
        self.assertEqual(G.point_in_polygon((4, 2), sq), "boundary")
        self.assertEqual(G.point_in_polygon((5, 2), sq), "outside")


class TestFeedback(unittest.TestCase):
    def setUp(self):
        self.m = mk_mission(no_fly_zones=[Z], max_distance=100)
        self.plan = mk_plan([("B", 70, 0, 99), ("A", 30, 0, 60)])
        self.res = verify(self.m, self.plan)

    def test_levels(self):
        b = build_feedback(self.m, self.plan, self.res, "binary", "regenerate")
        self.assertEqual(b.payload, {"valid": False})
        l = build_feedback(self.m, self.plan, self.res, "localized", "regenerate")
        self.assertNotIn("evidence", l.payload["violations"][0])
        e = build_feedback(self.m, self.plan, self.res, "evidence", "targeted")
        types = {v["type"] for v in e.payload["violations"]}
        self.assertTrue({"ALTITUDE_ABOVE_MAX", "ORDER_VIOLATION"} <= types)
        self.assertIn("must_remain_unchanged", e.payload)
        self.assertIn("edit_scope", e.payload)
        seg = [v for v in e.payload["violations"] if v["type"] == "NFZ_SEGMENT"]
        self.assertTrue(all(v["zone_id"] == "Z1" and "closest_point_on_segment" in v["evidence"] for v in seg))
        p = repair_prompt(self.m, self.plan, e)
        self.assertIn("[FEEDBACK]", p)
        self.assertIn("ALTITUDE_ABOVE_MAX", p)


class TestPlannerParsing(unittest.TestCase):
    def test_code_fence_ok_and_missing_field_retry(self):
        m = mk_mission()
        fenced = "```json\n" + json.dumps(plan_json(GOOD, (100, 0, 60))) + "\n```"
        s = ScriptedLLM([{"waypoints": [{"id": "A", "x": 1, "y": 2}], "destination": {"x": 1, "y": 1, "altitude": 5}}, fenced])
        out = LLMPlanner(s).initial_plan(m, 0)
        self.assertIsNotNone(out.plan)
        self.assertEqual(len(out.calls), 2)
        self.assertIn("missing fields", out.calls[0]["parse_error"])
        self.assertIn("PARSE_ERROR", s.prompts[1])

    def test_retry_exhaustion(self):
        out = LLMPlanner(ScriptedLLM(["nope"]), max_parse_retries=2).initial_plan(mk_mission(), 0)
        self.assertIsNone(out.plan)
        self.assertEqual(len(out.calls), 3)

    def test_records(self):
        out = LLMPlanner(ScriptedLLM([plan_json(GOOD, (100, 0, 60))])).initial_plan(mk_mission(), 3)
        c = out.calls[0]
        for k in ("model", "prompt_version", "timestamp", "temperature", "role", "attempt", "prompt_sha256"):
            self.assertIn(k, c)


class TestLLMClients(unittest.TestCase):
    def test_anthropic_request_and_parse(self):
        seen = {}
        def tr(url, headers, body):
            seen.update(url=url, headers=headers, body=body)
            return 200, {"content": [{"type": "text", "text": "hi"}], "usage": {"input_tokens": 5, "output_tokens": 2}, "model": "m"}
        c = AnthropicClient("m", api_key="k", transport=tr)
        r = c.generate("sys", "usr", 0.0, 100, 1)
        self.assertEqual((r.text, r.prompt_tokens, r.completion_tokens, r.is_mock), ("hi", 5, 2, False))
        self.assertTrue(seen["url"].endswith("/v1/messages"))
        self.assertEqual(seen["headers"]["x-api-key"], "k")

    def test_openai_and_errors(self):
        c = OpenAICompatClient("m", api_key="k", transport=lambda u, h, b: (200, {"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}}))
        self.assertEqual(c.generate("s", "u", 0, 10, 1).text, "x")
        bad = OpenAICompatClient("m", api_key="k", transport=lambda u, h, b: (401, {"error": "no"}), max_retries=0)
        with self.assertRaises(LLMError):
            bad.generate("s", "u", 0, 10, 1)

    def test_missing_key(self):
        old = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with self.assertRaises(LLMError):
                AnthropicClient("m")
        finally:
            if old:
                os.environ["ANTHROPIC_API_KEY"] = old

    def test_mock_is_labelled_and_deterministic(self):
        c = make_client({"provider": "mock"})
        self.assertTrue(c.is_mock)
        p1 = LLMPlanner(c).initial_plan(mk_mission(), 7).plan
        p2 = LLMPlanner(make_client({"provider": "mock"})).initial_plan(mk_mission(), 7).plan
        self.assertEqual(p1.to_dict() if p1 else None, p2.to_dict() if p2 else None)


class TestE2E(unittest.TestCase):
    def test_nl_to_final_result_with_mock(self):
        m = mk_mission(no_fly_zones=[Z], max_distance=160)
        m.description = "Fly from base to destination via A then B at 40-80 m avoiding Z1, within 160 units."
        statuses = set()
        for seed in range(10):
            r = run_session(m, {"provider": "mock"}, LoopConfig(max_iterations=3), seed)
            self.assertTrue(r["is_mock"])
            statuses.add(r["status"])
            fv = r["final_verification"]
            if r["status"] == "SUCCESS":
                self.assertTrue(fv["valid"] and fv["mission_preserved"])
                FlightPlan.from_dict(r["final_plan"])
                self.assertTrue(verify(m, FlightPlan.from_dict(r["final_plan"])).success)  # independent re-verification
            self.assertEqual(r["iterations"][0]["outcome"], "INITIAL")
        self.assertIn("SUCCESS", statuses)

    def test_plan_diff(self):
        a = mk_plan(GOOD)
        b = mk_plan([("A", 30, 0, 61), ("B", 70, 0, 60)])
        self.assertEqual(plan_diff(a, b)["modified"], ["A"])


if __name__ == "__main__":
    unittest.main()


class TestFreeProviders(unittest.TestCase):
    def test_ollama_needs_no_key_and_presets(self):
        from src.planner.llm import PRESETS
        c = make_client({"provider": "ollama", "model": "x"})
        self.assertEqual((c.name, c.api_key), ("ollama", None))
        self.assertTrue(c.base_url.startswith("http://localhost"))
        for p in ("gemini", "groq", "openrouter"):
            old = os.environ.pop(PRESETS[p]["key_env"], None)
            with self.assertRaises(LLMError):
                make_client({"provider": p, "model": "x"})
            os.environ[PRESETS[p]["key_env"]] = "k"
            self.assertEqual(make_client({"provider": p, "model": "x"}).name, p)
            del os.environ[PRESETS[p]["key_env"]]
            if old:
                os.environ[PRESETS[p]["key_env"]] = old

    def test_no_seed_sent_when_unsupported_and_pacing(self):
        seen = []
        tr = lambda u, h, b: (seen.append((h, b)) or (200, {"choices": [{"message": {"content": "x"}}]}))
        c = OpenAICompatClient("m", base_url="http://x/v1", key_env=None, send_seed=False, transport=tr, min_interval_s=0.2)
        import time
        t = time.monotonic(); c.generate("s", "u", 0, 5, 1); c.generate("s", "u", 0, 5, 1)
        self.assertGreaterEqual(time.monotonic() - t, 0.2)
        self.assertNotIn("seed", seen[0][1])
        self.assertNotIn("Authorization", seen[0][0])


class TestBudgetResume(unittest.TestCase):
    def test_budget_stops_and_resume_completes(self):
        import tempfile
        from src.experiments.runner import run_experiment
        from src.evaluation.report import load_runs
        with tempfile.TemporaryDirectory() as t:
            d = run_experiment("config/default.yaml", "config/experiments_free.yaml", raw_root=t, seeds=[0], limit=6, max_calls=10)
            m = json.load(open(os.path.join(d, "manifest.json")))
            self.assertTrue(m["stopped_early_by_call_budget"])
            n1 = len(open(os.path.join(d, "runs.jsonl")).read().strip().split("\n"))
            self.assertLess(n1, 6 * 7)
            run_experiment("config/default.yaml", "config/experiments_free.yaml", raw_root=t, seeds=[0], limit=6, resume_dir=d)
            df, _ = load_runs(d)
            self.assertEqual(len(df), 6 * 7)
            self.assertEqual(len(df.run_id.unique()), 42)
            piv = df[~df.condition.isin(["E_classical", "F_astar"])].pivot_table(index=["scenario_id", "seed"], columns="condition", values="initial_valid")
            self.assertTrue((piv.nunique(axis=1) == 1).all())     # LLM conditions share one initial plan
            self.assertTrue(df[df.condition == "E_classical"].n_llm_calls.eq(0).all())

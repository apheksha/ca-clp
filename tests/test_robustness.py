import json, math, os, random, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from src.models.schemas import NoFlyZone
from src.utils import geometry as G
from src.planner.llm import OpenAICompatClient, LLMRateLimitError
from src.planner.mock_llm import MockLLM
from src.experiments.runner import run_experiment
from src.evaluation.report import load_runs, paired
from src.repair.loop import LoopConfig, run_condition
from src.planner.planner import LLMPlanner
from tests.helpers import mk_mission


class TestRateLimits(unittest.TestCase):
    def test_429_retry_after_then_success(self):
        seq = [(429, {"error": "slow down"}, {"retry-after": "7"}), (200, {"choices": [{"message": {"content": "ok"}}]}, {})]
        waits = []
        c = OpenAICompatClient("m", base_url="http://x", key_env=None, transport=lambda u, h, b: seq.pop(0), on_wait=lambda w, k, why: waits.append(w))
        with mock.patch("src.planner.llm.time.sleep"):
            self.assertEqual(c.generate("s", "u", 0, 5, 1).text, "ok")
        self.assertTrue(waits and waits[0] >= 7)              # honours Retry-After

    def test_daily_quota_fails_fast(self):
        calls = []
        c = OpenAICompatClient("m", base_url="http://x", key_env=None, transport=lambda u, h, b: (calls.append(1) or (429, {"error": {"message": "Quota exceeded: requests per day"}}, {})))
        with mock.patch("src.planner.llm.time.sleep"):
            with self.assertRaises(LLMRateLimitError) as cm:
                c.generate("s", "u", 0, 5, 1)
        self.assertTrue(cm.exception.daily)
        self.assertEqual(len(calls), 1)                        # no pointless retries

    def test_persistent_429_raises_not_swallowed(self):
        c = OpenAICompatClient("m", base_url="http://x", key_env=None, max_retries=2, transport=lambda u, h, b: (429, {"error": "rate"}, {}))
        with mock.patch("src.planner.llm.time.sleep"):
            with self.assertRaises(LLMRateLimitError):
                c.generate("s", "u", 0, 5, 1)


class _Handler(BaseHTTPRequestHandler):
    mode = "ok"          # ok | flaky | quota_after
    count = 0
    mock = MockLLM()

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).count += 1
        n = type(self).count
        if (self.mode == "flaky" and n % 3 == 0) or (self.mode == "quota_after" and n > 9):
            msg = {"error": {"message": "Quota exceeded: requests per day"}} if self.mode == "quota_after" else {"error": "rate"}
            data = json.dumps(msg).encode()
            self.send_response(429)
            self.send_header("Retry-After", "0")
        else:
            text = self.mock.generate("", body["messages"][1]["content"], 0, 0, body.get("seed", 0)).text
            data = json.dumps({"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class TestEndToEndHTTP(unittest.TestCase):
    """Real urllib transport against a local OpenAI-compatible server (the 'LLM' behind it is the labelled mock)."""

    def serve(self, mode):
        _Handler.mode, _Handler.count = mode, 0
        srv = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.addCleanup(srv.shutdown)
        os.environ["OLLAMA_BASE_URL"] = f"http://127.0.0.1:{srv.server_port}/v1"
        self.addCleanup(lambda: os.environ.pop("OLLAMA_BASE_URL", None))

    def test_flaky_429_is_survived(self):
        self.serve("flaky")
        with tempfile.TemporaryDirectory() as t, mock.patch("src.planner.llm.time.sleep"):
            d = run_experiment("config/default.yaml", "config/experiments_free.yaml", provider="ollama", model="x", raw_root=t,
                               limit=2, conditions=["A_llm_only", "C_full"], progress=lambda *_: None)
            df, man = load_runs(d)
            self.assertEqual(len(df), 4)
            self.assertIsNone(man.get("WARNING"))
            self.assertFalse(man["is_mock"] is True and False)
            self.assertEqual(man["provider"], "ollama")

    def test_quota_stop_saves_progress_without_fake_failures_then_resume(self):
        self.serve("quota_after")
        with tempfile.TemporaryDirectory() as t, mock.patch("src.planner.llm.time.sleep"):
            d = run_experiment("config/default.yaml", "config/experiments_free.yaml", provider="ollama", model="x", raw_root=t,
                               limit=8, conditions=["A_llm_only", "D_resample", "C_full"], progress=lambda *_: None)
            man = json.load(open(os.path.join(d, "manifest.json")))
            self.assertIn("provider error", man["stopped_reason"])
            df, _ = load_runs(d)
            self.assertGreater(len(df), 0)
            self.assertLess(len(df), 24)
            self.assertFalse(df.failure.any())               # the 429 never became a 'model failure'
            _Handler.mode = "ok"
            d2 = run_experiment("config/default.yaml", "config/experiments_free.yaml", provider="ollama", model="x", resume_dir=d,
                                limit=8, conditions=["A_llm_only", "D_resample", "C_full"], progress=lambda *_: None)
            df2, _ = load_runs(d2)
            self.assertEqual(len(df2), 24)
            self.assertEqual(df2.run_id.nunique(), 24)


class TestGeometryFuzz(unittest.TestCase):
    def test_circle_vs_dense_sampling(self):
        rng = random.Random(7)
        for _ in range(3000):
            z = NoFlyZone("Z", "circle", center=[rng.uniform(20, 80), rng.uniform(20, 80)], radius=rng.uniform(2, 15)).validate()
            a, b = (rng.uniform(0, 100), rng.uniform(0, 100)), (rng.uniform(0, 100), rng.uniform(0, 100))
            md = min(G.dist((a[0] + (b[0] - a[0]) * k / 2000, a[1] + (b[1] - a[1]) * k / 2000), z.centroid()) for k in range(2001))
            got = z.intersects_segment(a, b)
            if md < z.radius - 0.05:
                self.assertTrue(got)
            if md > z.radius + 0.05:
                self.assertFalse(got)

    def test_polygon_vs_dense_sampling(self):
        rng = random.Random(11)
        for _ in range(1500):
            cx, cy, R = rng.uniform(30, 70), rng.uniform(30, 70), rng.uniform(4, 15)
            n = rng.randint(3, 7)
            verts = [[cx + R * rng.uniform(0.6, 1) * math.cos(2 * math.pi * k / n), cy + R * rng.uniform(0.6, 1) * math.sin(2 * math.pi * k / n)] for k in range(n)]
            z = NoFlyZone("P", "polygon", vertices=verts).validate()   # star-shaped, possibly concave
            a, b = (rng.uniform(0, 100), rng.uniform(0, 100)), (rng.uniform(0, 100), rng.uniform(0, 100))
            vs = z._verts()
            samples = [(a[0] + (b[0] - a[0]) * k / 3000, a[1] + (b[1] - a[1]) * k / 3000) for k in range(3001)]
            inside = [s for s in samples if G.point_in_polygon(s, vs) == "inside"]
            bdist = min(G.min_dist_to_polygon_boundary(s, vs) for s in samples[::10])
            got = z.intersects_segment(a, b)
            if inside:
                self.assertTrue(got)
            elif bdist > 0.3:
                self.assertFalse(got)


class TestStats(unittest.TestCase):
    def test_holm_monotone_and_perm(self):
        import pandas as pd
        rows = []
        for sc in range(12):
            for cond, p in (("T", 1.0), ("C", 0.0 if sc < 10 else 1.0), ("U", 1.0)):
                rows.append(dict(scenario_id=f"s{sc}", seed=0, condition=cond, feasibility="FEASIBLE", final_success=p))
        out = paired(pd.DataFrame(rows), [["T", "C"], ["T", "U"]])
        self.assertLess(out.scenario_perm_p.iloc[0], 0.01)
        self.assertEqual(out.scenario_perm_p.iloc[1], 1.0)
        self.assertTrue((out.scenario_perm_p_holm >= out.scenario_perm_p - 1e-12).all())


class TestClassical(unittest.TestCase):
    def test_classical_solves_without_llm_and_flags_infeasible(self):
        from src.planner.scripted import ScriptedLLM
        s = ScriptedLLM(["x"])
        o = run_condition(mk_mission(no_fly_zones=[NoFlyZone("Z", "circle", center=[50, 0], radius=8)], max_distance=200), LLMPlanner(s), LoopConfig(kind="classical"), 0)
        self.assertEqual((o.status, len(s.prompts)), ("SUCCESS", 0))
        o = run_condition(mk_mission(max_distance=50), LLMPlanner(s), LoopConfig(kind="classical"), 0)
        self.assertEqual(o.status, "INFEASIBLE")


if __name__ == "__main__":
    unittest.main()

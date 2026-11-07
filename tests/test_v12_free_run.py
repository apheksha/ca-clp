"""v12: free-tier robustness of the LLM client (rate limits, quotas, system-role fallback) and the one-command run (run_free.py)."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from src.planner import llm as L

GEMINI_MINUTE = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                           "message": "You exceeded your current quota. Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
                                      "limit: 15, model: gemini-3.5-flash-lite\nPlease retry in 16.5s.",
                           "details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}, {"retryDelay": "16s"}]}}
GEMINI_DAY = {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Quota exceeded for metric: generate_content_free_tier_requests, limit: 1000",
                        "details": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}}
GEMINI_DAY_WITH_DELAY = json.loads(json.dumps(GEMINI_MINUTE).replace("PerMinute", "PerDay"))     # Gemini may attach a short retryDelay even to a daily limit
GROQ_TPD = {"error": {"message": "Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): Limit 200000, Used 199500, Requested 2300. "
                                 "Please try again in 7m12.5s.", "type": "tokens", "code": "rate_limit_exceeded"}}
GROQ_TPM = {"error": {"message": "Rate limit reached for model x on tokens per minute (TPM): Limit 8000. Please try again in 2.4s."}}
OK = {"choices": [{"message": {"content": '{"waypoints": [], "destination": {"x": 1, "y": 2}}'}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class FakeTransport:
    def __init__(self, replies):
        self.replies, self.bodies = list(replies), []

    def __call__(self, url, headers, body):
        self.bodies.append(body)
        r = self.replies.pop(0)
        return (r[0], r[1], r[2] if len(r) > 2 else {})


def client(replies, **kw):
    t = FakeTransport(replies)
    c = L.OpenAICompatClient("m", api_key="k", base_url="http://x", transport=t, max_retries=3, backoff=0.01, **kw)
    return c, t


class QuotaClassification(unittest.TestCase):
    def test_classify(self):
        self.assertEqual(L.classify_429(json.dumps(GEMINI_MINUTE)), "transient")     # v11 called this daily and stopped the run
        self.assertEqual(L.classify_429(json.dumps(GEMINI_DAY)), "daily")
        self.assertEqual(L.classify_429(json.dumps(GROQ_TPD)), "daily")
        self.assertEqual(L.classify_429(json.dumps(GROQ_TPM)), "transient")

    def test_wait_parsing(self):
        self.assertAlmostEqual(L.parse_wait_s({}, GEMINI_MINUTE), 16.5)
        self.assertAlmostEqual(L.parse_wait_s({}, GROQ_TPD), 432.5)
        self.assertAlmostEqual(L.parse_wait_s({"retry-after": "30"}, {}), 30.0)
        self.assertAlmostEqual(L.parse_wait_s({}, {"m": "try again in 1h2m3s"}), 3723.0)
        self.assertIsNone(L.parse_wait_s({}, GEMINI_DAY))


@mock.patch.object(L.time, "sleep", lambda s: None)
class ClientBehaviour(unittest.TestCase):
    def test_per_minute_quota_is_retried_not_fatal(self):
        c, t = client([(429, GEMINI_MINUTE), (429, GEMINI_MINUTE), (200, OK)])
        r = c.generate("sys", "user", 0.0, 100, 0)
        self.assertIn("waypoints", r.text)
        self.assertEqual(len(t.bodies), 3)

    def test_daily_quota_with_short_window_waits(self):
        short = json.loads(json.dumps(GROQ_TPD).replace("7m12.5s", "40s"))
        c, t = client([(429, short), (200, OK)])
        self.assertIn("waypoints", c.generate("s", "u", 0.0, 100, 0).text)

    def test_daily_quota_without_reset_hint_raises_daily(self):
        c, _ = client([(429, GEMINI_DAY)])
        with self.assertRaises(L.LLMRateLimitError) as cm:
            c.generate("s", "u", 0.0, 100, 0)
        self.assertTrue(cm.exception.daily)
        self.assertEqual(L.stop_info(cm.exception)["kind"], "daily_quota")

    def test_daily_limit_with_misleading_short_delay_gives_up_after_a_few_waits(self):
        c, t = client([(429, GEMINI_DAY_WITH_DELAY)] * 10)
        with self.assertRaises(L.LLMRateLimitError) as cm:
            c.generate("s", "u", 0.0, 100, 0)
        self.assertTrue(cm.exception.daily)
        self.assertEqual(len(t.bodies), 7)            # 6 waits, then stop: a daily limit is not waited for indefinitely

    def test_long_daily_window_raises_with_retry_after(self):
        c, _ = client([(429, GROQ_TPD)], max_daily_wait_s=60)
        with self.assertRaises(L.LLMRateLimitError) as cm:
            c.generate("s", "u", 0.0, 100, 0)
        self.assertTrue(cm.exception.daily)
        self.assertAlmostEqual(cm.exception.retry_after_s, 432.5)

    def test_persistent_minute_limit_is_rate_limit_not_daily(self):
        c, _ = client([(429, GEMINI_MINUTE)] * 4)
        with self.assertRaises(L.LLMRateLimitError) as cm:
            c.generate("s", "u", 0.0, 100, 0)
        self.assertFalse(cm.exception.daily)
        self.assertEqual(L.stop_info(cm.exception)["kind"], "rate_limit")

    def test_system_role_rejected_is_merged_and_recorded(self):
        err = {"error": {"code": 400, "message": "Developer instruction is not enabled for models/gemma-4-31b-it", "status": "INVALID_ARGUMENT"}}
        c, t = client([(400, err), (200, OK), (200, OK)])
        r = c.generate("SYSTEM TEXT", "USER TEXT", 0.0, 100, 0)
        self.assertTrue(r.raw["system_merged"])
        self.assertEqual([m["role"] for m in t.bodies[1]["messages"]], ["user"])
        self.assertIn("SYSTEM TEXT", t.bodies[1]["messages"][0]["content"])
        c.generate("s", "u", 0.0, 100, 0)                       # remembered for the session
        self.assertEqual(len(t.bodies[2]["messages"]), 1)

    def test_other_400_is_not_retried_and_is_bad_request(self):
        c, t = client([(400, {"error": {"message": "model not found"}})])
        with self.assertRaises(L.LLMError) as cm:
            c.generate("s", "u", 0.0, 100, 0)
        self.assertEqual(len(t.bodies), 1)
        self.assertEqual(L.stop_info(cm.exception)["kind"], "bad_request")

    def test_outage_kinds(self):
        self.assertEqual(L.stop_info(L.LLMError('HTTP 599: {"error": "network error: timed out"}'))["kind"], "network")
        self.assertEqual(L.stop_info(L.LLMError("HTTP 503: overloaded"))["kind"], "server")

    def test_auth_error_kind(self):
        self.assertEqual(L.stop_info(L.LLMError("HTTP 401: invalid key"))["kind"], "auth")



class InterruptedWrites(unittest.TestCase):
    """A run killed while appending (power cut, closed laptop) must resume cleanly: the half line is dropped and redone."""

    def test_partial_last_line(self):
        from src.utils.io import append_jsonl, read_jsonl
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "runs.jsonl")
            append_jsonl(p, {"a": 1})
            append_jsonl(p, {"a": 2})
            with open(p, "a", encoding="utf-8") as fh:
                fh.write('{"a": 3, "half')
            self.assertEqual([r["a"] for r in read_jsonl(p)], [1, 2])
            append_jsonl(p, {"a": 4})
            self.assertEqual([r["a"] for r in read_jsonl(p)], [1, 2, 4])

    def test_corrupt_middle_line_is_an_error(self):
        from src.utils.io import read_jsonl
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "runs.jsonl")
            open(p, "w", encoding="utf-8").write('{"a": 1}\nnot json\n{"a": 2}\n')
            with self.assertRaises(ValueError):
                read_jsonl(p)


class DotEnv(unittest.TestCase):
    def test_windows_style_env_file(self):
        from src.utils.dotenv import load_dotenv
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, ".env")
            open(p, "wb").write('\ufeffZZ_TEST_KEY_A="abc123"\r\nexport ZZ_TEST_KEY_B=xyz  # comment\r\n'.encode("utf-8"))
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ZZ_TEST_KEY_A", None)
                os.environ.pop("ZZ_TEST_KEY_B", None)
                load_dotenv(p)
                self.assertEqual(os.environ["ZZ_TEST_KEY_A"], "abc123")
                self.assertEqual(os.environ["ZZ_TEST_KEY_B"], "xyz")


class AutoModelChoice(unittest.TestCase):
    def test_resolve(self):
        rf = __import__("importlib").import_module("run_free")
        av = ["gemini-3.5-flash-lite", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.8-flash-preview-09", "gemma-4-31b-it", "gemma-4-12b-it",
              "gemma-3n-e4b-it", "gemma-3-27b-it", "text-embedding-005"]
        self.assertEqual(rf.resolve_auto("auto-gemma", av)[0], "gemma-4-31b-it")
        self.assertNotIn("gemma-3n-e4b-it", rf.resolve_auto("auto-gemma", av))
        self.assertEqual(rf.resolve_auto("auto-flash", av)[0], "gemini-3.8-flash")
        self.assertNotIn("gemini-3.5-flash-lite", rf.resolve_auto("auto-flash", av))
        self.assertEqual(rf.resolve_auto("gemma-4-31b-it", av), ["gemma-4-31b-it"])



class ResumeAfterQuota(unittest.TestCase):
    """End to end: a daily quota that runs out every 12 calls. Re-running the same command must finish every stage without duplicating any run."""

    def test_sessions_resume_without_duplicates(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, CA_CLP_FLAKY_AFTER="12", PYTHONUTF8="1")
            cmd = [sys.executable, os.path.join(ROOT, "run_free.py"), "--dry-run", "--quick", "--no-wait", "--skip-tests", "--only", "second_models,dynamic", "--out-base", d]
            codes = []
            for _ in range(12):
                r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
                codes.append(r.returncode)
                with open(os.path.join(d, "free_run", "state.json"), encoding="utf-8") as fh:
                    st = json.load(fh)
                jobs = {k: v for k, v in st["jobs"].items() if k.startswith(("second:", "dynamic"))}
                if jobs and all(v["status"] == "done" for v in jobs.values()):
                    break
            import yaml
            n_extra = len(yaml.safe_load(open(os.path.join(ROOT, "config", "free_run.yaml"), encoding="utf-8")).get("second_models") or [])
            self.assertEqual(len([k for k in jobs if k.startswith("second:")]), n_extra, "every extra model in config/free_run.yaml must have run")
            self.assertIn("dynamic", jobs)
            self.assertTrue(all(v["status"] == "done" for v in jobs.values()), (codes, {k: v["status"] for k, v in jobs.items()}, r.stdout[-1500:]))
            self.assertGreater(len(codes), 1, "the simulated quota should have interrupted at least one session")
            for v in jobs.values():
                with open(os.path.join(v["dir"], "runs.jsonl"), encoding="utf-8") as fh:
                    lines = [json.loads(l) for l in fh if l.strip()]
                keys = [(x["scenario_id"], x["condition"], x["seed"]) for x in lines]
                self.assertEqual(len(keys), len(set(keys)), "a run was recorded twice")



class OracleCache(unittest.TestCase):
    def test_cache_hit_and_tamper_protection(self):
        from src.benchmark.generator import load_scenarios
        from src.benchmark.oracle_cache import assess_cached, key
        from src.models.schemas import Mission
        s = [x for x in load_scenarios("benchmark/real") if x["metadata"]["feasibility"] == "FEASIBLE"][0]
        m = Mission.from_dict(s["mission"])
        with tempfile.TemporaryDirectory() as d:
            f1 = assess_cached(m, d)
            path = os.path.join(d, key(m) + ".json")
            self.assertTrue(os.path.exists(path))
            with mock.patch("src.benchmark.oracle_cache.assess", side_effect=AssertionError("cache not used")):
                f2 = assess_cached(m, d)
            self.assertEqual((f1.status, f1.witness_distance), (f2.status, f2.witness_distance))
            data = json.load(open(path, encoding="utf-8"))
            data["witness_plan"]["waypoints"] = []          # a witness that cannot pass the verifier
            json.dump(data, open(path, "w", encoding="utf-8"))
            calls = []
            from src.benchmark import feasibility as F
            with mock.patch("src.benchmark.oracle_cache.assess", side_effect=lambda mm: calls.append(1) or F.assess(mm)):
                f3 = assess_cached(m, d)
            self.assertEqual(len(calls), 1, "a tampered FEASIBLE entry must be recomputed")
            self.assertEqual(f3.status, "FEASIBLE")


if __name__ == "__main__":
    unittest.main()

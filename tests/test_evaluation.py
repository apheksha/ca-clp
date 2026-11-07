import unittest, tempfile, os, json
import pandas as pd
from src.experiments.runner import run_experiment
from src.evaluation.report import evaluate_run, wilson, load_runs


class TestEvaluation(unittest.TestCase):
    def test_runner_and_metrics_from_raw(self):
        with tempfile.TemporaryDirectory() as t:
            d = run_experiment("config/default.yaml", "config/experiments.yaml", raw_root=t, seeds=[0, 1], limit=3, conditions=["A_llm_only", "C_full", "D_resample"])
            man = json.load(open(os.path.join(d, "manifest.json")))
            self.assertTrue(man["is_mock"] and man["WARNING"])
            self.assertEqual(set(man["scenario_files_sha256"]), set(man["scenario_ids"]))     # scenario hashes are recorded
            lines = open(os.path.join(d, "runs.jsonl")).read().strip().split("\n")
            self.assertEqual(len(lines), 3 * 3 * 2)
            ids = [json.loads(l)["run_id"] for l in lines]
            self.assertEqual(len(set(ids)), len(ids))
            # raw data are never overwritten: a second run gets a new directory
            d2 = run_experiment("config/default.yaml", "config/experiments.yaml", raw_root=t, seeds=[0], limit=1, conditions=["A_llm_only"])
            self.assertNotEqual(d, d2)
            out = evaluate_run(d, os.path.join(t, "p"), os.path.join(t, "f"))
            s = pd.read_csv(os.path.join(out, "summary_by_condition.csv"))
            df, _ = load_runs(d)
            for c in s.condition:
                self.assertAlmostEqual(s[s.condition == c].final_success_rate.iloc[0], df[df.condition == c].final_success.mean())
            # shared initial plans => same initial validity across conditions for the same scenario+seed
            piv = df.pivot_table(index=["scenario_id", "seed"], columns="condition", values="initial_valid")
            self.assertTrue((piv.nunique(axis=1) == 1).all())
            self.assertTrue(os.path.exists(os.path.join(t, "f", man["experiment_run_id"], "success_by_condition.png")))

    def test_wilson(self):
        lo, hi = wilson(5, 10)
        self.assertTrue(0.23 < lo < 0.24 and 0.76 < hi < 0.77)


if __name__ == "__main__":
    unittest.main()

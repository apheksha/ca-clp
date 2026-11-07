"""Unit tests for the 10 audit metric fixes and run folder management."""
import json
import math
import os
import tempfile
import unittest
import pandas as pd

from src.evaluation.metrics import run_metrics
from src.evaluation.report import summarize, write_tables
from src.pipeline.final import make_run_dir, find_latest_run_dir


class TestAuditMetricFixes(unittest.TestCase):
    def test_final_success_requires_both_final_valid_and_released(self):
        # Case 1: valid and released -> True
        rec_ok = {
            "iterations": [{"calls": [], "verification": {"score": 0, "success": True, "valid": True}}],
            "final_verification": {"score": 0, "valid": True, "success": True, "violations": [], "preservation_violations": [], "per_constraint": {}},
            "final_status": "SUCCESS",
            "metrics": {"released": True},
            "is_mock": True,
            "timing": {"wall_time_s": 0.1},
            "scenario_meta": {},
        }
        m1 = run_metrics(rec_ok)
        self.assertTrue(m1["final_valid"])
        self.assertTrue(m1["released"])
        self.assertTrue(m1["final_success"])

        # Case 2: valid but NOT released -> False
        rec_unrel = dict(rec_ok, metrics={"released": False})
        m2 = run_metrics(rec_unrel)
        self.assertTrue(m2["final_valid"])
        self.assertFalse(m2["released"])
        self.assertFalse(m2["final_success"])

        # Case 3: released but NOT valid (like A_llm_only failure) -> False
        rec_inv = dict(rec_ok, final_verification={"score": 2, "valid": False, "success": False, "violations": [], "preservation_violations": [], "per_constraint": {}})
        m3 = run_metrics(rec_inv)
        self.assertFalse(m3["final_valid"])
        self.assertTrue(m3["released"])
        self.assertFalse(m3["final_success"])

    def test_distance_overhead_vs_witness_non_negative_and_success_gated(self):
        # Case 1: Successful plan with rd > wd -> positive overhead
        rec_succ = {
            "iterations": [{"calls": [], "verification": {"score": 0, "success": True, "valid": True}}],
            "final_verification": {"score": 0, "valid": True, "success": True, "route_distance": 120.0, "violations": [], "preservation_violations": [], "per_constraint": {}},
            "final_status": "SUCCESS",
            "metrics": {"released": True},
            "is_mock": True,
            "timing": {"wall_time_s": 0.1},
            "scenario_meta": {"witness_distance": 100.0},
        }
        m1 = run_metrics(rec_succ)
        self.assertIsNotNone(m1["distance_overhead_vs_witness"])
        self.assertAlmostEqual(m1["distance_overhead_vs_witness"], 0.2)
        self.assertGreaterEqual(m1["distance_overhead_vs_witness"], 0.0)

        # Case 2: Successful plan with rd < wd due to floating point noise -> clamped to 0.0
        rec_clamp = {
            "iterations": [{"calls": [], "verification": {"score": 0, "success": True, "valid": True}}],
            "final_verification": {"score": 0, "valid": True, "success": True, "route_distance": 99.99, "violations": [], "preservation_violations": [], "per_constraint": {}},
            "final_status": "SUCCESS",
            "metrics": {"released": True},
            "is_mock": True,
            "timing": {"wall_time_s": 0.1},
            "scenario_meta": {"witness_distance": 100.0},
        }
        m2 = run_metrics(rec_clamp)
        self.assertEqual(m2["distance_overhead_vs_witness"], 0.0)

        # Case 3: Invalid or unreleased plan -> overhead is None
        rec_fail = {
            "iterations": [{"calls": [], "verification": {"score": 0, "success": True, "valid": True}}],
            "final_verification": {"score": 1, "valid": False, "success": False, "route_distance": 80.0, "violations": [{"type": "no_fly_zone"}], "preservation_violations": [], "per_constraint": {}},
            "final_status": "FAILURE",
            "metrics": {"released": False},
            "is_mock": True,
            "timing": {"wall_time_s": 0.1},
            "scenario_meta": {"witness_distance": 100.0},
        }
        m3 = run_metrics(rec_fail)
        self.assertIsNone(m3["distance_overhead_vs_witness"])

    def test_repair_success_denominator(self):
        df = pd.DataFrame([
            {"condition": "C1", "repair_needed": True, "repair_success": True, "any_regression": False, "initial_valid": False, "final_valid": True,
             "initial_success": False, "final_success": True, "released": True, "failure": False, "n_llm_calls": 2, "n_repair_iterations": 1,
             "waypoint_mods_total": 1, "collateral_edits": 0, "wall_time_s": 1.0, "verification_time_s": 0.1, "prompt_tokens": 100,
             "completion_tokens": 50, "n_parse_errors": 0, "n_new_violations_introduced": 0, "dyn_recurrence": False, "dyn_cascade": False},
            {"condition": "C1", "repair_needed": True, "repair_success": False, "any_regression": True, "initial_valid": False, "final_valid": False,
             "initial_success": False, "final_success": False, "released": False, "failure": True, "n_llm_calls": 3, "n_repair_iterations": 2,
             "waypoint_mods_total": 2, "collateral_edits": 1, "wall_time_s": 1.5, "verification_time_s": 0.2, "prompt_tokens": 200,
             "completion_tokens": 100, "n_parse_errors": 0, "n_new_violations_introduced": 1, "dyn_recurrence": True, "dyn_cascade": True},
            # Condition C2: no repair needed (classical/astar)
            {"condition": "C2", "repair_needed": False, "repair_success": False, "any_regression": False, "initial_valid": True, "final_valid": True,
             "initial_success": True, "final_success": True, "released": True, "failure": False, "n_llm_calls": 0, "n_repair_iterations": 0,
             "waypoint_mods_total": 0, "collateral_edits": 0, "wall_time_s": 0.05, "verification_time_s": 0.01, "prompt_tokens": 0,
             "completion_tokens": 0, "n_parse_errors": 0, "n_new_violations_introduced": 0, "dyn_recurrence": False, "dyn_cascade": False},
        ])
        s = summarize(df, ["condition"])
        row_c1 = s[s.condition == "C1"].iloc[0]
        self.assertEqual(row_c1["n_repair_needed"], 2)
        self.assertAlmostEqual(row_c1["repair_success_rate"], 0.5)

        row_c2 = s[s.condition == "C2"].iloc[0]
        self.assertEqual(row_c2["n_repair_needed"], 0)
        self.assertTrue(math.isnan(row_c2["repair_success_rate"]))

        # Check write_tables formats NaN as N/A in table_main.md
        with tempfile.TemporaryDirectory() as td:
            write_tables(td, s, pd.DataFrame())
            md = open(os.path.join(td, "table_main.md")).read()
            self.assertIn("N/A", md)

    def test_run_folder_creation_and_resume(self):
        with tempfile.TemporaryDirectory() as td:
            # Create a run folder
            r1 = make_run_dir("test-model", "reduced", root=td)
            self.assertIn("test-model_reduced_", r1)
            os.makedirs(r1, exist_ok=True)
            sp1 = os.path.join(r1, "state.json")
            json.dump({"model": "test-model", "profile": "reduced", "static_done": False, "dynamic_done": False}, open(sp1, "w"))

            # find_latest_run_dir should find r1
            latest = find_latest_run_dir("test-model", "reduced", roots=[td])
            self.assertEqual(latest, r1)

            # Starting fresh should create a distinct new run folder
            r2 = make_run_dir("test-model", "reduced", root=td)
            self.assertNotEqual(r1, r2)                      # an existing folder is never reused
            self.assertTrue(r1.startswith(os.path.join(td, "test-model_reduced_")))


if __name__ == "__main__":
    unittest.main()

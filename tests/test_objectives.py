"""Tests for the lit-survey objectives: energy, severity, A*, repair dynamics (recurrence / cascade / oscillation)."""
import unittest
from src.models.schemas import NoFlyZone
from src.verifier.verifier import verify
from src.planner.astar import astar_plan
from src.planner.planner import LLMPlanner
from src.planner.scripted import ScriptedLLM, plan_json
from src.repair.loop import LoopConfig, run_condition
from src.evaluation.repair_dynamics import repair_dynamics
from src.feedback.feedback import build_feedback
from tests.helpers import mk_mission, mk_plan, GOOD

Z = NoFlyZone("Z1", "circle", center=[50, 0], radius=5)


def rec_from(o):
    return {"iterations": [it.__dict__ for it in o.iterations], "final_verification": o.final_verification.to_dict() if o.final_verification else None}


class TestEnergy(unittest.TestCase):
    def test_energy_model_and_violation(self):
        m = mk_mission(max_energy=130, energy_k_v=1.0, energy_k_wp=2.0)
        p = mk_plan([("A", 30, 0, 60), ("B", 70, 0, 60)], dest=(100, 0, 60))
        r = verify(m, p)                       # distance 100 + 1.0*ascent 60 + 2*2 = 164
        self.assertAlmostEqual(r.energy, 164.0)
        self.assertAlmostEqual(r.ascent, 60.0)
        e = [v for v in r.violations if v.type == "ENERGY_EXCEEDED"]
        self.assertEqual(len(e), 1)
        self.assertEqual(e[0].family, "energy")
        self.assertAlmostEqual(e[0].evidence["excess"], 34.0)
        low = verify(m, mk_plan([("A", 30, 0, 40), ("B", 70, 0, 40)], dest=(100, 0, 40)))   # ascent 40 -> 100+40+4=144 still > 130
        self.assertIn("ENERGY_EXCEEDED", [v.type for v in low.violations])
        ok = verify(mk_mission(max_energy=150, energy_k_v=1.0), p.__class__.from_dict(mk_plan([("A", 30, 0, 40), ("B", 70, 0, 40)], dest=(100, 0, 40)).to_dict()))
        self.assertTrue(ok.valid)             # 100 + 40 = 140 <= 150

    def test_altitude_energy_tradeoff_and_start_altitude(self):
        m = mk_mission(max_energy=140, energy_k_v=1.0, start_altitude=60)
        r = verify(m, mk_plan(GOOD))           # already at 60 m: no climb needed
        self.assertAlmostEqual(r.ascent, 0.0)
        self.assertTrue(r.valid)

    def test_inactive_family_ignored(self):
        m = mk_mission(max_distance=10, active=["altitude", "no_fly_zone", "ordering"])
        self.assertTrue(verify(m, mk_plan(GOOD)).valid)

    def test_severity_and_hints_and_feedback(self):
        m = mk_mission(no_fly_zones=[Z])
        r = verify(m, mk_plan([("A", 30, 0, 200), ("B", 70, 0, 60)]))
        sev = {v.type: v.severity for v in r.violations}
        self.assertEqual(sev["ALTITUDE_ABOVE_MAX"], "high")
        self.assertIn(sev["NFZ_SEGMENT"], ("medium", "high"))
        fb = build_feedback(m, mk_plan([("A", 30, 0, 200), ("B", 70, 0, 60)]), r, "evidence", "targeted")
        self.assertEqual(fb.payload["n_violations"], len(r.violations))
        self.assertTrue(all("severity" in v and v["suggested_fix"] for v in fb.payload["violations"]))
        loc = build_feedback(m, mk_plan(GOOD), verify(m, mk_plan(GOOD)), "localized", "regenerate")
        self.assertIn("by_severity", loc.payload)

    def test_obstacle_category_in_evidence(self):
        ob = NoFlyZone("T1", "circle", center=[50, 0], radius=5, description="radio tower", category="obstacle")
        r = verify(mk_mission(no_fly_zones=[ob]), mk_plan(GOOD))
        self.assertEqual(r.violations[0].evidence["zone_category"], "obstacle")


class TestAStar(unittest.TestCase):
    def test_astar_valid_detour(self):
        m = mk_mission(max_distance=200, no_fly_zones=[NoFlyZone("Z1", "circle", center=[50, 0], radius=8)])
        p = astar_plan(m)
        self.assertIsNotNone(p)
        self.assertTrue(verify(m, p).success)
        self.assertGreater(len(p.waypoints), 2)

    def test_astar_respects_energy_by_low_altitude_and_none_for_blocked(self):
        m = mk_mission(max_distance=200, max_energy=140, energy_k_v=1.0)
        self.assertTrue(verify(m, astar_plan(m)).success)
        wall = [NoFlyZone("W1", "polygon", vertices=[[45, -500], [55, -500], [55, 500], [45, 500]])]
        mw = mk_mission(max_distance=5000, no_fly_zones=wall)
        pw = astar_plan(mw, resolution=60)                      # must go around the wall's far end: long but valid
        self.assertTrue(verify(mw, pw).success)
        self.assertGreater(verify(mw, pw).route_distance, 900)


class TestRepairDynamics(unittest.TestCase):
    def run_script(self, replies, **kw):
        m = mk_mission(no_fly_zones=[Z], max_distance=200)
        return run_condition(m, LLMPlanner(ScriptedLLM(replies)), LoopConfig(**kw), 0)

    BAD_ALT = plan_json([("A", 30, 0, 60), ("B", 70, 0, 120)], (100, 0, 60))                     # altitude(B) + segment(Z1)
    SEG_ONLY = plan_json([("A", 30, 0, 60), ("B", 70, 0, 60)], (100, 0, 60))                      # segment(Z1)
    ALT_ONLY = plan_json([("A", 30, 0, 60), ("W1", 50, 9, 60), ("B", 70, 0, 120)], (100, 0, 60))  # altitude(B) only
    GOODP = plan_json([("A", 30, 0, 60), ("W1", 50, 9, 60), ("B", 70, 0, 60)], (100, 0, 60))

    def test_recurrence_and_cascade(self):
        # round0: alt+seg ; round1: seg only (alt fixed) ; round2: alt only (seg fixed, alt RECURS = recurrence, new alt after seg fix = cascade)
        o = self.run_script([self.BAD_ALT, self.SEG_ONLY, self.ALT_ONLY, self.GOODP], max_iterations=3, regression_guard=False)
        d = repair_dynamics(rec_from(o))
        self.assertTrue(d["dyn_recurrence"])
        self.assertEqual(d["dyn_recurred_types"], ["ALTITUDE_ABOVE_MAX"])
        self.assertTrue(d["dyn_cascade"])
        self.assertIn("no_fly_zone->altitude", d["dyn_cascade_pairs"])
        self.assertEqual(d["dyn_iters_to_success"], 3)

    def test_oscillation_and_stuck(self):
        o = self.run_script([self.SEG_ONLY, self.ALT_ONLY, self.SEG_ONLY, self.ALT_ONLY], max_iterations=3, regression_guard=False)
        d = repair_dynamics(rec_from(o))
        self.assertTrue(d["dyn_oscillation"])
        o = self.run_script([self.SEG_ONLY] * 4, max_iterations=3)
        d = repair_dynamics(rec_from(o))
        self.assertTrue(d["dyn_stuck"])
        self.assertFalse(d["dyn_cascade"])

    def test_clean_convergence_has_no_flags(self):
        o = self.run_script([self.BAD_ALT, self.GOODP], max_iterations=3)
        d = repair_dynamics(rec_from(o))
        self.assertEqual((d["dyn_recurrence"], d["dyn_cascade"], d["dyn_oscillation"], d["dyn_iters_to_success"]), (False, False, False, 1))


if __name__ == "__main__":
    unittest.main()

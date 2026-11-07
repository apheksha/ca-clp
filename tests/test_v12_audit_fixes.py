"""Regression tests for the v12 audit fixes (each test fails on the v11 code and passes after the fix)."""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.helpers import mk_mission, mk_plan
from src.models.schemas import NoFlyZone, RequiredWaypoint
from src.verifier.verifier import verify
from src.repair.direct_patch import apply_full
from src.repair.loop import LoopConfig, run_condition
from src.planner.planner import PlannerOutput


def _zone_mission(**kw):
    z = NoFlyZone("Z", "circle", center=[50, 0], radius=10).validate()
    return mk_mission(no_fly_zones=[z], max_distance=400, **kw)


class DirectFullSeesPreservationViolations(unittest.TestCase):
    """v11 bug: apply_full only inspected res.violations, but MISSING/MOVED/DESTINATION_CHANGED are preservation violations, so they never triggered the rebuild."""

    def test_missing_required_waypoint_is_rebuilt(self):
        m = _zone_mission(ordering=[])
        bad = mk_plan([("A", 30, 0, 60)])                       # B dropped; no ORDER violation, so only a preservation violation is present
        r = verify(m, bad)
        self.assertEqual({v.type for v in r.preservation_violations}, {"MISSING_REQUIRED_WAYPOINT"})
        fixed, log = apply_full(m, bad, r)
        self.assertTrue(log.get("rebuilt_skeleton"))
        self.assertIn("MISSING_REQUIRED_WAYPOINT", log["structural_violations"])
        self.assertTrue(verify(m, fixed).success)

    def test_moved_required_waypoint_and_changed_destination_are_rebuilt(self):
        m = _zone_mission(ordering=[])
        bad = mk_plan([("A", 30, 25, 60), ("B", 70, 0, 60)], dest=(100, 40, 60))
        types = {v.type for v in verify(m, bad).all_violations()}
        self.assertTrue({"REQUIRED_WAYPOINT_MOVED", "DESTINATION_CHANGED"} <= types)
        fixed, log = apply_full(m, bad, verify(m, bad))
        self.assertTrue(log.get("rebuilt_skeleton"))
        self.assertTrue(verify(m, fixed).success)

    def test_mission_id_mismatch_alone_does_not_discard_the_plan(self):
        m = _zone_mission(ordering=[])
        good_geometry = mk_plan([("A", 30, 0, 60), ("W1", 50, 14, 60), ("B", 70, 0, 60)], mid="OTHER")
        r = verify(m, good_geometry)
        self.assertEqual({v.type for v in r.all_violations()}, {"MISSION_ID_MISMATCH"})
        fixed, log = apply_full(m, good_geometry, r)
        self.assertFalse(log.get("rebuilt_skeleton", False))       # the detour waypoint W1 is kept
        self.assertIn("W1", [w.id for w in fixed.waypoints])
        self.assertTrue(verify(m, fixed).success)

    def test_direct_full_condition_solves_dropped_waypoint_without_llm(self):
        m = _zone_mission(ordering=[])
        bad = mk_plan([("A", 30, 0, 60)])
        o = run_condition(m, None, LoopConfig(name="G", kind="direct_full", feedback_level="prescriptive", max_iterations=3), 0, initial=PlannerOutput(bad, [], None))
        self.assertEqual(o.status, "SUCCESS")
        self.assertEqual(sum(len(it.calls) for it in o.iterations), 0)


class LLMValueClassNames(unittest.TestCase):
    def test_structural_class_uses_verifier_type_names(self):
        from src.evaluation.llm_value import _classes
        self.assertEqual(_classes({"DESTINATION_CHANGED"}), "structure/order")
        self.assertEqual(_classes({"MISSION_ID_MISMATCH"}), "structure/order")
        self.assertEqual(_classes({"NFZ_SEGMENT", "DISTANCE_EXCEEDED"}), "zone+budget")


class EvaluateWritesPerRunFolder(unittest.TestCase):
    """v11 bug: `run.py evaluate <root>/raw/<run>` wrote into <root>/processed/ without a run subfolder, so a second evaluation overwrote the first."""

    def test_targets(self):
        sys.path.insert(0, ROOT)
        import importlib
        run = importlib.import_module("run")
        with tempfile.TemporaryDirectory() as t:
            raw = os.path.join(t, "outputs", "raw", "real_x")
            os.makedirs(raw)
            proc, fig, sub = run.evaluation_targets(raw)
            self.assertEqual(os.path.normpath(proc), os.path.normpath(os.path.join(t, "outputs", "processed")))
            self.assertTrue(sub)
            fin = os.path.join(t, "outputs", "runs", "m_full_1")
            os.makedirs(os.path.join(fin, "raw", "real_y"))
            open(os.path.join(fin, "state.json"), "w").write("{}")
            proc2, _, sub2 = run.evaluation_targets(os.path.join(fin, "raw", "real_y"))
            self.assertEqual(os.path.normpath(proc2), os.path.normpath(os.path.join(fin, "processed")))
            self.assertFalse(sub2)

    def test_two_runs_do_not_overwrite_each_other(self):
        from src.experiments.runner import run_experiment
        with tempfile.TemporaryDirectory() as t:
            raw_root = os.path.join(t, "outputs", "raw")
            d1 = run_experiment("config/default.yaml", "config/experiments.yaml", "mock", "mock-heuristic-v1", raw_root=raw_root, seeds=[0], limit=2, conditions=["A_llm_only"], progress=lambda *_: None)
            d2 = run_experiment("config/default.yaml", "config/experiments.yaml", "mock", "mock-heuristic-v1", raw_root=raw_root, seeds=[0], limit=3, conditions=["A_llm_only"], progress=lambda *_: None)
            for d in (d1, d2):
                r = subprocess.run([sys.executable, "run.py", "evaluate", d], cwd=ROOT, capture_output=True, text=True, timeout=300)
                self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            for d in (d1, d2):
                rid = os.path.basename(d)
                self.assertTrue(os.path.exists(os.path.join(t, "outputs", "processed", rid, "summary_by_condition.csv")), rid)
            self.assertFalse(os.path.exists(os.path.join(t, "outputs", "processed", "summary_by_condition.csv")))


class ModelFreeSessionsAndProvenance(unittest.TestCase):
    """v12: provider 'none' (never generates text) + manifest provenance kept on resume (v11 overwrote env/code fingerprint and the condition list)."""

    def test_none_client_cannot_generate(self):
        from src.planner.llm import make_client, LLMError
        c = make_client({"provider": "none", "model": "none"})
        self.assertFalse(c.is_mock)
        with self.assertRaises(LLMError):
            c.generate("s", "u", 0.0, 10, 0)

    def test_resume_adds_direct_controls_and_keeps_provenance(self):
        import json
        from src.experiments.runner import run_experiment
        from src.utils.io import read_jsonl
        with tempfile.TemporaryDirectory() as t:
            d = run_experiment("config/default.yaml", "config/experiments.yaml", "mock", "mock-heuristic-v1", raw_root=t, seeds=[0], limit=4,
                               conditions=["A_llm_only", "C_prescriptive"], progress=lambda *_: None)
            before = json.load(open(os.path.join(d, "manifest.json")))
            n0 = len(read_jsonl(os.path.join(d, "runs.jsonl")))
            run_experiment("config/default.yaml", "config/experiments.yaml", "none", "none", seeds=[0], limit=4, resume_dir=d,
                           conditions=["G_direct_patch", "G_direct_full_witness"], progress=lambda *_: None)
            after = json.load(open(os.path.join(d, "manifest.json")))
            recs = read_jsonl(os.path.join(d, "runs.jsonl"))
            self.assertEqual(len(recs), n0 + 8)
            self.assertEqual(after["model"], before["model"])                         # the run still belongs to the model that made the first plans
            self.assertEqual(after["env"]["code_sha256"], before["env"]["code_sha256"])
            self.assertEqual(len(after["env_resumes"]), 1)
            self.assertTrue({"A_llm_only", "C_prescriptive", "G_direct_patch", "G_direct_full_witness"} <= set(after["conditions"]))
            g = [r for r in recs if r["condition"].startswith("G_")]
            self.assertTrue(all(sum(len(it["calls"]) for it in r["iterations"][1:]) == 0 for r in g))
            # the shared first plan is the one the A/C records used
            a = {(r["scenario_id"]): r["initial_plan"] for r in recs if r["condition"] == "A_llm_only"}
            self.assertTrue(all(r["initial_plan"] == a[r["scenario_id"]] for r in g))

    def test_none_refuses_model_conditions(self):
        from src.experiments.runner import run_experiment
        with tempfile.TemporaryDirectory() as t:
            d = run_experiment("config/default.yaml", "config/experiments.yaml", "mock", "mock-heuristic-v1", raw_root=t, seeds=[0], limit=1,
                               conditions=["A_llm_only"], progress=lambda *_: None)
            with self.assertRaises(SystemExit):
                run_experiment("config/default.yaml", "config/experiments.yaml", "none", "none", seeds=[0], limit=1, resume_dir=d, conditions=["C_full"], progress=lambda *_: None)




class SpecExtraction(unittest.TestCase):
    """v12: NL -> spec -> classical planner -> verify against the true mission."""

    def _m(self, sid="real_chicago_00", folder="benchmark/real"):
        import json
        from src.models.schemas import Mission
        return Mission.from_dict(json.load(open(os.path.join(ROOT, folder, sid + ".json")))["mission"])

    def _truth_reply(self, m):
        import json
        from src.nl.spec_extraction import truth_spec
        t = truth_spec(m)
        t["zones"] = [{"id": z.id, "kind": z.kind, "center": z.center, "radius": z.radius, "vertices": z.vertices, "z_min": z.z_min, "z_max": z.z_max} for z in t["zones"]]
        return t

    def test_prompt_carries_every_number_needed(self):
        from src.nl.spec_extraction import spec_prompt
        m = self._m()
        p = spec_prompt(m)
        for z in m.no_fly_zones:
            self.assertIn(z.id, p)
        self.assertIn(f"{m.max_altitude:g}", p)

    def test_correct_reply_succeeds(self):
        import json
        from src.nl.spec_extraction import evaluate_reply
        m = self._m()
        r = evaluate_reply("```json\n" + json.dumps(self._truth_reply(m)) + "\n```", m)
        self.assertTrue(r["parsed"] and r["all_fields_correct"] and r["end_to_end_success"])

    def test_shrunken_zone_is_caught_by_the_verifier(self):
        import json
        from src.nl.spec_extraction import evaluate_reply
        m = self._m()
        t = self._truth_reply(m)
        blocked = [z for z in m.no_fly_zones if any(z.intersects_segment(a, b) for a, b in zip([m.start_xy] + [r.xy for r in m.required_waypoints], [r.xy for r in m.required_waypoints] + [m.dest_xy]))]
        self.assertTrue(blocked)
        for z in t["zones"]:
            if z["id"] == blocked[0].id:
                z["radius"] = z["radius"] * 0.3
        r = evaluate_reply(json.dumps(t), m)
        self.assertFalse(r["fields"]["zones"])
        self.assertFalse(r["end_to_end_success"])
        self.assertTrue({"NFZ_SEGMENT", "NFZ_WAYPOINT"} & set(r["violations_under_truth"]), r["violations_under_truth"])

    def test_missing_altitude_band_fails_end_to_end(self):
        import json
        from src.nl.spec_extraction import evaluate_reply
        m = self._m()
        t = self._truth_reply(m)
        t["min_altitude"] = t["max_altitude"] = None
        r = evaluate_reply(json.dumps(t), m)
        self.assertFalse(r["fields"]["altitude_band"])
        self.assertFalse(r["end_to_end_success"])

    def test_unparseable_reply(self):
        from src.nl.spec_extraction import evaluate_reply
        r = evaluate_reply("Sure! The start is at the depot.", self._m())
        self.assertFalse(r["parsed"])
        self.assertFalse(r["end_to_end_success"])

    def test_inactive_families_are_not_expected(self):
        from src.nl.spec_extraction import truth_spec
        m = self._m("combo_nfz_000", "benchmark/generated")
        t = truth_spec(m)
        self.assertIsNone(t["min_altitude"])
        self.assertIsNone(t["max_distance"])
        self.assertTrue(t["zones"])


class AltitudeMargin(unittest.TestCase):
    def test_lift_moves_into_band_and_is_reverified_by_caller(self):
        from src.simulator.study import lift_altitudes
        m = mk_mission(min_altitude=40, max_altitude=80)
        p = mk_plan([("A", 30, 0, 40), ("B", 70, 0, 80)], dest=(100, 0, 40))
        q = lift_altitudes(m, p, 10.0)
        self.assertEqual([w.altitude for w in q.ordered()] + [q.destination.altitude], [50.0, 70.0, 50.0])
        self.assertTrue(verify(m, q).success)
        narrow = mk_mission(min_altitude=40, max_altitude=46)
        q2 = lift_altitudes(narrow, mk_plan([("A", 30, 0, 40), ("B", 70, 0, 46)], dest=(100, 0, 40)), 10.0)
        self.assertTrue(all(abs(a - 43.0) < 1e-9 for a in [w.altitude for w in q2.ordered()] + [q2.destination.altitude]))   # band narrower than 2*margin -> mid-band


class EdgeClearance(unittest.TestCase):
    def test_default_witness_unchanged_and_clearance_kept_along_legs(self):
        import json
        from src.models.schemas import Mission
        from src.benchmark.feasibility import assess
        from src.utils import geometry as G
        for sid in ("hard_000", "medium_003", "easy_005"):
            rec = json.load(open(os.path.join(ROOT, "benchmark/generated", sid + ".json")))
            m = Mission.from_dict(rec["mission"])
            f0 = assess(m)
            self.assertAlmostEqual(f0.witness_distance, rec["metadata"]["witness_distance"], places=2)      # default behaviour = the published benchmark
            f2 = assess(m, edge_clearance=2.0)
            if f2.status != "FEASIBLE":
                continue
            pts = [p for _, p in f2.witness_plan.route_points(m)]
            for z in m.no_fly_zones:
                if z.kind != "circle":
                    continue
                dmin = min(G.closest_on_segment(z.centroid(), a, b)[2] for a, b in zip(pts, pts[1:])) - z.radius
                self.assertGreaterEqual(dmin, 2.0 - 1e-6, f"{sid}/{z.id}")


class OSMOverpassIds(unittest.TestCase):
    """Overpass-turbo GeoJSON exports use feature ids such as "way/37931109"; the loader used them verbatim and every zone failed id validation."""

    def test_overpass_turbo_ids_load(self):
        import json
        from src.geodata.projection import LocalFrame
        from src.geodata.osm import load_osm_buildings
        ring = [[77.5600, 13.0300], [77.5603, 13.0300], [77.5603, 13.0303], [77.5600, 13.0303], [77.5600, 13.0300]]
        fc = {"type": "FeatureCollection", "features": [
            {"type": "Feature", "id": "way/37931109", "properties": {"building": "university", "building:levels": "6"}, "geometry": {"type": "Polygon", "coordinates": [ring]}},
            {"type": "Feature", "id": "relation/1234567890123456789012345678901", "properties": {"building": "yes"}, "geometry": {"type": "Polygon", "coordinates": [[[x + 0.001, y] for x, y in ring]]}}]}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "b.geojson")
            json.dump(fc, open(p, "w"))
            zs = load_osm_buildings(p, LocalFrame(13.03, 77.56))
        self.assertEqual(len(zs), 2)
        self.assertEqual(zs[0].id, "BLD_way_37931109")
        self.assertLessEqual(len(zs[1].id), 32)
        self.assertAlmostEqual(zs[0].z_max, 18.0)      # 6 levels x 3 m
        self.assertAlmostEqual(zs[1].z_max, 10.0)      # default height



class EvaluateUnknownDifficulty(unittest.TestCase):
    """make_figures dropped difficulty labels it did not know (e.g. blr_easy of the Bengaluru set) and then crashed on an empty plot."""

    def test_new_difficulty_label(self):
        import json, shutil
        from src.experiments.runner import run_experiment
        from src.evaluation.report import evaluate_run
        with tempfile.TemporaryDirectory() as t:
            sdir = os.path.join(t, "scen")
            os.makedirs(sdir)
            src = sorted(f for f in os.listdir(os.path.join(ROOT, "benchmark", "generated")) if f.startswith("easy_"))[:2]
            for f in src:
                d = json.load(open(os.path.join(ROOT, "benchmark", "generated", f)))
                d["difficulty"] = "blr_easy"
                json.dump(d, open(os.path.join(sdir, f), "w"))
            run = run_experiment("config/default.yaml", "config/experiments.yaml", "none", "none", raw_root=os.path.join(t, "outputs", "raw"), scenario_dir=sdir,
                                 seeds=[0], conditions=["E_classical"], progress=lambda *_: None)
            out = evaluate_run(run, processed_root=os.path.join(t, "outputs", "processed"), figures_root=os.path.join(t, "outputs", "figures"))
            self.assertTrue(os.path.exists(os.path.join(out, "summary_by_condition.csv")))


if __name__ == "__main__":
    unittest.main()

"""Ten hand-written demonstration cases. Plans/replies are fixtures from ScriptedLLM (NOT LLM output);
verification, feedback, loop and plots are the real system."""
from __future__ import annotations
import json
import os
from src.models.schemas import Mission, RequiredWaypoint, NoFlyZone
from src.benchmark.generator import nl_description
from src.planner.planner import LLMPlanner
from src.planner.scripted import ScriptedLLM, plan_json
from src.repair.loop import LoopConfig, run_condition
from src.simulator.visualize import plot_history


def base_mission(**kw) -> Mission:
    d = dict(mission_id="DEMO", description="", start=[0, 50], destination=[100, 50],
             required_waypoints=[RequiredWaypoint("A", 30, 50), RequiredWaypoint("B", 70, 50)],
             min_altitude=40, max_altitude=80, max_distance=140,
             no_fly_zones=[NoFlyZone("Z1", "circle", center=[50, 50], radius=8, description="restricted area 1"),
                           NoFlyZone("Z2", "polygon", vertices=[[40, 20], [60, 20], [60, 30], [40, 30]], description="restricted area 2")],
             ordering=[["A", "B"]])
    d.update(kw)
    m = Mission(**d).validate()
    m.description = nl_description(m)
    return m


GOOD = plan_json([("A", 30, 50, 60), ("W1", 50, 62, 60), ("B", 70, 50, 60)], (100, 50, 60))
CASES = {
    "01_valid": dict(replies=[GOOD]),
    "02_altitude_violation": dict(replies=[plan_json([("A", 30, 50, 95), ("W1", 50, 62, 60), ("B", 70, 50, 20)], (100, 50, 60)), GOOD]),
    "03_waypoint_in_nfz": dict(replies=[plan_json([("A", 30, 50, 60), ("W1", 52, 52, 60), ("B", 70, 50, 60)], (100, 50, 60)), GOOD]),
    "04_segment_crosses_nfz": dict(replies=[plan_json([("A", 30, 50, 60), ("B", 70, 50, 60)], (100, 50, 60)), GOOD]),
    "05_distance_violation": dict(replies=[plan_json([("A", 30, 50, 60), ("W1", 50, 62, 60), ("W2", 50, 95, 60), ("W3", 55, 62, 60), ("B", 70, 50, 60)], (100, 50, 60)), GOOD]),
    "06_ordering_violation": dict(replies=[plan_json([("B", 70, 50, 60), ("W1", 50, 62, 60), ("A", 30, 50, 60)], (100, 50, 60)), GOOD]),
    "07_multiple_violations": dict(replies=[plan_json([("B", 70, 50, 99), ("A", 30, 50, 60)], (100, 50, 60)), GOOD], max_iterations=3),
    "08_multi_iteration_repair": dict(replies=[
        plan_json([("B", 70, 50, 99), ("A", 30, 50, 60)], (100, 50, 60)),                                   # 3+ violations
        plan_json([("A", 30, 50, 60), ("B", 70, 50, 60)], (100, 50, 60)),                                   # only segment crossing left
        GOOD], max_iterations=3),
    "09_repair_regression": dict(replies=[
        plan_json([("A", 30, 50, 60), ("B", 70, 50, 60)], (100, 50, 60)),                                   # 1 violation
        plan_json([("B", 70, 50, 99), ("W1", 50, 55, 60), ("A", 30, 50, 60)], (100, 50, 60)),               # regression
        GOOD], max_iterations=3, regression_guard=False),
    "10_infeasible": dict(replies=[plan_json([("A", 30, 50, 60), ("B", 70, 50, 60)], (100, 50, 60))], mission=dict(max_distance=80)),
}


def run_demos(out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for name, c in CASES.items():
        m = base_mission(**c.get("mission", {}))
        planner = LLMPlanner(ScriptedLLM(c["replies"]))
        cfg = LoopConfig(max_iterations=c.get("max_iterations", 3), regression_guard=c.get("regression_guard", True))
        o = run_condition(m, planner, cfg, seed=0)
        rec = {"case": name, "llm": "ScriptedLLM fixture (not a language model)", "status": o.status,
               "iterations": [it.__dict__ for it in o.iterations], "final_verification": o.final_verification.to_dict() if o.final_verification else None,
               "feasibility": o.feasibility}
        with open(os.path.join(out_dir, name + ".json"), "w") as f:
            json.dump(rec, f, indent=1)
        plot_history(m, rec["iterations"], os.path.join(out_dir, name + ".png"), suptitle=f"{name}  -> {o.status}  (scripted fixture)")
        summary[name] = {"status": o.status, "outcomes": [i.outcome for i in o.iterations], "scores": [i.score for i in o.iterations]}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary

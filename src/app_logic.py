"""UI-independent session logic (used by app.py and tests)."""
from __future__ import annotations
from src.models.schemas import Mission
from src.planner.llm import make_client
from src.planner.planner import LLMPlanner
from src.repair.loop import LoopConfig, run_condition
from src.simulator.visualize import plot_history


def run_session(mission: Mission, provider_cfg: dict, loop_cfg: LoopConfig, seed: int = 0, client=None, on_event=None) -> dict:
    client = client or make_client(provider_cfg)
    if on_event and hasattr(client, "on_wait"):
        client.on_wait = lambda w, k, why: on_event({"type": "rate_limit", "wait": w, "retry": k, "why": why})
    planner = LLMPlanner(client, provider_cfg.get("temperature", 0.0), provider_cfg.get("max_tokens", 2000), provider_cfg.get("max_parse_retries", 2))
    from src.benchmark.oracle_cache import assess_cached       # v12: cached oracle (terrain missions take minutes otherwise)
    o = run_condition(mission, planner, loop_cfg, seed, on_event=on_event, feasibility=assess_cached(mission))
    its = [it.__dict__ for it in o.iterations]
    return {"is_mock": client.is_mock, "model": client.model, "provider": client.name, "status": o.status, "released": o.released, "iterations": its,
            "initial_plan": o.initial.plan.to_dict() if o.initial.plan else None,
            "final_plan": o.final_plan.to_dict() if o.final_plan else None,
            "final_verification": o.final_verification.to_dict() if o.final_verification else None,
            "feasibility": o.feasibility, "history_fig": plot_history(mission, its),
            "n_llm_calls": sum(len(i["calls"]) for i in its), "wall_time_s": o.wall_time_s}


def run_dynamic_session(mission: Mission, change: str, progress: float, provider_cfg: dict, loop_cfg: LoopConfig, seed: int = 0, client=None, on_event=None) -> dict | None:
    """Objective 4 for the UI: fly `progress` of a valid plan, apply `change`, then repair the REMAINING route with the closed loop."""
    import random
    from src.dynamic.changes import comfortable_plan, execution_state, build_dynamic_mission
    from src.benchmark.feasibility import assess
    from src.planner.planner import PlannerOutput
    plan = comfortable_plan(mission)
    if plan is None:
        return None
    st = execution_state(mission, plan, progress)
    built = build_dynamic_mission(mission, plan, st, change, random.Random(f"{mission.mission_id}|{change}|{progress}"))
    if built is None:
        return None
    new, rem, info = built
    client = client or make_client(provider_cfg)
    if on_event and hasattr(client, "on_wait"):
        client.on_wait = lambda w, k, why: on_event({"type": "rate_limit", "wait": w, "retry": k, "why": why})
    planner = LLMPlanner(client, provider_cfg.get("temperature", 0.0), provider_cfg.get("max_tokens", 2000), provider_cfg.get("max_parse_retries", 2))
    o = run_condition(new, planner, loop_cfg, seed, initial=PlannerOutput(rem, [], None), feasibility=assess(new), on_event=on_event)
    its = [it.__dict__ for it in o.iterations]
    return {"is_mock": client.is_mock, "status": o.status, "description": new.description, "state": st.to_dict(), "info": info, "iterations": its,
            "history_fig": plot_history(new, its), "final_verification": o.final_verification.to_dict() if o.final_verification else None,
            "n_llm_calls": sum(len(i["calls"]) for i in its), "feasibility": o.feasibility}

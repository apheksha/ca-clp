"""All metrics are pure functions of the raw per-run record (ExperimentResult dict)."""
from __future__ import annotations
from src.evaluation.repair_dynamics import repair_dynamics
FAMILIES = ["altitude", "no_fly_zone", "distance", "ordering", "energy", "mission_preservation"]


def _calls(rec):
    return [c for it in rec["iterations"] for c in it["calls"]]


def run_metrics(rec: dict, pricing: dict | None = None) -> dict:
    its = rec["iterations"]
    v0 = its[0]["verification"]
    fv = rec["final_verification"]
    calls = _calls(rec)
    pt = sum(c.get("prompt_tokens") or 0 for c in calls)
    ct = sum(c.get("completion_tokens") or 0 for c in calls)
    cost = None
    per_call = [c.get("cost_usd") for c in calls]
    if calls and all(x is not None for x in per_call) and not rec["is_mock"]:
        cost = sum(per_call)                      # registry-based, per call (config/models.yaml)
    elif pricing and pricing.get("input_per_mtok") is not None and not rec["is_mock"]:
        cost = pt / 1e6 * pricing["input_per_mtok"] + ct / 1e6 * pricing["output_per_mtok"]   # legacy global pricing block
    req_lat = sum((c.get("request_latency_s") or c.get("latency_s") or 0.0) for c in calls)
    rl_wait = sum((c.get("rate_limit_wait_s") or 0.0) for c in calls)
    # Enforce strict final_success = final_valid=True AND released=True
    rel = bool(rec["metrics"].get("released", False)) if "metrics" in rec else False
    fv_valid = bool(fv and fv.get("valid", False))
    fv_success = bool(fv and fv.get("success", False))
    fin_success = bool(fv_valid and rel and fv_success)

    m = {
        "initial_plan_parsed": v0 is not None,
        "initial_valid": bool(v0 and v0["valid"]),
        "initial_success": bool(v0 and v0["success"]),
        "final_valid": fv_valid,
        "final_success": fin_success,
        "released": rel,
        # three mutually exclusive run outcomes (reviewer: report release safety, mission completion and rejection separately)
        #   completed  = released AND valid AND mission-preserving        (== final_success)
        #   unsafe     = released AND NOT (valid AND mission-preserving)  (the gate failed / no gate)
        #   withheld   = not released (rejected: no verified plan within budget, or infeasible mission correctly refused)
        "outcome_completed": fin_success,
        "outcome_unsafe_release": bool(rel and not fin_success),
        "outcome_withheld": not rel,
        "init_extraction_error": bool(v0 and v0.get("preservation_violations")),   # initial plan lost / moved a required waypoint or the destination (constraint-extraction slip)
        "init_n_violations": v0["score"] if v0 else None,
        "final_n_violations": fv["score"] if fv else None,
        "repair_needed": bool(v0 and not v0["success"]),
        "repair_success": bool(v0 and not v0["success"] and fin_success),
        "n_repair_iterations": len(its) - 1,
        "n_regressions": sum(1 for it in its[1:] if it["regression"]),
        "any_regression": any(it["regression"] for it in its[1:]),
        "n_new_violations_introduced": sum(len(it["new_violations"]) for it in its[1:]),
        "failure": rec["final_status"] == "FAILURE",
        "final_status": rec["final_status"],
        "route_distance_initial": v0.get("route_distance") if v0 else None,
        "route_distance_final": fv.get("route_distance") if fv else None,
        "waypoint_mods_total": sum(it.get("diff", {}).get("n_changes", 0) for it in its[1:] if it.get("diff")),
        "collateral_edits": sum(it.get("diff", {}).get("collateral_edits", 0) for it in its[1:] if it.get("diff")),
        "n_llm_calls": len(calls),
        "n_parse_errors": sum(1 for c in calls if c.get("parse_error")),
        "n_format_normalized": sum(1 for c in calls if c.get("normalized")),
        "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost, "request_latency_s": req_lat, "rate_limit_wait_s": rl_wait,
        "wall_time_s": rec.get("timing", {}).get("wall_time_s", 0.0),
        "verification_time_s": sum((it.get("verification") or {}).get("verification_time_s", 0) for it in its),
    }
    m["route_energy_final"] = fv.get("energy") if fv else None
    m["flight_time_final_s"] = fv.get("flight_time_s") if fv else None
    we = rec.get("scenario_meta", {}).get("witness_energy")
    m["energy_overhead_vs_witness"] = max(0.0, fv["energy"] / we - 1) if (we and fv and fv.get("energy")) else None
    m["init_violation_types"] = [v["type"] for v in (v0.get("violations", []) + v0.get("preservation_violations", []))] if v0 else []
    m["final_violation_types"] = [v["type"] for v in (fv.get("violations", []) + fv.get("preservation_violations", []))] if fv else []
    m.update(repair_dynamics(rec))
    # distance_overhead_vs_witness: only computed when the plan is valid AND released (fin_success).
    # When valid, the LLM route is a legal path whose distance is >= the witness (optimal feasible
    # route), so overhead >= 0 by construction; max(0,...) absorbs floating-point noise only.
    # For invalid or unreleased plans the overhead is undefined → None.
    wd = rec.get("scenario_meta", {}).get("witness_distance")
    rd = m["route_distance_final"]
    if wd and rd is not None and fin_success:
        m["distance_overhead_vs_witness"] = max(0.0, rd / wd - 1.0)
    else:
        m["distance_overhead_vs_witness"] = None
    for f in FAMILIES:
        m[f"init_viol_{f}"] = bool(v0 and not v0.get("per_constraint", {}).get(f, {"satisfied": True})["satisfied"])
        m[f"final_viol_{f}"] = bool(fv and not fv.get("per_constraint", {}).get(f, {"satisfied": True})["satisfied"])
    return m

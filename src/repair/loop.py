"""Conditions (baselines + closed-loop variants) and the repair loop with outcome/regression classification."""
from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from src.models.schemas import Mission, FlightPlan, RepairIteration, VerificationResult
from src.verifier.verifier import verify
from src.feedback.feedback import build_feedback, repair_prompt
from src.planner.planner import LLMPlanner, PlannerOutput
from src.repair.diff import plan_diff, collateral_edits
from src.benchmark.feasibility import assess, Feasibility
from src.repair.direct_patch import apply_prescriptive, apply_full

# per-iteration outcomes
SUCCESS, PARTIAL, NO_CHANGE, LATERAL, REGRESSION, PARSE_FAIL = "SUCCESS", "PARTIAL_REPAIR", "NO_CHANGE", "LATERAL", "REGRESSION", "PARSE_FAILURE"


RESAMPLE_TEMPERATURE = 0.7   # the compute-matched baseline MUST sample: at temperature 0 'fresh' generations are (nearly) identical


@dataclass
class LoopConfig:
    name: str = "C_full"
    kind: str = "closed_loop"          # llm_only | llm_verify | resample | closed_loop | direct_patch | direct_full (LLM-free controls; no repair LLM calls) | classical | astar
    feedback_level: str = "evidence"   # binary | localized | evidence | prescriptive
    repair_mode: str = "targeted"      # regenerate | targeted
    regression_guard: bool = True
    max_iterations: int = 3
    feasibility_precheck: bool = True
    temperature: float | None = None   # sampling temperature for fresh generations (resample); None -> RESAMPLE_TEMPERATURE
    resample_nonce: bool = False       # resample only: add a per-sample id to the prompt so cached/identical generations are avoided
    witness_fallback: bool = False     # direct_full only: if deterministic repair fails, release the oracle's verified witness route (pure direct execution of the verifier's route)

    def to_dict(self):
        return asdict(self)


@dataclass
class LoopOutcome:
    initial: PlannerOutput
    initial_verification: dict | None
    iterations: list[RepairIteration]
    final_plan: FlightPlan | None
    final_verification: VerificationResult | None
    status: str
    released: bool
    feasibility: dict | None
    oracle_inconsistent: bool = False
    wall_time_s: float = 0.0


def classify(prev: VerificationResult, new: VerificationResult) -> str:
    if new.success:
        return SUCCESS
    if new.score < prev.score:
        return PARTIAL
    if new.score > prev.score:
        return REGRESSION
    ps, ns = {v.key() for v in prev.all_violations()}, {v.key() for v in new.all_violations()}
    return NO_CHANGE if ps == ns else LATERAL


def run_status(initial: VerificationResult | None, final: VerificationResult | None, any_plan: bool, infeasible_stop: bool) -> str:
    if not any_plan or final is None:
        return "FAILURE"
    if final.success:
        return "SUCCESS"
    if infeasible_stop:
        return "INFEASIBLE"
    if initial is None or final.score < initial.score:
        return "PARTIAL_REPAIR"
    if final.score == initial.score:
        return "NO_CHANGE"
    return "REGRESSION"


def _verify_plan(m, out: PlannerOutput):
    return verify(m, out.plan) if out.plan else None


def run_condition(mission: Mission, planner: LLMPlanner, cfg: LoopConfig, seed: int,
                  initial: PlannerOutput | None = None, feasibility: Feasibility | None = None, on_event=None) -> LoopOutcome:
    t0 = time.perf_counter()
    ev = on_event or (lambda e: None)
    if cfg.kind in ("classical", "astar"):  # non-LLM references: visibility-graph planner (= feasibility witness) or grid A* (survey Objective 2)
        feas = feasibility if feasibility is not None else assess(mission)
        if cfg.kind == "astar":
            from src.planner.astar import astar_plan
            ap = astar_plan(mission) if feas.status != "INFEASIBLE" else None
            init = PlannerOutput(ap, [], None if ap else f"A* found no route ({feas.status}: {feas.reason})")
        else:
            init = PlannerOutput(feas.witness_plan if feas.status == "FEASIBLE" else None, [],
                                 None if feas.status == "FEASIBLE" else f"no witness plan ({feas.status}: {feas.reason})")
        v0 = _verify_plan(mission, init)
        its = [RepairIteration(0, init.plan.to_dict() if init.plan else None, v0.to_dict() if v0 else None, None, "INITIAL",
                               v0.score if v0 else None, None, False, [], {}, [], init.error)]
        st = "SUCCESS" if v0 and v0.success else ("INFEASIBLE" if feas.status == "INFEASIBLE" else "FAILURE")
        return LoopOutcome(init, v0.to_dict() if v0 else None, its, init.plan, v0, st, bool(v0 and v0.success), feas.to_dict(), False, time.perf_counter() - t0)
    ev({"type": "planning", "msg": "asking the planner for an initial flight plan"} if initial is None else {"type": "planning", "msg": "re-using the shared initial plan"})
    init = initial if initial is not None else planner.initial_plan(mission, seed, 0)
    v0 = _verify_plan(mission, init)
    ev({"type": "verified", "iteration": 0, "score": v0.score if v0 else None, "valid": bool(v0 and v0.valid),
        "parsed": v0 is not None, "violations": [v.type for v in v0.all_violations()] if v0 else []})
    iters: list[RepairIteration] = [RepairIteration(0, init.plan.to_dict() if init.plan else None, v0.to_dict() if v0 else None, None,
                                                    "INITIAL", v0.score if v0 else None, None, False, [], {}, init.calls, init.error)]
    feas = feasibility if feasibility is not None else (assess(mission) if cfg.kind == "closed_loop" else None)
    oracle_bad = bool(v0 and v0.success and feas is not None and feas.status == "INFEASIBLE")

    def finish(final_plan, final_v, status, released):
        return LoopOutcome(init, v0.to_dict() if v0 else None, iters, final_plan, final_v, status, released,
                           feas.to_dict() if feas else None, oracle_bad, time.perf_counter() - t0)

    if cfg.kind == "llm_only":  # descriptive baseline: any parsed plan is released WITHOUT a verification gate (verified here only for scoring)
        return finish(init.plan, v0, run_status(v0, v0, init.plan is not None, False), init.plan is not None)
    if cfg.kind == "llm_verify":  # verified; only a verified plan is released; no repair
        return finish(init.plan, v0, run_status(v0, v0, init.plan is not None, False), bool(v0 and v0.success))
    if v0 is not None and v0.success:
        return finish(init.plan, v0, "SUCCESS", True)

    if cfg.kind == "resample":  # compute-matched baseline: fresh generations, verifier selects, no feedback
        best_plan, best_v = init.plan, v0
        for k in range(1, cfg.max_iterations + 1):
            out = planner.initial_plan(mission, seed, k, temperature=cfg.temperature if cfg.temperature is not None else RESAMPLE_TEMPERATURE, nonce=cfg.resample_nonce)
            v = _verify_plan(mission, out)
            prev_v = best_v
            outcome = (PARSE_FAIL if v is None else (classify(prev_v, v) if prev_v else (SUCCESS if v.success else PARTIAL)))
            iters.append(RepairIteration(k, out.plan.to_dict() if out.plan else None, v.to_dict() if v else None, None, outcome,
                                         v.score if v else None, prev_v.score if prev_v else None, outcome == REGRESSION, [],
                                         plan_diff(best_plan, out.plan), out.calls, out.error))
            if v is not None and (best_v is None or v.score < best_v.score):
                best_plan, best_v = out.plan, v
            if v is not None and v.success:
                break
        return finish(best_plan, best_v, run_status(v0, best_v, best_plan is not None, False), bool(best_v and best_v.success))

    if cfg.kind in ("direct_patch", "direct_full"):   # LLM-free baseline: apply the prescriptive detours programmatically to the shared initial plan (see repair/direct_patch.py)
        if init.plan is None:
            return finish(None, None, "FAILURE", False)
        if feas is not None and cfg.feasibility_precheck and feas.status == "INFEASIBLE":
            return finish(init.plan, v0, run_status(v0, v0, True, True), False)
        cur_plan, cur_v = init.plan, v0
        best_plan, best_v = cur_plan, cur_v
        for k in range(1, cfg.max_iterations + 1):
            new_plan, plog = (apply_full if cfg.kind == "direct_full" else apply_prescriptive)(mission, cur_plan, cur_v)
            v = verify(mission, new_plan)
            outcome = classify(cur_v, v)
            diff = plan_diff(cur_plan, new_plan)
            diff["collateral_edits"] = 0
            diff["direct_patch"] = plog
            iters.append(RepairIteration(k, new_plan.to_dict(), v.to_dict(), None, outcome, v.score, cur_v.score, outcome == REGRESSION, [], diff, [], None, base=k - 1))
            if v.score < best_v.score:
                best_plan, best_v = new_plan, v
            cur_plan, cur_v = (new_plan, v)
            if v.success or outcome in (NO_CHANGE, REGRESSION):
                break
        if cfg.kind == "direct_full" and cfg.witness_fallback and not (best_v and best_v.success):
            f = feas if feas is not None else assess(mission)
            if f.status == "FEASIBLE" and f.witness_plan is not None:
                wp = f.witness_plan
                wv = verify(mission, wp)
                iters.append(RepairIteration(len(iters) + 1, wp.to_dict(), wv.to_dict(), None, SUCCESS if wv.success else REGRESSION, wv.score, best_v.score if best_v else 0, False, [],
                                             {"witness_fallback": True, "collateral_edits": 0}, [], None, base=0))
                if wv.success:
                    best_plan, best_v = wp, wv
        return finish(best_plan, best_v, run_status(v0, best_v, True, False), bool(best_v and best_v.success))

    # ---- closed loop ----
    if feas is not None and cfg.feasibility_precheck and feas.status == "INFEASIBLE":
        return finish(init.plan, v0, run_status(v0, v0, init.plan is not None, True), False)

    ev({"type": "feasibility", "status": feas.status if feas else None, "reason": feas.reason if feas else None})
    cur_plan, cur_v = init.plan, v0               # plan the next repair starts from
    cur_idx, best_idx = 0, 0
    best_plan, best_v = init.plan, v0
    notes: list[str] = []
    for k in range(1, cfg.max_iterations + 1):
        ev({"type": "repair_start", "iteration": k, "of": cfg.max_iterations})
        if cur_plan is None:  # no parseable initial plan: ask for a fresh plan
            out = planner.initial_plan(mission, seed, k)
            fb_dict, scope = None, None
        else:
            fb = build_feedback(mission, cur_plan, cur_v, cfg.feedback_level, cfg.repair_mode, notes)
            out = planner.repair_plan(mission, repair_prompt(mission, cur_plan, fb, getattr(planner, "prompt_mode", "spec")), seed, k)
            fb_dict, scope = fb.to_dict(), fb.scope if cfg.repair_mode == "targeted" else None
        notes = []
        v = _verify_plan(mission, out)
        if v is None:
            iters.append(RepairIteration(k, None, None, fb_dict, PARSE_FAIL, None, cur_v.score if cur_v else None, False, [], {}, out.calls, out.error))
            continue
        diff = plan_diff(cur_plan, out.plan)
        diff["collateral_edits"] = collateral_edits(diff, scope)
        if cur_v is None:
            outcome, regress, new_v = (SUCCESS if v.success else PARTIAL), False, []
            prev_score = None
        else:
            outcome = classify(cur_v, v)
            regress = outcome == REGRESSION
            prev_keys = {x.key() for x in cur_v.all_violations()}
            new_v = [x.to_dict() for x in v.all_violations() if x.key() not in prev_keys]
            prev_score = cur_v.score
        iters.append(RepairIteration(k, out.plan.to_dict(), v.to_dict(), fb_dict, outcome, v.score, prev_score, regress, new_v, diff, out.calls, out.error, base=cur_idx))
        ev({"type": "verified", "iteration": k, "score": v.score, "outcome": outcome, "valid": v.valid, "violations": [x.type for x in v.all_violations()]})
        if best_v is None or v.score < best_v.score:
            best_plan, best_v, best_idx = out.plan, v, k
        if v.success:
            cur_plan, cur_v, cur_idx = out.plan, v, k
            break
        if regress and cfg.regression_guard:
            cur_plan, cur_v, cur_idx = best_plan, best_v, best_idx
            notes.append(f"Your previous repair attempt made the plan worse ({prev_score} -> {v.score} violations) and was discarded; "
                         f"you are repairing the best plan so far again.")
        else:
            cur_plan, cur_v, cur_idx = out.plan, v, k
    final_plan, final_v = (best_plan, best_v) if cfg.regression_guard else (cur_plan, cur_v)
    return finish(final_plan, final_v, run_status(v0, final_v, final_plan is not None, False), bool(final_v and final_v.success))

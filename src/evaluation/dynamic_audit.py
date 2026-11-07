"""Post-hoc audit of dynamic (mid-flight change) runs recorded with code older than v11-FINAL.

Background (CHANGELOG, FINAL): before the fix, the stale remaining plan carried the ORIGINAL mission id while the re-planning mission has id `<id>_dyn`. Every stale
plan therefore failed mission preservation (MISSION_ID_MISMATCH), so (a) the case filter "keep a case only if the change invalidates the stale plan" removed
nothing, and cases whose stale plan was still geometrically valid were counted; (b) A_llm_only (stale plan kept) scored 0 % by construction.

This module does NOT re-run anything. For every case it re-verifies the logged stale plan against the logged dynamic mission with the id corrected, and reports:
  * how many cases were truly affected by the change (stale plan invalid after the id fix) and how many were not;
  * success per condition on ALL cases (as originally tabulated) and on the TRULY AFFECTED cases only (the corrected estimate);
  * the same per change type and per progress point; paired comparisons on the affected subset (scenario-level sign-flip test, Holm);
  * count reconciliation (cases, seeds, replans per condition, missing units);
  * selection-bias evidence: which source missions entered the study, and (optional) how the same missions fared in the static study compared with the rest.
Caveat that remains (state it): in the old logs the first repair prompt of every LLM condition also listed the spurious MISSION_ID_MISMATCH. The model never
outputs a mission id (the parser assigns the correct one), so the final success criterion is unaffected, but the feedback text contained one extra item."""
from __future__ import annotations
import json
import os
from collections import Counter, defaultdict
import pandas as pd
from src.models.schemas import Mission, FlightPlan
from src.verifier.verifier import verify
from src.evaluation.report import wilson, paired
from src.utils.io import read_jsonl

ID_ONLY = {"MISSION_ID_MISMATCH"}


def case_table(recs: list[dict]) -> pd.DataFrame:
    rows, seen = [], {}
    for r in recs:
        cid = r["scenario_id"]
        if cid in seen:
            continue
        m = Mission.from_dict(r["dynamic"]["mission"])
        stale = FlightPlan.from_dict(r["initial_plan"])
        logged_id = stale.mission_id
        v_logged = verify(m, stale)
        stale.mission_id = m.mission_id
        v = verify(m, stale)
        seen[cid] = True
        rows.append({"case_id": cid, "orig_scenario": r["dynamic"]["orig_scenario"], "difficulty": r["difficulty"], "change": r["dynamic"]["change"],
                     "progress": r["dynamic"]["progress"], "feasibility_after_change": r["scenario_meta"]["feasibility"],
                     "stale_id_logged": logged_id, "dynamic_id": m.mission_id, "id_bug_present": logged_id != m.mission_id,
                     "logged_stale_violations": sorted({x.type for x in v_logged.all_violations()}),
                     "stale_valid_after_id_fix": bool(v.success), "truly_affected": not v.success,
                     "stale_violation_types_after_fix": sorted({x.type for x in v.all_violations()})})
    return pd.DataFrame(rows)


def direct_controls_on_logged_cases(recs: list[dict], cases: pd.DataFrame) -> pd.DataFrame:
    """Model-free controls computed post hoc on EXACTLY the logged cases (same dynamic mission, same stale plan with the id corrected): G_direct_patch and
    G_direct_full_witness. Paired with the logged LLM conditions on the truly affected cases, they answer 'what does the model add in the dynamic setting?'."""
    from src.benchmark.feasibility import assess
    from src.planner.planner import PlannerOutput
    from src.repair.loop import LoopConfig, run_condition
    cfgs = {"G_direct_patch": LoopConfig(name="G_direct_patch", kind="direct_patch", feedback_level="prescriptive", max_iterations=3),
            "G_direct_full_witness": LoopConfig(name="G_direct_full_witness", kind="direct_full", feedback_level="prescriptive", max_iterations=3, witness_fallback=True)}
    first = {}
    for r in recs:
        first.setdefault(r["scenario_id"], r)
    rows = []
    for cid, r in first.items():
        m = Mission.from_dict(r["dynamic"]["mission"])
        stale = FlightPlan.from_dict(r["initial_plan"])
        stale.mission_id = m.mission_id
        feas = assess(m)
        for name, cfg in cfgs.items():
            o = run_condition(m, None, cfg, 0, initial=PlannerOutput(stale, [], None), feasibility=feas)
            rows.append({"case_id": cid, "condition": name, "feasibility": feas.status, "final_success": bool(o.released and o.final_verification is not None and o.final_verification.success),
                         "status": o.status})
    return pd.DataFrame(rows)


def _rate_rows(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    out = []
    for key, g in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        k, n = int(g.final_success.sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({**dict(zip(by, key)), "n": n, "successes": k, "success_rate": k / n if n else float("nan"), "wilson_lo": lo, "wilson_hi": hi})
    return pd.DataFrame(out)


def audit_dynamic_run(run_dir: str, out_dir: str, static_run_dir: str | None = None, pairs: list | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    recs = read_jsonl(os.path.join(run_dir, "runs.jsonl"))
    cases = case_table(recs)
    cases.to_csv(os.path.join(out_dir, "cases_reverified.csv"), index=False)
    aff = dict(zip(cases.case_id, cases.truly_affected))
    rows = [{"case_id": r["scenario_id"], "scenario_id": r["dynamic"]["orig_scenario"], "condition": r["condition"], "seed": f"{r['seed']}|{r['scenario_id']}",
             "run_seed": r["seed"], "change": r["dynamic"]["change"], "progress": r["dynamic"]["progress"], "feasibility": r["scenario_meta"]["feasibility"],
             "truly_affected": aff[r["scenario_id"]], "final_success": bool(r["metrics"].get("final_success")),
             "n_llm_calls": sum(len(it.get("calls", [])) for it in r["iterations"])} for r in recs]
    df = pd.DataFrame(rows)
    feas = df[df.feasibility == "FEASIBLE"]
    allc = _rate_rows(feas, ["condition"]).rename(columns=lambda c: c if c == "condition" else f"all_{c}")
    corr = _rate_rows(feas[feas.truly_affected], ["condition"]).rename(columns=lambda c: c if c == "condition" else f"affected_{c}")
    summ = allc.merge(corr, on="condition", how="left")
    summ.to_csv(os.path.join(out_dir, "success_all_vs_truly_affected.csv"), index=False)
    _rate_rows(feas[feas.truly_affected], ["condition", "change"]).to_csv(os.path.join(out_dir, "affected_success_by_change.csv"), index=False)
    _rate_rows(feas[feas.truly_affected], ["condition", "progress"]).to_csv(os.path.join(out_dir, "affected_success_by_progress.csv"), index=False)
    pairs = pairs if pairs is not None else man.get("pairs", [])
    nonllm = {"E_classical", "F_astar"}
    pc = paired(feas[feas.truly_affected & ~feas.condition.isin(nonllm)], pairs) if pairs else pd.DataFrame()
    pc.to_csv(os.path.join(out_dir, "affected_paired.csv"), index=False)
    # ---- model-free controls on the same logged cases (seed-0 units), paired with the LLM conditions ----
    g = direct_controls_on_logged_cases(recs, cases)
    g["truly_affected"] = g.case_id.map(aff)
    g.to_csv(os.path.join(out_dir, "direct_controls_posthoc.csv"), index=False)
    s0 = feas[(feas.run_seed == 0) & feas.truly_affected]
    gg = g[(g.feasibility == "FEASIBLE") & g.truly_affected]
    vs = []
    for llm in ("C_prescriptive", "C_full", "X_evidence_regen"):
        a = s0[s0.condition == llm].set_index("case_id").final_success
        if not len(a):
            continue
        for ctl in ("G_direct_patch", "G_direct_full_witness"):
            b = gg[gg.condition == ctl].set_index("case_id").final_success
            idx = a.index.intersection(b.index)
            aa, bb = a.loc[idx].astype(bool), b.loc[idx].astype(bool)
            from src.evaluation.llm_value import mcnemar_exact
            n10, n01 = int((aa & ~bb).sum()), int((~aa & bb).sum())
            vs.append({"llm": llm, "direct": ctl, "n_cases": len(idx), "llm_success": float(aa.mean()), "direct_success": float(bb.mean()), "both": int((aa & bb).sum()),
                       "only_llm": n10, "only_direct": n01, "neither": int((~aa & ~bb).sum()), "mcnemar_exact_p": mcnemar_exact(n10, n01)})
    pd.DataFrame(vs).to_csv(os.path.join(out_dir, "llm_vs_direct_on_affected_cases.csv"), index=False)
    # ---- count reconciliation ----
    rec_rows = []
    units_all = set(zip(df.case_id, df.run_seed))
    for cond, g in df.groupby("condition"):
        u = set(zip(g.case_id, g.run_seed))
        rec_rows.append({"condition": cond, "records": len(g), "cases_seed0": int((g.run_seed == 0).sum()), "repeats_other_seeds": int((g.run_seed != 0).sum()),
                         "seeds_present": sorted(int(s) for s in g.run_seed.unique()), "feasible_records": int((g.feasibility == "FEASIBLE").sum()),
                         "feasible_truly_affected": int(((g.feasibility == "FEASIBLE") & g.truly_affected).sum()), "missing_units_vs_union": len(units_all - u)})
    recon = pd.DataFrame(rec_rows)
    recon.to_csv(os.path.join(out_dir, "count_reconciliation.csv"), index=False)
    # ---- selection ----
    sel = {"source_missions": sorted(cases.orig_scenario.unique()), "n_source_missions": int(cases.orig_scenario.nunique()),
           "difficulty_counts_cases": dict(Counter(cases.difficulty)), "selection_mode_in_manifest": (man.get("experiment_config", {}).get("dynamic", {}) or {}).get("selection", "first (legacy default)"),
           "n_per_difficulty": (man.get("experiment_config", {}).get("dynamic", {}) or {}).get("n_per_difficulty"), "stopped_reason": man.get("stopped_reason")}
    if static_run_dir:
        from src.evaluation.report import load_runs
        st, _ = load_runs(static_run_dir)
        st = st[st.feasibility == "FEASIBLE"]
        insel = st.scenario_id.isin(sel["source_missions"])
        comp = {}
        for cond in ("A_llm_only", "C_full", "C_prescriptive", "X_binary_feedback", "D_resample"):
            g = st[st.condition == cond]
            if len(g):
                comp[cond] = {"static_success_selected": float(g[insel.loc[g.index]].final_success.mean()), "n_selected_runs": int(insel.loc[g.index].sum()),
                              "static_success_not_selected": float(g[~insel.loc[g.index]].final_success.mean()) if (~insel.loc[g.index]).any() else None,
                              "n_not_selected_runs": int((~insel.loc[g.index]).sum())}
        sel["static_success_selected_vs_rest"] = comp
    summary = {"run": os.path.basename(os.path.normpath(run_dir)), "n_cases": int(len(cases)), "n_truly_affected": int(cases.truly_affected.sum()),
               "n_not_affected_but_counted": int((~cases.truly_affected).sum()), "id_bug_present_in_log": bool(cases.id_bug_present.any()),
               "not_affected_by_change": {ch: [int((~g.truly_affected).sum()), int(len(g))] for ch, g in cases.groupby("change")},
               "feasible_after_change": dict(Counter(cases.feasibility_after_change)), "selection": sel,
               "caveat": "first repair prompt of LLM conditions listed a spurious MISSION_ID_MISMATCH in these logs; final success is unaffected (the parser assigns the id)"}
    json.dump(summary, open(os.path.join(out_dir, "audit_summary.json"), "w"), indent=1, default=str)
    return summary

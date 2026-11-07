"""Evaluation for Objective 4 (dynamic changes), computed from raw dynamic runs only."""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.evaluation.metrics import run_metrics
from src.evaluation.report import paired, wilson
from src.utils.io import read_jsonl

NONLLM = ["E_classical", "F_astar"]


def load_dynamic(run_dir: str):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        man = json.load(f)
    recs = read_jsonl(os.path.join(run_dir, "runs.jsonl"))
    rows = []
    for r in recs:
        m = run_metrics(r, man.get("pricing"))
        d = r["dynamic"]
        rows.append({"run_id": r["run_id"], "case_id": r["scenario_id"], "scenario_id": d["orig_scenario"], "difficulty": r["difficulty"], "condition": r["condition"],
                     "seed": f"{r['seed']}|{r['scenario_id']}", "feasibility": r["scenario_meta"]["feasibility"], "change": d["change"], "progress": d["progress"],
                     "detection_delay": d.get("detection_delay", 0.0), "exposure": d.get("exposure"), "affected": d["affected"], "failure_cause": d["failure_cause"], "remaining_waypoints": d["remaining_waypoints"],
                     "released": r["metrics"]["released"], "oracle_inconsistent": r["metrics"].get("oracle_inconsistent", False), **m})
    return pd.DataFrame(rows), man, recs


def summarize_dynamic(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cond, ch), g in df.groupby(["condition", "change"], sort=False):
        f = g[g.feasibility == "FEASIBLE"]
        prog_values = sorted(g.progress.unique()) if not g.empty else []
        k, n = int(f.final_success.sum()), len(f)
        lo, hi = wilson(k, n)
        succ = f[f.final_success]
        total_cases = len(g)
        k_all = int(g.final_success.sum())
        rows.append({"condition": cond, "change": ch, "n_cases": total_cases, "n_affected": int(g.affected.sum()),
                     "n_feasible_after_change": n, "replan_success_rate": k / n if n else np.nan,
                     "replan_success_rate_all": k_all / total_cases if total_cases else np.nan,
                     "ci_lo": lo, "ci_hi": hi,
                     "mean_rounds_to_success": succ.dyn_iters_to_success.mean() if len(succ) else np.nan,
                     "mean_llm_calls": g.n_llm_calls.mean(), "recurrence_rate": f.dyn_recurrence.mean() if n else np.nan,
                     "cascade_rate": f.dyn_cascade.mean() if n else np.nan, "mean_waypoint_mods": g.waypoint_mods_total.mean(),
                     "infeasible_after_change": int((g.feasibility == "INFEASIBLE").sum()),
                     "progress_values": prog_values})
    return pd.DataFrame(rows)


def balanced_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only (case, seed) units that were completed by EVERY condition present. Used so that conditions are compared on identical cases when a run
    was interrupted (e.g. one condition has 171 replans while the others have 172): unbalanced counts would otherwise bias paired comparisons."""
    conds = df.condition.nunique()
    key = df.case_id + "|" + df.seed.astype(str)
    per_unit = df.assign(_k=key).groupby("_k").condition.nunique()
    keep = per_unit[per_unit == conds].index
    return df[key.isin(keep)]


def count_reconciliation(df: pd.DataFrame, man: dict) -> pd.DataFrame:
    """One row per condition: cases, replans (= case x seed units) and completeness against the manifest. Documents exactly which counts a paper table may cite."""
    rows = []
    comp = man.get("completeness", {})
    exp_units = df.assign(_k=df.case_id + "|" + df.seed.astype(str))._k.nunique()
    for cond, g in df.groupby("condition", sort=False):
        k = g.case_id + "|" + g.seed.astype(str)
        rows.append({"condition": cond, "n_cases_unique": g.case_id.nunique(), "n_source_scenarios": g.scenario_id.nunique(), "n_replans": k.nunique(),
                     "n_runs_in_file": len(g), "n_duplicate_runs": len(g) - k.nunique(), "units_seen_in_any_condition": exp_units,
                     "missing_vs_other_conditions": exp_units - k.nunique()})
    t = pd.DataFrame(rows)
    t["run_complete"] = comp.get("complete")
    t["termination_reason"] = comp.get("termination_reason")
    return t


def failure_causes(df: pd.DataFrame) -> pd.DataFrame:
    f = df[(~df.final_success) & df.failure_cause.notna()]
    return f.groupby(["condition", "failure_cause"]).size().unstack(fill_value=0).reset_index()


def evaluate_dynamic(run_dir: str, processed_root="outputs/processed", figures_root="outputs/figures", n_examples: int = 4, subfolder: bool = True) -> str:
    df, man, recs = load_dynamic(run_dir)
    rid = man["experiment_run_id"]
    out = os.path.join(processed_root, rid) if subfolder else processed_root
    fig_dir = os.path.join(figures_root, rid) if subfolder else figures_root
    os.makedirs(out, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    df.drop(columns=["dyn_recurred_types", "dyn_cascade_pairs"], errors="ignore").to_csv(os.path.join(out, "per_run_dynamic.csv"), index=False)
    sm = summarize_dynamic(df)
    sm.to_csv(os.path.join(out, "dynamic_summary_by_condition_change.csv"), index=False)
    fc = failure_causes(df)
    fc.to_csv(os.path.join(out, "dynamic_failure_causes.csv"), index=False)
    # progress values actually present in the data (documented in README.txt below)
    prog_present = sorted(df.progress.unique()) if not df.empty else []
    prog_doc = ", ".join(f"progress={p}" for p in prog_present) if prog_present else "none"
    pc = paired(df[~df.condition.isin(NONLLM)], man.get("pairs", []))
    pc.to_csv(os.path.join(out, "dynamic_paired.csv"), index=False)
    count_reconciliation(df, man).to_csv(os.path.join(out, "dynamic_counts_reconciliation.csv"), index=False)
    bal = balanced_subset(df)
    summarize_dynamic(bal).to_csv(os.path.join(out, "dynamic_summary_balanced.csv"), index=False)
    paired(bal[~bal.condition.isin(NONLLM)], man.get("pairs", [])).to_csv(os.path.join(out, "dynamic_paired_balanced.csv"), index=False)
    json.dump({"case_selection": man.get("case_selection"), "completeness": man.get("completeness"), "n_cases": man.get("n_cases"),
               "note": "selection bias check: compare 'selected' with 'eligible'; 'seeded_random' selection is per-difficulty and reproducible from selection_seed."},
              open(os.path.join(out, "dynamic_selection_and_completeness.json"), "w"), indent=1)
    if "detection_delay" in df.columns and df.detection_delay.nunique() > 1:   # detection-latency study
        lat = df[df.feasibility == "FEASIBLE"].groupby(["condition", "detection_delay"]).final_success.agg(["mean", "count"]).rename(columns={"mean": "replan_success", "count": "n"}).reset_index()
        lat.to_csv(os.path.join(out, "dynamic_latency_success.csv"), index=False)
        ex = df[(df.condition == df.condition.iloc[0]) & df.exposure.notna()].groupby("detection_delay").exposure.mean().rename("exposure_rate").reset_index()
        ex.to_csv(os.path.join(out, "dynamic_latency_exposure.csv"), index=False)
    by_prog = df[df.feasibility == "FEASIBLE"].groupby(["condition", "progress"]).final_success.mean().unstack()
    by_prog.to_csv(os.path.join(out, "dynamic_success_by_progress.csv"))
    tag = "MOCK RUN - pipeline test only" if man["is_mock"] else ""
    pv = sm.pivot(index="condition", columns="change", values="replan_success_rate")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    pv.plot(kind="bar", ax=ax)
    ax.set_ylabel("re-plan success (feasible cases)")
    ax.set_title("Robustness to mid-flight changes\n" + tag, fontsize=9, color="red" if man["is_mock"] else "black")
    plt.xticks(rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "dynamic_success_by_change.png"), dpi=200)
    plt.close(fig)
    if len(fc):
        fig, ax = plt.subplots(figsize=(9, 4.2))
        fc.set_index("condition").plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Why re-planning failed\n" + tag, fontsize=9, color="red" if man["is_mock"] else "black")
        plt.xticks(rotation=30, ha="right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "dynamic_failure_causes.png"), dpi=200)
        plt.close(fig)
    from src.simulator.visualize import plot_dynamic
    shown = 0
    for r in recs:
        if r["condition"] == "C_full" and shown < n_examples:
            plot_dynamic(r, os.path.join(fig_dir, f"example_{r['scenario_id']}.png"), tag)
            shown += 1
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(("MOCK RUN: heuristic stand-in, not a language model.\n" if man["is_mock"] else "REAL LLM RUN\n")
                + f"kind=dynamic cases={man['n_cases']} runs={len(df)}\n" + f"progress values tested: {prog_doc}\n")
    return out

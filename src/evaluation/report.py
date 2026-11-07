"""Aggregation from raw JSONL only. Writes processed CSVs + figures. Nothing is hand-entered."""
from __future__ import annotations
import json
import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binomtest
from src.evaluation.metrics import run_metrics, FAMILIES
from src.utils.io import read_jsonl


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


CI_METHODS = {
    "proportions": "Wilson score interval, 95 % (z = 1.96), computed from counts k/n on the stated run set; endpoints in the CSVs are unrounded fractions in [0,1].",
    "proportions_cluster": "Percentile bootstrap over SCENARIOS (cluster bootstrap: all seeds/repeats of a scenario are resampled together), 2000 resamples, seed 0, 95 % (2.5th-97.5th percentile).",
    "paired_difference": "Treatment-minus-control success difference with a scenario-level percentile bootstrap CI (2000 resamples, seed 0); p-value from an exact (<=14 scenarios) or Monte-Carlo (20000 draws) sign-flip permutation test over per-scenario differences; Holm-Bonferroni across the listed comparisons.",
    "rounding": "Tables round fractions to 3 decimals (percent = 1 decimal when printed in a paper). Wilson endpoints are rounded OUTWARD in prose (lower down, upper up) so printed intervals never understate uncertainty.",
}


def cluster_bootstrap_ci(df: pd.DataFrame, col: str, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Scenario-clustered percentile bootstrap CI for the mean of a boolean column (appropriate when each scenario has several seeds/repeats)."""
    if "scenario_id" not in df.columns or col not in df.columns:
        return (float("nan"), float("nan"))
    g = df.groupby("scenario_id")[col].agg(["sum", "count"])
    if len(g) == 0:
        return (float("nan"), float("nan"))
    sums, cnts = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    est = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    return (float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)))


def load_runs(run_dir: str) -> tuple[pd.DataFrame, dict]:
    man = json.load(open(os.path.join(run_dir, "manifest.json")))
    recs = read_jsonl(os.path.join(run_dir, "runs.jsonl"))
    rows = []
    for r in recs:
        m = run_metrics(r, man.get("pricing"))  # recomputed from raw, not trusted from file
        rows.append({"run_id": r["run_id"], "scenario_id": r["scenario_id"], "difficulty": r["difficulty"], "condition": r["condition"],
                     "seed": r["seed"], "prompt_mode": r.get("prompt_mode", "spec"), "feasibility": r["scenario_meta"]["feasibility"], "source": r["scenario_meta"].get("source", "synthetic"),
                     "active": "+".join(r["scenario_meta"].get("active", [])), "n_active": r["scenario_meta"].get("n_active"), "released": r["metrics"]["released"],
                     "oracle_inconsistent": r["metrics"].get("oracle_inconsistent", False), **m})
    return pd.DataFrame(rows), man


def _rate(df, col):
    k, n = int(df[col].sum()), len(df)
    lo, hi = wilson(k, n)
    return pd.Series({f"{col}_rate": k / n if n else float("nan"), f"{col}_ci_lo": lo, f"{col}_ci_hi": hi})


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        r["n_runs"] = len(g)
        # Rates over total runs in group g
        for c in ["initial_valid", "final_valid", "initial_success", "final_success", "released", "any_regression", "failure"]:
            r.update(_rate(g, c).to_dict())
        need = g[g.repair_needed]
        r["n_repair_needed"] = len(need)
        # repair_success_rate computed among runs requiring repair
        r["repair_success_rate"] = need.repair_success.mean() if len(need) else float("nan")
        r["repair_regression_rate"] = need.any_regression.mean() if len(need) else float("nan")
        r["unsafe_release_rate"] = (g.released & ~g.final_success).mean()
        # release-safety triple (mutually exclusive, sums to 1): completed / unsafe release / withheld(rejected)
        r["completion_rate"] = g.final_success.mean()
        r["withheld_rate"] = (~g.released).mean()
        r["safe_release_precision"] = (g.final_success.sum() / g.released.sum()) if g.released.sum() else float("nan")   # P(success | released)
        r["init_extraction_error_rate"] = g.init_extraction_error.mean() if "init_extraction_error" in g else float("nan")
        lo_b, hi_b = cluster_bootstrap_ci(g, "final_success")
        r["final_success_boot_lo"], r["final_success_boot_hi"] = lo_b, hi_b
        # Recurrence and cascade rates (from repair dynamics)
        r["recurrence_rate"] = g.dyn_recurrence.mean() if "dyn_recurrence" in g else float("nan")
        r["cascade_rate"] = g.dyn_cascade.mean() if "dyn_cascade" in g else float("nan")
        for c in ["n_llm_calls", "n_repair_iterations", "waypoint_mods_total", "collateral_edits", "wall_time_s", "verification_time_s",
            "prompt_tokens", "completion_tokens", "n_parse_errors", "n_new_violations_introduced", "distance_overhead_vs_witness"]:
            r[f"mean_{c}"] = g[c].mean() if c in g.columns else float("nan")
        rows.append(r)
    return pd.DataFrame(rows)


def constraint_rates(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in df.groupby("condition", sort=False):
        for f in FAMILIES:
            rows.append({"condition": cond, "constraint": f, "initial_violation_rate": g[f"init_viol_{f}"].mean(), "final_violation_rate": g[f"final_viol_{f}"].mean()})
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame, pairs: list[list[str]], col: str = "final_success", feasible_only: bool = True, n_boot: int = 2000) -> pd.DataFrame:
    d = df[df.feasibility == "FEASIBLE"] if feasible_only else df
    rng = np.random.default_rng(0)
    rows = []
    for t, c in pairs:
        a = d[d.condition == t].set_index(["scenario_id", "seed"])[col].astype(int)
        b = d[d.condition == c].set_index(["scenario_id", "seed"])[col].astype(int)
        idx = a.index.intersection(b.index)
        if len(idx) == 0:
            continue
        a, b = a.loc[idx], b.loc[idx]
        n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        per_s = (a - b).groupby(level=0).mean()  # scenario-level paired difference (seeds averaged)
        boots = [per_s.sample(len(per_s), replace=True, random_state=int(rng.integers(1 << 31))).mean() for _ in range(n_boot)]
        dv = per_s.values
        if len(dv) <= 14:   # exact sign-flip test over scenarios (the independent units)
            import itertools
            obs = abs(dv.mean())
            allm = [abs((dv * np.array(sg)).mean()) for sg in itertools.product([-1, 1], repeat=len(dv))]
            p_perm = float(np.mean([m_ >= obs - 1e-12 for m_ in allm]))
        else:
            rs = np.random.default_rng(1)
            obs = abs(dv.mean())
            sims = np.abs((rs.choice([-1, 1], size=(20000, len(dv))) * dv).mean(axis=1))
            p_perm = float((np.sum(sims >= obs - 1e-12) + 1) / (20001))
        rows.append({"treatment": t, "control": c, "n_scenarios": len(per_s), "scenario_perm_p": p_perm, "n_pairs": len(idx), "treatment_rate": a.mean(), "control_rate": b.mean(),
                     "diff": a.mean() - b.mean(), "diff_boot_ci_lo": float(np.percentile(boots, 2.5)), "diff_boot_ci_hi": float(np.percentile(boots, 97.5)),
                     "treat_only_success": n10, "control_only_success": n01, "mcnemar_exact_p": p})
    out = pd.DataFrame(rows)
    if len(out):   # Holm-Bonferroni across the listed comparisons (primary p = scenario-level permutation test)
        order = np.argsort(out.scenario_perm_p.values)
        m_ = len(out)
        adj = np.empty(m_)
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (m_ - rank) * out.scenario_perm_p.values[i]))
            adj[i] = run
        out["scenario_perm_p_holm"] = adj
    return out


def hardness(df: pd.DataFrame) -> pd.DataFrame:
    """Which constraints are hardest? Initial and final violation rate per constraint x difficulty (feasible scenarios; LLM conditions only)."""
    d = df[(df.feasibility == "FEASIBLE") & (~df.condition.isin(["E_classical", "F_astar"]))]
    rows = []
    for (cond, diff), g in d.groupby(["condition", "difficulty"], sort=False):
        for f in FAMILIES:
            rows.append({"condition": cond, "difficulty": diff, "constraint": f, "initial_violation_rate": g[f"init_viol_{f}"].mean(),
                         "final_violation_rate": g[f"final_viol_{f}"].mean(), "n_runs": len(g)})
    return pd.DataFrame(rows)


def interaction_analysis(df: pd.DataFrame, col: str = "final_success") -> pd.DataFrame:
    """Survey Objective 2: single vs combined constraints. Compares the observed pass rate of each combination with the rate predicted
    if the constraints failed independently (product of the single-constraint pass rates). observed < predicted => harmful interaction."""
    d = df[(df.difficulty == "combo") & (df.feasibility == "FEASIBLE") & ~df.condition.isin(["E_classical", "F_astar"])]
    rows = []
    for cond, g in d.groupby("condition", sort=False):
        single = {a: grp[col].mean() for a, grp in g[g.n_active == 1].groupby("active")}
        for act, grp in g.groupby("active"):
            members = act.split("+")
            if len(members) == 1 or not all(mm in single for mm in members):
                continue
            pred = float(np.prod([single[mm] for mm in members]))
            obs = float(grp[col].mean())
            rows.append({"condition": cond, "constraints": act, "n_active": len(members), "n_runs": len(grp), "observed_pass": obs,
                         "predicted_if_independent": pred, "interaction_effect": obs - pred})
    return pd.DataFrame(rows)


def violation_frequency(df: pd.DataFrame) -> pd.DataFrame:
    e = df[["condition", "difficulty", "init_violation_types"]].explode("init_violation_types").dropna()
    if e.empty:
        return pd.DataFrame()
    t = e.groupby(["condition", "init_violation_types"]).size().rename("count").reset_index()
    tot = df.groupby("condition").size().rename("n_runs")
    t = t.merge(tot, on="condition")
    t["per_run"] = t["count"] / t["n_runs"]
    return t.rename(columns={"init_violation_types": "violation_type"})


def repair_dynamics_tables(df: pd.DataFrame):
    d = df[(df.feasibility == "FEASIBLE") & df.repair_needed & ~df.condition.isin(["A_llm_only", "B_llm_verify", "E_classical", "F_astar"])]
    rows = []
    for cond, g in d.groupby("condition", sort=False):
        succ = g[g.final_success]
        rows.append({"condition": cond, "n_repair_runs": len(g), "success_rate": g.final_success.mean(), "recurrence_rate": g.dyn_recurrence.mean(),
                     "cascade_rate": g.dyn_cascade.mean(), "oscillation_rate": g.dyn_oscillation.mean(), "stuck_rate": g.dyn_stuck.mean(),
                     "mean_rounds_to_success": succ.dyn_iters_to_success.mean() if len(succ) else float("nan"),
                     "mean_calls_per_success": g.n_llm_calls.sum() / max(len(succ), 1), "mean_new_violations": g.n_new_violations_introduced.mean(),
                     "unsolved_rate": 1 - g.final_success.mean()})
    summary = pd.DataFrame(rows)
    pairs = d[["condition", "dyn_cascade_pairs"]].explode("dyn_cascade_pairs").dropna()
    mat = pairs.groupby(["condition", "dyn_cascade_pairs"]).size().rename("count").reset_index().rename(columns={"dyn_cascade_pairs": "fixed_family->new_family"})
    rec = d[["condition", "dyn_recurred_types"]].explode("dyn_recurred_types").dropna()
    rec = rec.groupby(["condition", "dyn_recurred_types"]).size().rename("count").reset_index().rename(columns={"dyn_recurred_types": "recurring_violation_type"})
    return summary, mat, rec


def write_tables(out: str, sf: pd.DataFrame, pc: pd.DataFrame):
    cols = ["condition", "n_runs", "initial_success_rate", "final_success_rate", "final_success_ci_lo", "final_success_ci_hi",
            "released_rate", "unsafe_release_rate", "withheld_rate", "safe_release_precision", "repair_success_rate", "repair_regression_rate", "any_regression_rate",
            "recurrence_rate", "cascade_rate", "mean_n_llm_calls", "mean_collateral_edits"]
    t = sf[[c for c in cols if c in sf.columns]].copy()
    # Format numeric cols to 3 decimals, preserving N/A for NaN
    num_cols = [c for c in t.columns if c not in ("condition", "n_runs")]
    t[num_cols] = t[num_cols].round(3)
    t.to_csv(os.path.join(out, "table_main.csv"), index=False)
    t_disp = t.fillna("N/A")
    with open(os.path.join(out, "table_main.tex"), "w") as f:
        f.write(t.to_latex(index=False, float_format="%.3f", na_rep="N/A", escape=True))
    with open(os.path.join(out, "table_main.md"), "w") as f:
        f.write(t_disp.to_markdown(index=False) if hasattr(t_disp, "to_markdown") and _has_tabulate() else t_disp.to_string(index=False))
    if len(pc):
        with open(os.path.join(out, "table_paired.tex"), "w") as f:
            f.write(pc[["treatment", "control", "n_scenarios", "diff", "diff_boot_ci_lo", "diff_boot_ci_hi", "scenario_perm_p", "scenario_perm_p_holm"]].round(4).to_latex(index=False, escape=True))


CLAIM_LEVELS_MD = """# What the numbers of this evaluation do and do not support

| Level | Claim | Evidence in this folder | Status |
|---|---|---|---|
| **L1 geometric route feasibility** | the released plan satisfies the *modeled* constraints (altitude, zones incl. 3D/terrain, ordering, distance, energy model, mission preservation) | `summary_by_condition_feasible_only.csv`, `release_*` columns, `llm_value/` | evaluated (deterministic verifier; closed-set geometry, tolerance 1e-9) |
| **L2 simulated execution feasibility** | the verified plan, flown by a simulated vehicle with wind, gusts, localization error and a battery model, arrives, completes the mission, and does not intrude into zones | `<raw run>/simulation/executions.csv`, `execution_summary.csv`, `claim_gap.json`, `plots/` ({SIM}) | {SIM_STATUS} |
| **L3 actual UAV flight safety** | the mission can be flown safely by a real vehicle in real airspace (regulation, weather, failures, people, other traffic) | none | **not assessed — no claim is made** |

Rules for writing about results: say "verified plans are geometrically feasible under the modeled constraints" (L1); say "in simulation, X % of verified plans were executed without intrusion" only with the simulation
configuration (vehicle profile, wind, gust, seeds) and the words *uncalibrated simulator*; never say "safe" or "flight-ready" without L3 evidence. A plan that passes L1 and fails L2 is a **finding** (see claim_gap.json), not a simulator bug to hide.
"""


def write_claim_levels(out: str, run_dir: str):
    sim = os.path.join(run_dir, "simulation", "execution_summary.csv")
    have = os.path.exists(sim)
    with open(os.path.join(out, "claim_levels.md"), "w") as f:
        f.write(CLAIM_LEVELS_MD.replace("{SIM}", "present" if have else "not generated: run `python run.py simulate <raw_dir>`").replace("{SIM_STATUS}", "evaluated (built-in simulator, uncalibrated)" if have else "**not evaluated for this run**"))


def write_methods_and_inventory(out: str, df: pd.DataFrame, man: dict):
    """(1) stats_methods.json: exact CI / test / rounding conventions. (2) conditions_inventory.csv: EVERY configured condition with its kind and run count,
    so a paper table can state which conditions are shown, which are omitted, and why (reviewer: '14 conditions' vs. the rows of Table II)."""
    with open(os.path.join(out, "stats_methods.json"), "w") as f:
        json.dump(CI_METHODS, f, indent=1)
    rows = []
    cfgs = man.get("conditions", {})
    n_llm = 0
    for name, c in cfgs.items():
        g = df[df.condition == name]
        uses_llm = c["kind"] not in ("classical", "astar")
        n_llm += int(uses_llm)
        rows.append({"condition": name, "kind": c["kind"], "feedback_level": c.get("feedback_level") if c["kind"] in ("closed_loop", "direct_patch") else "",
                     "repair_mode": c.get("repair_mode") if c["kind"] == "closed_loop" else "", "uses_llm_calls": uses_llm,
                     "llm_calls_after_initial_plan": c["kind"] in ("closed_loop", "resample"), "n_runs": len(g), "n_feasible_runs": int((g.feasibility == "FEASIBLE").sum())})
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(out, "conditions_inventory.csv"), index=False)
    with open(os.path.join(out, "conditions_inventory.txt"), "w") as f:
        f.write(f"{len(cfgs)} configured conditions: {n_llm} start from an LLM plan, {len(cfgs) - n_llm} use no model at all.\n"
                "'uses_llm_calls' = the condition needs the (shared) initial LLM plan; 'llm_calls_after_initial_plan' = it makes further LLM calls (repair/resample).\n"
                "Report all rows in the paper's condition table or state explicitly which are omitted.\n")


def _has_tabulate():
    try:
        import tabulate  # noqa
        return True
    except ImportError:
        return False


def make_figures(df: pd.DataFrame, summ_cond: pd.DataFrame, summ_diff: pd.DataFrame, cons: pd.DataFrame, fig_dir: str, is_mock: bool):
    os.makedirs(fig_dir, exist_ok=True)
    tag = "MOCK RUN - pipeline test only, not LLM evidence" if is_mock else ""
    d = summ_cond.copy()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(d))
    err = [np.nan_to_num(np.clip(d.final_success_rate - d.final_success_ci_lo, 0, None)), np.nan_to_num(np.clip(d.final_success_ci_hi - d.final_success_rate, 0, None))]
    ax.bar(x - 0.2, d.initial_success_rate, 0.4, label="initial (before repair)", color="#bbbbbb")
    ax.bar(x + 0.2, d.final_success_rate, 0.4, yerr=err, label="final (95% Wilson CI)", color="#1f77b4", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(d.condition, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("valid AND mission-preserving (all scenarios)")
    ax.set_title("Success rate by condition\n" + tag, fontsize=9, color="red" if is_mock else "black")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "success_by_condition.png"), dpi=200)
    plt.close(fig)
    pv = summ_diff.pivot(index="condition", columns="difficulty", values="final_success_rate")
    known = ["easy", "medium", "hard", "corridor", "vertical", "combo", "real_easy", "real_medium", "real_hard", "blr_easy", "blr_medium", "blr_hard", "infeasible"]
    pv = pv.reindex(columns=[c for c in known if c in pv.columns] + sorted(c for c in pv.columns if c not in known))   # v12: unknown difficulty labels are kept, not dropped
    fig, ax = plt.subplots(figsize=(8, 4.5))
    pv.plot(kind="bar", ax=ax)
    ax.set_ylabel("final success rate")
    ax.set_title("Success by difficulty\n" + tag, fontsize=9, color="red" if is_mock else "black")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "success_by_difficulty.png"), dpi=200)
    plt.close(fig)
    sel = cons[cons.condition.isin(["A_llm_only", "C_full"])]
    if len(sel):
        fig, ax = plt.subplots(figsize=(8, 4))
        labs = FAMILIES
        for i, (cond, col, color) in enumerate([("A_llm_only", "initial_violation_rate", "#cccccc"), ("C_full", "final_violation_rate", "#1f77b4")]):
            s = sel[sel.condition == cond].set_index("constraint").reindex(labs)
            ax.bar(np.arange(len(labs)) + (i - 0.5) * 0.4, s[col], 0.4, label=f"{cond} ({'initial' if 'initial' in col else 'final'})", color=color)
        ax.set_xticks(np.arange(len(labs)))
        ax.set_xticklabels(labs, fontsize=8)
        ax.set_ylabel("fraction of runs with >=1 violation")
        ax.legend(fontsize=8)
        ax.set_title("Constraint-specific violation rates\n" + tag, fontsize=9, color="red" if is_mock else "black")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "constraint_violation_rates.png"), dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for _, r in d.iterrows():
        ax.scatter(r.mean_n_llm_calls, r.final_success_rate)
        ax.annotate(r.condition, (r.mean_n_llm_calls, r.final_success_rate), fontsize=7)
    ax.set_xlabel("mean LLM calls per run")
    ax.set_ylabel("final success rate")
    ax.set_title("Success vs LLM-call cost\n" + tag, fontsize=9, color="red" if is_mock else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "success_vs_cost.png"), dpi=200)
    plt.close(fig)


def evaluate_run(run_dir: str, processed_root: str = "outputs/processed", figures_root: str = "outputs/figures", subfolder: bool = True) -> str:
    df, man = load_runs(run_dir)
    rid = man["experiment_run_id"]
    out = os.path.join(processed_root, rid) if subfolder else processed_root
    fig_dir = os.path.join(figures_root, rid) if subfolder else figures_root
    os.makedirs(out, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    df.to_csv(os.path.join(out, "per_run_metrics.csv"), index=False)
    sc = summarize(df, ["condition"])
    sd = summarize(df, ["condition", "difficulty"])
    sf = summarize(df[df.feasibility == "FEASIBLE"], ["condition"])
    cr = constraint_rates(df[df.feasibility == "FEASIBLE"])
    sc.to_csv(os.path.join(out, "summary_by_condition.csv"), index=False)
    sf.to_csv(os.path.join(out, "summary_by_condition_feasible_only.csv"), index=False)
    sd.to_csv(os.path.join(out, "summary_by_condition_difficulty.csv"), index=False)
    cr.to_csv(os.path.join(out, "constraint_violation_rates.csv"), index=False)
    df.groupby(["condition", "final_status"]).size().unstack(fill_value=0).to_csv(os.path.join(out, "status_distribution.csv"))
    pc = paired(df, man.get("pairs", []))
    pc.to_csv(os.path.join(out, "paired_comparisons.csv"), index=False)
    hardness(df).to_csv(os.path.join(out, "constraint_hardness_by_difficulty.csv"), index=False)
    summarize(df, ["condition", "source"]).to_csv(os.path.join(out, "summary_by_condition_source.csv"), index=False)
    write_tables(out, sf, pc)
    write_methods_and_inventory(out, df, man)
    write_claim_levels(out, run_dir)
    try:
        from src.evaluation.llm_value import llm_value_analysis
        llm_value_analysis(run_dir, os.path.join(out, "llm_value"))
    except Exception as e:   # the extra analysis must never break the standard evaluation
        print(f"[warn] llm_value analysis skipped: {e}")
    ia = interaction_analysis(df)
    ia.to_csv(os.path.join(out, "constraint_interaction.csv"), index=False)
    violation_frequency(df).to_csv(os.path.join(out, "violation_type_frequency.csv"), index=False)
    df[df.feasibility == "FEASIBLE"].groupby(["condition", "n_active"]).final_success.mean().unstack().to_csv(os.path.join(out, "success_by_number_of_active_constraints.csv"))
    ds, cm, rc = repair_dynamics_tables(df)
    ds.to_csv(os.path.join(out, "repair_dynamics_by_condition.csv"), index=False)
    cm.to_csv(os.path.join(out, "cascade_family_pairs.csv"), index=False)
    rc.to_csv(os.path.join(out, "recurring_violation_types.csv"), index=False)
    if len(ia):
        a_ = ia[ia.condition == "A_llm_only"]
        if len(a_):
            fig, ax = plt.subplots(figsize=(9, 4.2))
            x = np.arange(len(a_))
            ax.bar(x - 0.2, a_.observed_pass, 0.4, label="observed (A_llm_only)")
            ax.bar(x + 0.2, a_.predicted_if_independent, 0.4, label="predicted if independent", color="#bbb")
            ax.set_xticks(x)
            ax.set_xticklabels(a_.constraints.str.replace("no_fly_zone", "nfz"), rotation=30, ha="right", fontsize=7)
            ax.set_ylabel("pass rate")
            ax.set_title("Constraint interaction (single-shot LLM)" + ("\nMOCK RUN" if man["is_mock"] else ""), fontsize=9, color="red" if man["is_mock"] else "black")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(fig_dir, "constraint_interaction.png"), dpi=200)
            plt.close(fig)
    make_figures(df, sf, summarize(df[df.feasibility == "FEASIBLE"], ["condition", "difficulty"]), cr, fig_dir, man["is_mock"])
    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(("MOCK RUN: produced by a heuristic stand-in, NOT a language model. Pipeline test only.\n" if man["is_mock"] else "REAL LLM RUN\n")
                + f"model={man['model']} provider={man['provider']} seeds={man['seeds']} n_runs={len(df)}\n"
                + f"oracle inconsistencies (valid plan on INFEASIBLE scenario): {int(df.oracle_inconsistent.sum())}\n")
    return out

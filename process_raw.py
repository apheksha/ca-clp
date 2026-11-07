# --------------------------------------------------------------
# process_raw.py  –  read existing raw run and write processed CSVs
# --------------------------------------------------------------

import json, os, sys, numpy as np, pandas as pd

# -----------------------------------------------------------------
# USER SETTINGS – this script writes under outputs/runs/<folder>/processed
# -----------------------------------------------------------------
import os
RAW_ROOT = r"outputs/runs/gemini-3.5-flash-lite_full_20261023-181720/raw/real_20261023-181728_55def8"
# -----------------------------------------------------------------
# find the actual run folder (the one that contains state.json) so
# results land in outputs/runs/<model>_<dataset>_<profile>_<timestamp>/
# -----------------------------------------------------------------
def find_run_root(path):
    p = os.path.abspath(path)
    cur = p
    while True:
        if os.path.isfile(os.path.join(cur, "state.json")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    # fallback: everything above the "raw" segment
    parts = p.split(os.sep)
    if "raw" in parts:
        return os.sep.join(parts[:parts.index("raw")])
    return p


run_root = find_run_root(RAW_ROOT)
OUT_DIR = os.path.join(run_root, "processed")
# -----------------------------------------------------------------


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p_hat = k / n
    d = 1 + z*z / n
    c_ = (p_hat + z*z / (2 * n)) / d
    h = z * np.sqrt(p_hat * (1 - p_hat) / n + z*z / (4 * n * n)) / d
    return c_ - h, c_, c_ + h


def safe_sum(series, column):
    if column in series.columns:
        return int(series[column].sum())
    return 0


def summary_by(df_, group_by):
    rows = []
    for key, g in df_.groupby(group_by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(group_by, key))
        r["n_runs"] = len(g)

        for c in ["initial_valid", "final_valid", "initial_success", "final_success",
                  "released", "any_regression", "failure"]:
            k = safe_sum(g, c)
            lo, p_hat, hi = wilson_ci(k, len(g))
            r[f"{c}_rate"] = p_hat
            r[f"{c}_ci_lo"] = lo
            r[f"{c}_ci_hi"] = hi

        need = g[g.repair_needed] if "repair_needed" in g.columns else pd.DataFrame()
        r["n_repair_needed"] = len(need)
        r["repair_success_rate"] = need.final_success.mean() if not need.empty else np.nan
        r["repair_regression_rate"] = need.any_regression.mean() if not need.empty else np.nan
        r["unsafe_release_rate"] = (g.released & ~g.final_success).mean()

        r["recurrence_rate"] = g.dyn_recurrence.mean() if "dyn_recurrence" in g.columns else np.nan
        r["cascade_rate"]    = g.dyn_cascade.mean()    if "dyn_cascade"    in g.columns else np.nan

        for col in ["n_llm_calls","n_repair_iterations","waypoint_mods_total","collateral_edits",
                    "wall_time_s","verification_time_s","prompt_tokens","completion_tokens",
                    "n_parse_errors","n_new_violations_introduced","distance_overhead_vs_witness"]:
            r[f"mean_{col}"] = g[col].mean() if col in g.columns else np.nan

        rows.append(r)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
def main():
    manifest_path = os.path.join(RAW_ROOT, "manifest.json")
    runs_path     = os.path.join(RAW_ROOT, "runs.jsonl")

    if not os.path.isfile(manifest_path):
        sys.exit(f"❌  manifest.json not found at {manifest_path}")
    if not os.path.isfile(runs_path):
        sys.exit(f"❌  runs.jsonl not found at {runs_path}")

    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)

    records = []
    with open(runs_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            # -----------------------------------------------------------------
            # BUILD THE RECORD – ADD final_status HERE  <-- INSERT THESE LINES
            # -----------------------------------------------------------------
            final_status = rec.get("final_status") or rec["metrics"].get("final_status")
            record = {
                "run_id":            rec.get("run_id"),
                "scenario_id":       rec.get("scenario_id"),
                "difficulty":        rec.get("difficulty"),
                "condition":         rec.get("condition"),
                "seed":              rec.get("seed"),
                "feasibility":       rec["scenario_meta"].get("feasibility"),
                "source":            rec["scenario_meta"].get("source", "synthetic"),
                "active":            "+".join(rec["scenario_meta"].get("active", [])),
                "n_active":          rec["scenario_meta"].get("n_active"),
                "released":          rec["metrics"].get("released"),
                "oracle_inconsistent": rec["metrics"].get("oracle_inconsistent", False),

                # metrics the evaluator uses
                "n_llm_calls":        rec["metrics"].get("n_llm_calls"),
                "n_repair_iterations":rec["metrics"].get("n_repair_iterations"),
                "waypoint_mods_total":rec["metrics"].get("waypoint_mods_total"),
                "collateral_edits":   rec["metrics"].get("collateral_edits"),
                "wall_time_s":        rec["metrics"].get("wall_time_s"),
                "verification_time_s":rec["metrics"].get("verification_time_s"),
                "prompt_tokens":      rec["metrics"].get("prompt_tokens"),
                "completion_tokens":  rec["metrics"].get("completion_tokens"),
                "n_parse_errors":     rec["metrics"].get("n_parse_errors"),
                "n_new_violations_introduced": rec["metrics"].get("n_new_violations_introduced"),
                "dyn_recurrence":     rec["metrics"].get("dyn_recurrence", False),
                "dyn_cascade":        rec["metrics"].get("dyn_cascade", False),

                # outcome flags (may be missing)
                "final_success":      rec["metrics"].get("final_success", False),
                "initial_success":    rec["metrics"].get("initial_success", False),
                "initial_valid":      rec["metrics"].get("initial_valid", False),
                "final_valid":        rec["metrics"].get("final_valid", False),

                # <-- ADD: final_status column
                "final_status": final_status,
            }
            # -----------------------------------------------------------------

            records.append(record)

    df = pd.DataFrame(records)

    # -----------------------------------------------------------------
    # rest of the script (creation of CSVs) – unchanged
    # -----------------------------------------------------------------
    os.makedirs(OUT_DIR, exist_ok=True)

    # a) per‑run metrics
    df.to_csv(os.path.join(OUT_DIR, "per_run_metrics.csv"), index=False)

    # b) summary by condition
    summ_cond = summary_by(df, ["condition"])
    summ_cond.to_csv(os.path.join(OUT_DIR, "summary_by_condition.csv"), index=False)

    # c) summary by condition + difficulty
    summ_diff = summary_by(df, ["condition", "difficulty"])
    summ_diff.to_csv(os.path.join(OUT_DIR, "summary_by_condition_difficulty.csv"), index=False)

    # d) constraint‑violation rates (stub)
    conds = df.condition.unique()
    cr_rows = []
    for c in conds:
        g = df[df.condition == c]
        cr_rows.append({"condition": c, "n_runs": len(g),
                        "initial_violation_rate": np.nan, "final_violation_rate": np.nan})
    pd.DataFrame(cr_rows).to_csv(os.path.join(OUT_DIR, "constraint_violation_rates.csv"), index=False)

    # e) status distribution – now works because final_status exists
    if "final_status" in df.columns:
        status_df = df.groupby(["condition", "final_status"]).size().unstack(fill_value=0)
        status_df.to_csv(os.path.join(OUT_DIR, "status_distribution.csv"))
    else:
        placeholder = pd.DataFrame({"condition": df["condition"].unique() if "condition" in df.columns else []})
        for s in ["SUCCESS", "FAILURE"]:
            placeholder[s] = np.nan
        placeholder.to_csv(os.path.join(OUT_DIR, "status_distribution.csv"), index=False)
        print("⚠️  'final_status' column not found – status_distribution.csv written with placeholders.")

    # f) paired comparisons (empty)
    pd.DataFrame(columns=["treatment","control","n_scenarios","diff","diff_boot_ci_lo",
                         "diff_boot_ci_hi","scenario_perm_p","scenario_perm_p_holm",
                         "n_pairs","treatment_rate","control_rate","diff",
                         "treat_only_success","control_only_success","mcnemar_exact_p"])\
       .to_csv(os.path.join(OUT_DIR, "paired_comparisons.csv"), index=False)

    print(f"✅  Processed CSVs written to: {OUT_DIR}")
    print("   • per_run_metrics.csv")
    print("   • summary_by_condition.csv")
    print("   • summary_by_condition_difficulty.csv")
    print("   • constraint_violation_rates.csv")
    print("   • status_distribution.csv")
    print("   • paired_comparisons.csv (empty – can be filled later)")


if __name__ == "__main__":
    main()
# --------------------------------------------------------------
"""Side-by-side comparison of several raw run folders (different models and/or prompt modes), computed from the raw JSONL only.
Reports, per (model, prompt_mode, condition): n, final success (Wilson 95 %), unsafe-release rate, rejection rate, calls, extraction-error rate."""
from __future__ import annotations
import os
import pandas as pd
from src.evaluation.report import load_runs, wilson


def compare_runs(run_dirs: list[str], out_dir: str = "outputs/processed/model_comparison") -> str:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for d in run_dirs:
        df, man = load_runs(d)
        df = df[df.feasibility == "FEASIBLE"]
        for cond, g in df.groupby("condition", sort=False):
            k, n = int(g.final_success.sum()), len(g)
            lo, hi = wilson(k, n)
            rows.append({"run": os.path.basename(os.path.normpath(d)), "provider": man["provider"], "model": man["model"], "prompt_mode": man.get("prompt_mode", "spec"),
                         "is_mock": man["is_mock"], "condition": cond, "n_runs": n, "final_success": k / n if n else float("nan"), "ci95_lo": lo, "ci95_hi": hi,
                         "unsafe_release_rate": float((g.released & ~g.final_success).mean()), "rejection_rate": float((~g.released).mean()),
                         "mean_llm_calls": g.n_llm_calls.mean(), "init_extraction_error_rate": float(g.init_extraction_error.mean()) if "init_extraction_error" in g else float("nan")})
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir, "model_comparison_long.csv"), index=False)
    if len(t):
        wide = t.assign(key=t.model + " [" + t.prompt_mode + "]").pivot(index="condition", columns="key", values="final_success")
        wide.round(3).to_csv(os.path.join(out_dir, "model_comparison_success.csv"))
    with open(os.path.join(out_dir, "README.txt"), "w") as f:
        f.write("Model / prompt-mode comparison. CI: Wilson 95 % (z=1.96) on feasible scenarios. MOCK runs are pipeline tests only.\n")
    return out_dir

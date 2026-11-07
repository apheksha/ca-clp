"""Merged final report for one finished run folder (outputs/final/<...>). Only states what is computed from the raw files.
No interpretation text is generated: numbers, gates and the list of claims the data does / does not license are rule-based."""
from __future__ import annotations
import json
import os
import pandas as pd
from src.evaluation.report import evaluate_run, load_runs, summarize
from src.evaluation.dynamic_report import evaluate_dynamic, load_dynamic
from src.utils.io import read_jsonl


def _md(df: pd.DataFrame, floatfmt=".3f") -> str:
    if df is None or len(df) == 0:
        return "_(empty)_\n"
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        row_str = []
        for v in r.values:
            if isinstance(v, float):
                if pd.isna(v):
                    row_str.append("N/A")
                else:
                    row_str.append(format(v, floatfmt))
            else:
                row_str.append(str(v))
        out.append("| " + " | ".join(row_str) + " |")
    return "\n".join(out) + "\n"


def gates(df: pd.DataFrame, man: dict, dyn: pd.DataFrame | None, dman: dict | None) -> list[tuple[str, bool, str]]:
    g = []
    g.append(("run is a real LLM run (is_mock false)", not man["is_mock"], f"provider={man['provider']} model={man['model']}"))
    g.append(("static run finished without stop", man.get("stopped_reason") is None, str(man.get("stopped_reason"))))
    g.append(("prompt version recorded", bool(man.get("prompt_version")), str(man.get("prompt_version"))))
    g.append(("code fingerprint recorded", bool(_env(man).get("code_sha256")), str(_env(man).get("code_sha256"))[:16]))
    g.append(("oracle inconsistencies == 0", int(df.oracle_inconsistent.sum()) == 0, str(int(df.oracle_inconsistent.sum()))))
    feas = df[df.feasibility == "FEASIBLE"]
    a = feas[feas.condition == "A_llm_only"]
    if len(a):
        r = float(a.final_success.mean())
        g.append(("A_llm_only success strictly between 0 and 1 (benchmark can discriminate)", 0 < r < 1, f"{r:.3f} on {len(a)} feasible missions"))
        zone = a[a.active.str.contains("no_fly_zone", na=False)]
        if len(zone):
            g.append(("floor check: zone-bearing missions not all failed under A", float(zone.final_success.mean()) > 0, f"{zone.final_success.mean():.3f} on {len(zone)}"))
    if dman is not None:
        g.append(("dynamic run finished without stop", dman.get("stopped_reason") is None, str(dman.get("stopped_reason"))))
    return g


def build_final_report(sdir: str) -> str:
    with open(os.path.join(sdir, "state.json"), encoding="utf-8") as fh:
        st = json.load(fh)
    if not st.get("static_dir"):
        raise SystemExit(f"{sdir}: state.json has no static run yet (nothing to report). Run `python run.py final ...` first.")
    proc = os.path.join(sdir, "processed")
    fig = os.path.join(sdir, "figures")
    evaluate_run(st["static_dir"], processed_root=proc, figures_root=fig, subfolder=False)
    d_proc = evaluate_dynamic(st["dynamic_dir"], processed_root=proc, figures_root=fig, subfolder=False) if st.get("dynamic_dir") else None
    df, man = load_runs(st["static_dir"])
    dyn = dman = None
    if st.get("dynamic_dir"):
        dyn, dman, _ = load_dynamic(st["dynamic_dir"])
    feas = df[df.feasibility == "FEASIBLE"]
    n_feas = feas.scenario_id.nunique()
    n_all = df.scenario_id.nunique()
    n_infeas = n_all - n_feas

    L = [f"# Final report: {man['provider']} / {man['model']} / profile {st['profile']}",
         "Generated automatically from raw run files. Contains numbers and rule-based checks only; interpretation is the authors' job.\n",
         "## Provenance",
         f"- run folder: `{sdir}`",
         f"- static run raw: `{st['static_dir']}`; dynamic run raw: `{st.get('dynamic_dir')}`",
         f"- code sha256: `{_env(man).get('code_sha256')}`; git commit: `{_env(man).get('git_commit')}` (dirty: {_env(man).get('git_dirty')})",
         f"- prompt version: `{man.get('prompt_version')}`; seeds: {man.get('seeds')}; sessions (resumes): {st['sessions']}",
         f"- runs: static {len(df)} ({n_feas} feasible, {n_infeas} infeasible scenarios)" + (f", dynamic {len(dyn)}" if dyn is not None else ""),
         f"- parse-error rate: {_parse_rate(st['static_dir'])}", "",
         "## Metric Definitions & Methodological Decisions",
          "- **distance_overhead_vs_witness Formula**: Computed as max(0, route_distance_final / witness_distance - 1), restricted to runs where fin_success == True (plan is valid AND released). For invalid or unreleased plans the field is None (excluded from column means). When a plan is valid its route is a legal path whose distance >= the witness (optimal feasible route) by construction, ensuring overhead >= 0.",
         "- **final_success Formula**: Strictly defined as `final_valid == True AND released == True`. Only flight plans that both satisfy all verifier constraints and pass the release gate are counted as successes.",
         "- **repair_success Rate Formula**: Computed among runs requiring repair: `(runs with repair_success=true) / (runs with repair_needed=true)`. For conditions requiring 0 repairs (e.g. classical planners), reported as `N/A`.",
         "- **any_regression Rate Formula**: Computed over ALL runs per condition: `(runs with any_regression=true) / (total runs per condition)`.",
         "- **Recurrence vs Cascade Breakdown**: `recurrence_rate` measures violations that were fixed and reappeared later in repair; `cascade_rate` measures where fixing one violation directly introduced a new violation elsewhere.",
          "- **A_llm_only — Unverified Descriptive Baseline**: the first parsed plan is released without any verification gate (released_rate = parse rate); the verifier is applied afterwards only to score it. final_success is therefore the single-shot validity rate and is directly comparable with the verified conditions (B_llm_verify, D_resample, C_*), which release only plans that pass the verifier.",
          "- **Release Rate vs Final Success**: released_rate is the share of runs that released a plan. unsafe_release_rate = released AND NOT final_success, i.e. invalid plans released; it can only be non-zero for conditions without a verification gate (A_llm_only).",
          "- **Sample Size**: n_runs counts ALL scenarios (feasible + infeasible) per condition. Wilson CI uses the actual n from the data (never hardcoded). The primary feasible-missions table is shown separately for interpretability; the complete table includes all scenarios. Both use consistent n throughout all CSV and markdown outputs.",
         "- **Dynamic Experiments Progress**: Route progress is computed as horizontal distance traveled divided by total planned route distance (`progress = distance_traveled / total_route_distance`).",
         "- **Energy Model**: Energy proxy model is analytical: `E = distance + k_v * ascent + k_wp * n_waypoints` (default k_v=0.5, k_wp=0.0). Parameters under 'mock:' in config apply ONLY to synthetic mock provider tests, never real LLM runs.",
         "", "## Quality gates"]
    gs = gates(df, man, dyn, dman)
    for n, ok, d in gs:
        L.append(f"- [{'PASS' if ok else 'FAIL'}] {n} ({d})")

    L += ["", f"## Static results: Primary Feasible Missions (n = {n_feas} scenarios, seeds {man.get('seeds')})",
          _md(_compact(summarize(feas, ["condition"]))),
          "", f"## Static results: Complete Benchmark including Infeasible (n = {n_all} scenarios, seeds {man.get('seeds')})",
          _md(_compact(summarize(df, ["condition"]))),
          "### Paired comparisons (scenario-level permutation, Holm)",
          _md(pd.read_csv(os.path.join(proc, "paired_comparisons.csv")) if _exists(proc, man, "paired_comparisons.csv") else None),
          "### Success by number of active constraints",
          _md(pd.read_csv(os.path.join(proc, "success_by_number_of_active_constraints.csv")) if _exists(proc, man, "success_by_number_of_active_constraints.csv") else None)]
    if dyn is not None and d_proc:
        L += ["## Dynamic results (Objective 4)",
              _md(pd.read_csv(os.path.join(proc, "dynamic_summary_by_condition_change.csv")) if _exists(proc, man, "dynamic_summary_by_condition_change.csv") else None),
              "### Dynamic paired",
              _md(pd.read_csv(os.path.join(proc, "dynamic_paired.csv")) if _exists(proc, man, "dynamic_paired.csv") else None)]
    L += ["## What the data licenses (rule-based)"] + [f"- {x}" for x in _licensed(df, man)]
    path = os.path.join(sdir, "FINAL_REPORT.md")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    return path


COMPACT = ["condition", "n_runs", "initial_success_rate", "final_success_rate", "final_success_ci_lo", "final_success_ci_hi", "released_rate",
           "unsafe_release_rate", "repair_success_rate", "repair_regression_rate", "any_regression_rate", "recurrence_rate", "cascade_rate",
           "mean_n_llm_calls", "mean_collateral_edits", "mean_prompt_tokens", "mean_wall_time_s"]


def _compact(t):
    return t[[c for c in COMPACT if c in t.columns]]


def _env(man):
    return man.get("env", {}) or {}


def _exists(proc, man, name):
    return proc and os.path.exists(os.path.join(proc, name))


def _parse_rate(run_dir: str) -> str:
    n = bad = 0
    for r in read_jsonl(os.path.join(run_dir, "runs.jsonl")):
        for it in r.get("iterations", []):
            for c in it.get("calls", []):
                n += 1
                bad += bool(c.get("parse_error"))
    return f"{bad}/{n} calls" if n else "n/a"


def _licensed(df: pd.DataFrame, man: dict) -> list[str]:
    out = []
    feas = df[df.feasibility == "FEASIBLE"]
    ns = feas.scenario_id.nunique()
    out.append(f"Scope: one model ({man['model']}), {ns} feasible missions, seeds {man.get('seeds')}. Do not generalise to other models.")
    a = feas[feas.condition == "A_llm_only"]
    rel = df[df.condition == "A_llm_only"]
    if len(rel):
        out.append(f"Verifier gate: A_llm_only (no gate) released an invalid plan on a feasible mission in {int(((rel.feasibility == 'FEASIBLE') & ~rel.final_success & rel.released).sum())} runs (raw count; see table for rates).")
    out.append("A condition difference may be claimed only where the paired table shows Holm-adjusted p < 0.05; otherwise report it as not distinguished.")
    if len(a) and float(a.final_success.mean()) in (0.0, 1.0):
        out.append("A_llm_only is at floor/ceiling: conditions cannot be ranked on this model; say so.")
    return out

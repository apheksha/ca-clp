"""Does the LLM add value once the deterministic planner can already compute a feasible route?

Compares, on the SAME first plans and the SAME scenarios (paired):
    LLM repair      : C_prescriptive (verifier-computed detours handed to the model), C_full (evidence feedback only)
    direct execution: G_direct_patch      the prescriptive detours applied programmatically (no model after the first plan)
                      G_direct_full       + deterministic repair of order / missing / moved waypoints
                      G_direct_full_witness  + fallback: release the verifier's own witness route
                      E_classical         the visibility-graph planner from scratch (reference)
Outputs (in <run_dir>/../processed/<run>/llm_value/ or `out_dir`):
    llm_value_summary.csv   per (llm, direct) pair: success rates (Wilson 95 %), paired counts (both / only LLM / only direct / neither), exact McNemar p, route overhead, calls/tokens/time/cost
    llm_value_cases.csv     every scenario where the outcomes differ, with the violation classes of the first plan
    llm_value.md            a plain-language answer generated from the numbers (never claims more than the data support)
The analysis is descriptive: with the verifier gate, a null result ('direct execution matches the LLM') is a legitimate finding."""
from __future__ import annotations
import math
import os
import numpy as np
import pandas as pd
from src.evaluation.report import load_runs, wilson
from src.utils.io import read_jsonl

LLM_CONDS = ("C_prescriptive", "C_full")
DIRECT_CONDS = ("G_direct_patch", "G_direct_full", "G_direct_full_witness", "E_classical")


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _classes(types: set) -> str:
    zone = {"NFZ_WAYPOINT", "NFZ_SEGMENT"} & types
    alt = {"ALTITUDE_BELOW_MIN", "ALTITUDE_ABOVE_MAX"} & types
    struct = {"MISSING_REQUIRED_WAYPOINT", "REQUIRED_WAYPOINT_MOVED", "DESTINATION_CHANGED", "ORDER_VIOLATION", "MISSION_ID_MISMATCH"} & types   # names as emitted by src/verifier/verifier.py
    budget = {"DISTANCE_EXCEEDED", "ENERGY_EXCEEDED"} & types
    terr = {"TERRAIN_CLEARANCE", "ABOVE_MAX_AGL"} & types
    parts = [n for n, s in (("zone", zone), ("altitude", alt), ("structure/order", struct), ("budget", budget), ("terrain", terr)) if s]
    return "+".join(parts) if parts else "none"


def llm_value_analysis(run_dir: str, out_dir: str | None = None, llm_conds=LLM_CONDS, direct_conds=DIRECT_CONDS) -> str | None:
    df, man = load_runs(run_dir)
    df = df[df.feasibility == "FEASIBLE"]
    present = set(df.condition)
    llm = [c for c in llm_conds if c in present]
    direct = [c for c in direct_conds if c in present]
    if not llm or not [d for d in direct if d.startswith("G_")]:
        return None
    if not out_dir:
        rd = os.path.normpath(run_dir); par = os.path.dirname(rd)
        if os.path.basename(par) == "raw":   # standard layout: <root>/raw/<run> -> <root>/processed/<run>/llm_value
            out_dir = os.path.join(os.path.dirname(par), "processed", os.path.basename(rd), "llm_value")
        else:                                # run folder elsewhere (e.g. a copy): keep results inside it
            out_dir = os.path.join(rd, "llm_value")
    os.makedirs(out_dir, exist_ok=True)
    first_types = {}
    for r in read_jsonl(os.path.join(run_dir, "runs.jsonl")):
        if r["condition"] == (llm[0]):
            its = r.get("iterations") or []
            v = (its[0].get("verification") if its else None) or {}
            first_types[(r["scenario_id"], r["seed"])] = {x["type"] for x in v.get("violations", []) + v.get("preservation_violations", [])}
    key = lambda g: g.scenario_id + "|" + g.seed.astype(str)
    rows, cases, lines = [], [], []
    for a in llm:
        ga = df[df.condition == a].assign(k=lambda g: key(g)).set_index("k")
        for b in direct:
            gb = df[df.condition == b].assign(k=lambda g: key(g)).set_index("k")
            common = ga.index.intersection(gb.index)
            if not len(common):
                continue
            sa, sb = ga.loc[common, "final_success"].astype(bool), gb.loc[common, "final_success"].astype(bool)
            both, only_a, only_b = int((sa & sb).sum()), int((sa & ~sb).sum()), int((~sa & sb).sum())
            neither = int((~sa & ~sb).sum())
            n = len(common)
            lo_a, hi_a = wilson(int(sa.sum()), n)
            lo_b, hi_b = wilson(int(sb.sum()), n)
            ov_a = ga.loc[common][sa.values & sb.values].distance_overhead_vs_witness.dropna()
            ov_b = gb.loc[common][sa.values & sb.values].distance_overhead_vs_witness.dropna()
            row = {"llm": a, "direct": b, "n_paired": n, "llm_success": sa.mean(), "llm_ci": f"{lo_a:.3f}-{hi_a:.3f}", "direct_success": sb.mean(), "direct_ci": f"{lo_b:.3f}-{hi_b:.3f}",
                   "both": both, "only_llm": only_a, "only_direct": only_b, "neither": neither, "mcnemar_p": mcnemar_exact(only_a, only_b),
                   "llm_minus_direct_pts": 100 * (sa.mean() - sb.mean()),
                   "mean_route_overhead_llm": ov_a.mean() if len(ov_a) else None, "mean_route_overhead_direct": ov_b.mean() if len(ov_b) else None,
                   "llm_calls": ga.loc[common].n_llm_calls.mean(), "direct_calls": gb.loc[common].n_llm_calls.mean(),
                   "llm_prompt_tokens": ga.loc[common].prompt_tokens.mean(), "direct_prompt_tokens": gb.loc[common].prompt_tokens.mean(),
                   "llm_time_s": ga.loc[common].wall_time_s.mean() if "wall_time_s" in ga else None, "llm_cost_usd": ga.loc[common].cost_usd.mean() if "cost_usd" in ga else None}
            rows.append(row)
            for k_, sa_, sb_ in zip(common, sa, sb):
                if sa_ != sb_:
                    sid, sd = k_.rsplit("|", 1)
                    cases.append({"llm": a, "direct": b, "scenario_id": sid, "seed": sd, "outcome": "only_llm_succeeds" if sa_ else "only_direct_succeeds",
                                  "first_plan_violation_classes": _classes(first_types.get((sid, int(sd)), set()))})
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "llm_value_summary.csv"), index=False)
    pd.DataFrame(cases).to_csv(os.path.join(out_dir, "llm_value_cases.csv"), index=False)
    # ---- plain-language answer ----
    lines.append("# Does the LLM add value after the deterministic planner has found a feasible route?\n")
    lines.append(f"Run: `{os.path.basename(os.path.normpath(run_dir))}` (model {man['model']}{' — MOCK, pipeline test only, numbers meaningless' if man['is_mock'] else ''}). Feasible scenarios only; paired on the same first plans.\n")
    for r in rows:
        verdict = ("no evidence of added value (the direct control is at least as successful)" if r["only_llm"] <= r["only_direct"] else
                   ("LLM succeeds on missions the direct control does not" + (" (statistically distinguishable, McNemar p=%.3g)" % r["mcnemar_p"] if r["mcnemar_p"] < 0.05 else " (not statistically distinguishable)")))
        lines.append(f"* **{r['llm']} vs {r['direct']}** (n={r['n_paired']}): LLM {100 * r['llm_success']:.1f}% [{r['llm_ci']}] vs direct {100 * r['direct_success']:.1f}% [{r['direct_ci']}]; "
                     f"both {r['both']}, only LLM {r['only_llm']}, only direct {r['only_direct']}, neither {r['neither']}. → {verdict}. "
                     f"LLM calls {r['llm_calls']:.1f} vs {r['direct_calls']:.1f}.")
    wins = [c for c in cases if c["outcome"] == "only_llm_succeeds"]
    if wins:
        from collections import Counter
        lines.append("\nMissions only the LLM solved, by violation class of the first plan: " + ", ".join(f"{k}: {v}" for k, v in Counter(c["first_plan_violation_classes"] for c in wins).most_common()) + ".")
    lines.append("\nReading guide: a direct control that equals C_prescriptive means the success of prescriptive feedback is verifier computation, not LLM reasoning. Differences in route overhead and in the "
                 "cost columns are secondary outcomes. This analysis does not say anything about natural-language understanding (the structured specification is part of the prompt) or about simulated/actual flight.")
    open(os.path.join(out_dir, "llm_value.md"), "w").write("\n".join(lines) + "\n")
    return out_dir

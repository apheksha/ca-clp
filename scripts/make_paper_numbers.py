#!/usr/bin/env python
"""Computes EVERY number the paper cites, from the raw run files and the validation outputs, and writes
   paper/generated/numbers.json   (value + source file for each number)
   paper/generated/macros.tex     (\\newcommand per number; the paper uses only these)
   paper/generated/tab_static.tex, tab_dynamic.tex  (table bodies)
   paper/figures/fig_success.pdf  (success by condition with 95 % intervals)
Nothing is typed by hand. Re-run after any new experiment:  python scripts/make_paper_numbers.py
Rounding rule (stated in the paper): percentages to one decimal; interval end points rounded to one decimal, no outward/inward adjustment."""
from __future__ import annotations
import glob
import json
import math
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from src.evaluation.report import load_runs, summarize, paired, wilson, cluster_bootstrap_ci

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
GEN = os.environ.get("PAPER_GEN_DIR", "paper/generated")
FIG = os.environ.get("PAPER_FIG_DIR", "paper/figures")
INCLUDE_MOCK = os.environ.get("PAPER_INCLUDE_MOCK") == "1"     # tests only: lets mock/sanity runs exercise the optional sections
os.makedirs(GEN, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

RUN_REAL = "outputs/raw/real_20261025-194958_3b2644_seed0-only__plus-direct"
RUN_SYN = "outputs/raw/real_20261023-181728_55def8__plus-direct"
DYN_REAL_OLD = "dyn-real_20261025-141536_7aae72_seed0-only"
DYN_SYN_OLD = "dyn-real_20261025-173134_6d86b6"
AUDIT = "outputs/review_v12/dynamic_audit"

N: dict = {}          # name -> {"value": ..., "src": ...}
TEX: list = []


def put(name: str, value, src: str, fmt: str | None = None):
    """Register a number. `fmt` controls the printed macro (default: as is)."""
    N[name] = {"value": value, "src": src}
    if fmt is None:
        s = str(value)
    elif fmt == "pct":
        s = f"{100 * value:.1f}" if value >= 0 else f"$-${100 * abs(value):.1f}"      # typographic minus
    elif fmt == "pct0":
        s = f"{100 * value:.0f}"
    elif fmt == "p":                       # p-values: three decimals, '<0.001' below that
        s = "$<$0.001" if value < 0.001 else f"{value:.3f}"
    elif fmt == "sci":                     # e.g. 6.0\times10^{-8} (math mode)
        m_, e_ = f"{value:.1e}".split("e")
        s = f"${m_}\\times10^{{{int(e_)}}}$" if value < 0.001 else f"{value:.3f}"
    else:
        s = format(value, fmt)
    TEX.append((name, s))


DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def macro_name(name: str) -> str:
    """LaTeX macro names may only contain letters: digits are spelled out, everything else is dropped; the first letter of every word is capitalised."""
    s = "".join(DIGITS.get(ch, ch) for ch in name)
    out = "".join(ch for ch in s[:1].upper() + s[1:] if ch.isalpha())
    return "\\N" + out


# ------------------------------------------------------------------------------------------------------------------------------
# static results
# ------------------------------------------------------------------------------------------------------------------------------
ORDER = ["A_llm_only", "B_llm_verify", "D_resample", "X_binary_feedback", "X_binary_targeted", "X_localized", "X_one_iteration", "C_full", "X_no_guard",
         "X_no_precheck", "X_evidence_regen", "C_prescriptive", "G_direct_patch", "G_direct_full", "G_direct_full_witness", "E_classical", "F_astar"]
# v12.1: one identifier per condition, used identically in Table II, Fig. 2 and the text (review: consistent names)
IDENT = {"A_llm_only": "A", "B_llm_verify": "B", "D_resample": "D", "X_binary_feedback": "X\\_binary", "X_binary_targeted": "X\\_binary\\_targ",
         "X_localized": "X\\_localized", "X_one_iteration": "X\\_one\\_round", "C_full": "C\\_full", "X_no_guard": "X\\_no\\_guard", "X_no_precheck": "X\\_no\\_precheck",
         "X_evidence_regen": "X\\_evid\\_regen", "C_prescriptive": "C\\_prescriptive", "G_direct_patch": "G1", "G_direct_full": "G2", "G_direct_full_witness": "G3",
         "E_classical": "E", "F_astar": "F"}
WHAT = {"A_llm_only": "first plan released, no verifier", "B_llm_verify": "first plan, released only if valid", "D_resample": "new plan each round, $T{=}0.7$",
        "X_binary_feedback": "binary feedback, regenerate", "X_binary_targeted": "binary feedback, targeted", "X_localized": "localized feedback, regenerate",
        "X_one_iteration": "as C\\_full, one round", "C_full": "evidence, targeted, guard, precheck", "X_no_guard": "as C\\_full, no guard",
        "X_no_precheck": "as C\\_full, no precheck", "X_evidence_regen": "evidence, regenerate", "C_prescriptive": "evidence + verifier detours",
        "G_direct_patch": "detours applied, no model", "G_direct_full": "G1 + rebuild required points", "G_direct_full_witness": "G2, else oracle witness",
        "E_classical": "visibility-graph planner", "F_astar": "grid A* planner"}
LABEL = {k: (f"\\textbf{{{v}}}" if k in ("C_full", "C_prescriptive") else v) for k, v in IDENT.items()}

real, man_r = load_runs(RUN_REAL)
syn, man_s = load_runs(RUN_SYN)
real_f = real[real.feasibility == "FEASIBLE"]
syn_f = syn[syn.feasibility == "FEASIBLE"]
put("realMissions", real_f.scenario_id.nunique(), RUN_REAL)
put("synMissions", syn.scenario_id.nunique(), RUN_SYN)
put("synFeasible", syn_f.scenario_id.nunique(), RUN_SYN)
put("synInfeasible", syn.scenario_id.nunique() - syn_f.scenario_id.nunique(), RUN_SYN)
put("synSeeds", syn.seed.nunique(), RUN_SYN)
put("synRuns", len(syn_f[syn_f.condition == "C_full"]), RUN_SYN)
put("nConditions", len(ORDER), "scripts/make_paper_numbers.py ORDER")
put("nLLMConditions", sum(1 for c in ORDER if c.startswith(("A_", "B_", "D_", "X_", "C_"))), "ORDER")
put("nModelFreeConditions", sum(1 for c in ORDER if c.startswith(("G_", "E_", "F_"))), "ORDER")
put("modelId", man_r["model"], RUN_REAL + "/manifest.json")
put("realRegions", len({s.split("_")[1] if s.count("_") < 3 else "_".join(s.split("_")[1:-1]) for s in real_f.scenario_id.unique()}), RUN_REAL)

rows = []
for cond in ORDER:
    gr, gs = real_f[real_f.condition == cond], syn_f[syn_f.condition == cond]
    kr, nr = int(gr.final_success.sum()), len(gr)
    lr, hr = wilson(kr, nr)
    ks, ns = int(gs.final_success.sum()), len(gs)
    ls, hs = cluster_bootstrap_ci(gs, "final_success")
    r = {"condition": cond, "real_k": kr, "real_n": nr, "real": kr / nr, "real_lo": lr, "real_hi": hr, "syn_k": ks, "syn_n": ns, "syn": ks / ns, "syn_lo": ls, "syn_hi": hs,
         "calls_r": gr.n_llm_calls.mean(), "calls_s": gs.n_llm_calls.mean(), "tok_r": gr.prompt_tokens.mean(), "time_r": gr.wall_time_s.mean(),
         "unsafe_r": float((gr.released & ~gr.final_success).mean()), "unsafe_s": float((gs.released & ~gs.final_success).mean()),
         "withheld_r": float((~gr.released).mean()), "overhead_r": gr.distance_overhead_vs_witness.mean(), "overhead_s": gs.distance_overhead_vs_witness.mean()}
    rows.append(r)
    key = cond.replace("_", "")
    put(f"real{key}", r["real"], RUN_REAL, "pct")
    put(f"real{key}lo", lr, RUN_REAL, "pct")
    put(f"real{key}hi", hr, RUN_REAL, "pct")
    put(f"real{key}k", kr, RUN_REAL)
    put(f"syn{key}", r["syn"], RUN_SYN, "pct")
    put(f"real{key}calls", r["calls_r"], RUN_REAL, ".1f")
    put(f"real{key}tokens", r["tok_r"], RUN_REAL, ",.0f")
    put(f"real{key}time", r["time_r"], RUN_REAL, ".0f")
    put(f"real{key}overhead", r["overhead_r"] if not math.isnan(r["overhead_r"]) else 0.0, RUN_REAL, "pct")
tab = pd.DataFrame(rows)
tab.to_csv(f"{GEN}/static_table.csv", index=False)


def fmt_ci(p, lo, hi):
    return f"{100 * p:.1f} & \\scriptsize[{100 * lo:.1f}, {100 * hi:.1f}]"


lines = []
for r in rows:
    calls = f"{r['calls_r']:.1f}/{r['calls_s']:.1f}"
    if r["condition"] in ("E_classical", "F_astar"):
        calls = "0/0"
    w_r, w_s = 1 - r["real"] - r["unsafe_r"], 1 - r["syn"] - r["unsafe_s"]          # completed + unsafe + withheld = 1 (three exclusive outcomes)
    lines.append(f"{LABEL[r['condition']]} & {WHAT[r['condition']]} & {fmt_ci(r['real'], r['real_lo'], r['real_hi'])} & {100 * r['unsafe_r']:.1f} & {100 * w_r:.1f} & "
                 f"{fmt_ci(r['syn'], r['syn_lo'], r['syn_hi'])} & {100 * r['unsafe_s']:.1f} & {100 * w_s:.1f} & {calls} \\\\")
    put(f"real{r['condition'].replace('_', '')}withheld", w_r, RUN_REAL, "pct")
    put(f"syn{r['condition'].replace('_', '')}withheld", w_s, RUN_SYN, "pct")
    put(f"real{r['condition'].replace('_', '')}unsaferate", r["unsafe_r"], RUN_REAL, "pct")
    put(f"syn{r['condition'].replace('_', '')}unsaferate", r["unsafe_s"], RUN_SYN, "pct")
    if r["condition"] in ("B_llm_verify", "X_evidence_regen", "C_prescriptive", "G_direct_full_witness"):
        lines.append("\\midrule")
open(f"{GEN}/tab_static.tex", "w").write("\n".join(lines) + "\n")

# release safety and first-plan violations (A on real)
a_r = real_f[real_f.condition == "A_llm_only"]
a_s = syn_f[syn_f.condition == "A_llm_only"]
put("realAunsafe", float((a_r.released & ~a_r.final_success).mean()), RUN_REAL, "pct0")
put("synAunsafe", float((a_s.released & ~a_s.final_success).mean()), RUN_SYN, "pct0")
verified = [c for c in ORDER if c not in ("A_llm_only",)]
put("unsafeVerifiedAll", int(sum(((real_f[real_f.condition == c].released & ~real_f[real_f.condition == c].final_success).sum()) for c in verified)
                             + sum(((syn_f[syn_f.condition == c].released & ~syn_f[syn_f.condition == c].final_success).sum()) for c in verified)), "both runs")
put("realAzone", float(a_r.init_viol_no_fly_zone.mean()), RUN_REAL, "pct0")
put("synAzone", float(a_s.init_viol_no_fly_zone.mean()), RUN_SYN, "pct0")
for fam in ("altitude", "distance", "ordering", "energy", "mission_preservation"):
    put(f"realAviol{fam.replace('_', '')}", int(a_r[f"init_viol_{fam}"].sum()), RUN_REAL)
inf_s = syn[(syn.feasibility == "INFEASIBLE")]
put("synInfeasibleRuns", len(inf_s[inf_s.condition == "C_full"]), RUN_SYN)
put("synInfeasibleReleasedVerified", int(inf_s[inf_s.condition != "A_llm_only"].released.sum()), RUN_SYN)
put("synInfeasibleReleasedA", int(inf_s[inf_s.condition == "A_llm_only"].released.sum()), RUN_SYN)
cf_inf, np_inf = inf_s[inf_s.condition == "C_full"], inf_s[inf_s.condition == "X_no_precheck"]
put("precheckCallsOn", cf_inf.n_llm_calls.mean(), RUN_SYN, ".1f")
put("precheckCallsOff", np_inf.n_llm_calls.mean(), RUN_SYN, ".1f")
put("precheckTokOn", cf_inf.prompt_tokens.mean(), RUN_SYN, ",.0f")
put("precheckTokOff", np_inf.prompt_tokens.mean(), RUN_SYN, ",.0f")

# paired tests (scenario-level sign-flip permutation, Holm over the family below)
PAIRS = [["C_full", "A_llm_only"], ["C_full", "D_resample"], ["C_full", "X_binary_feedback"], ["C_full", "X_localized"], ["C_full", "X_evidence_regen"],
         ["C_full", "X_no_guard"], ["C_full", "X_one_iteration"], ["C_prescriptive", "C_full"], ["C_prescriptive", "G_direct_patch"], ["C_full", "G_direct_patch"]]
pr = paired(real_f, PAIRS)
ps = paired(syn_f, PAIRS)
pr.to_csv(f"{GEN}/paired_real.csv", index=False)
ps.to_csv(f"{GEN}/paired_syn.csv", index=False)
for df_, tag in ((pr, "real"), (ps, "syn")):
    for _, x in df_.iterrows():
        k = f"{tag}P{x.treatment.replace('_', '')}vs{x.control.replace('_', '')}"
        put(k + "diff", x["diff"], f"{GEN}/paired_{tag}.csv", "pct")
        put(k + "lo", x.diff_boot_ci_lo, f"{GEN}/paired_{tag}.csv", "pct")
        put(k + "hi", x.diff_boot_ci_hi, f"{GEN}/paired_{tag}.csv", "pct")
        put(k + "p", x.scenario_perm_p_holm, f"{GEN}/paired_{tag}.csv", "p")
# sign agreement between datasets
m_ = pr.merge(ps, on=["treatment", "control"], suffixes=("_r", "_s"))
put("pairsCompared", len(m_), "paired tables")
put("pairsSameSign", int((np.sign(m_.diff_r) == np.sign(m_.diff_s)).sum()), "paired tables")
put("pairsSameVerdict", int(((m_.scenario_perm_p_holm_r < 0.05) == (m_.scenario_perm_p_holm_s < 0.05)).sum()), "paired tables")
llm_conds = [c for c in ORDER if c.startswith(("A_", "D_", "X_", "C_"))]
rr = tab.set_index("condition").loc[llm_conds]
rho, prho = spearmanr(rr.real, rr.syn)
pc_, ppc = pearsonr(rr.real, rr.syn)
put("nCorrConditions", len(llm_conds), "ORDER")
put("spearman", rho, f"{GEN}/static_table.csv", ".2f")
put("spearmanp", prho, f"{GEN}/static_table.csv", "sci")
put("pearson", pc_, f"{GEN}/static_table.csv", ".2f")

# difficulty
for cond in ("C_full", "C_prescriptive", "A_llm_only"):
    for d in ("real_easy", "real_medium", "real_hard"):
        g = real_f[(real_f.condition == cond) & (real_f.difficulty == d)]
        put(f"diff{cond.replace('_', '')}{d.replace('_', '')}", float(g.final_success.mean()), RUN_REAL, "pct0")
    for d in ("easy", "medium", "hard", "corridor", "combo"):
        g = syn_f[(syn_f.condition == cond) & (syn_f.difficulty == d)]
        put(f"diff{cond.replace('_', '')}syn{d}", float(g.final_success.mean()), RUN_SYN, "pct0")
# combination family: success by number of active constraints, zone-bearing only (C_full)
cmb = syn_f[(syn_f.difficulty == "combo") & (syn_f.condition == "C_full")]
zone = cmb[cmb.active.str.contains("no_fly_zone")]
for k_, lab in ((1, "one"), (2, "two"), (3, "three")):
    put(f"comboZone{lab}", float(zone[zone.n_active == k_].final_success.mean()), RUN_SYN, "pct0")
put("comboZonefourfive", float(zone[zone.n_active >= 4].final_success.mean()), RUN_SYN, "pct0")
put("comboNoZone", float(cmb[~cmb.active.str.contains("no_fly_zone")].final_success.mean()), RUN_SYN, "pct0")

# repair dynamics (runs that needed repair)
for df_, tag, src in ((real_f, "real", RUN_REAL), (syn_f, "syn", RUN_SYN)):
    for cond in ("C_full", "X_binary_feedback", "C_prescriptive", "D_resample"):
        g = df_[(df_.condition == cond) & df_.repair_needed]
        k = f"{tag}dyn{cond.replace('_', '')}"
        put(k + "n", len(g), src)
        for col in ("final_success", "dyn_recurrence", "dyn_cascade", "dyn_oscillation", "dyn_stuck"):
            put(k + col.replace("_", "").replace("dyn", ""), float(g[col].mean()), src, "pct0")
    pairs_ = df_[(df_.condition == "C_full") & df_.repair_needed][["dyn_cascade_pairs"]].explode("dyn_cascade_pairs").dropna()
    c_ = Counter(pairs_.dyn_cascade_pairs)
    tot = sum(c_.values())
    cross = sum(v for k2, v in c_.items() if k2.split("->")[0] != k2.split("->")[1])
    bud = sum(v for k2, v in c_.items() if {"no_fly_zone"} & set(k2.split("->")) and ({"distance", "energy"} & set(k2.split("->"))))
    put(f"{tag}cascadePairs", tot, src)
    put(f"{tag}cascadeCross", cross, src)
    put(f"{tag}cascadeZoneBudget", bud, src)
    put(f"{tag}cascadeZoneBudgetShare", bud / tot if tot else 0.0, src, "pct0")
# guard ablation
for df_, tag, src in ((real_f, "real", RUN_REAL), (syn_f, "syn", RUN_SYN)):
    g, ng = df_[df_.condition == "C_full"], df_[df_.condition == "X_no_guard"]
    put(f"{tag}GuardDistViol", float(g.final_viol_distance.mean()), src, "pct")
    put(f"{tag}NoGuardDistViol", float(ng.final_viol_distance.mean()), src, "pct")
    put(f"{tag}NoGuardWorse", int((ng.final_n_violations > ng.init_n_violations).sum()), src)
    put(f"{tag}GuardWorse", int((g.final_n_violations > g.init_n_violations).sum()), src)
    put(f"{tag}OneRoundRounds", float(df_[(df_.condition == "C_full") & df_.final_success & df_.repair_needed].dyn_iters_to_success.mean()), src, ".1f")

# ------------------------------------------------------------------------------------------------------------------------------
# LLM vs direct execution
# ------------------------------------------------------------------------------------------------------------------------------
for df_, tag, src in ((real_f, "real", RUN_REAL), (syn_f, "syn", RUN_SYN)):
    key = df_.scenario_id + "|" + df_.seed.astype(str)
    a = df_[df_.condition == "C_prescriptive"].assign(k=key).set_index("k").final_success
    for ctl in ("G_direct_patch", "G_direct_full", "G_direct_full_witness"):
        b = df_[df_.condition == ctl].assign(k=key).set_index("k").final_success
        idx = a.index.intersection(b.index)
        put(f"{tag}OnlyLLM{ctl.replace('_', '')}", int((a.loc[idx] & ~b.loc[idx]).sum()), src)
        put(f"{tag}OnlyDirect{ctl.replace('_', '')}", int((~a.loc[idx] & b.loc[idx]).sum()), src)

# ------------------------------------------------------------------------------------------------------------------------------
# replication: the synthetic study repeated on 5-6 Nov 2026 (same model id, later code, seeds 0/1/2 in three separate runs; G1 run online)
# ------------------------------------------------------------------------------------------------------------------------------
REP_RUNS = sorted(d for d in glob.glob("outputs/raw/real_2026110[56]-*") if not d.endswith("__plus-direct"))
rep_ok = []
for d in REP_RUNS:
    m_ = json.load(open(f"{d}/manifest.json"))
    if m_.get("provider") == man_s.get("provider") and m_.get("model") == man_s.get("model") and not m_.get("is_mock") \
            and m_.get("data_sources") == ["synthetic"] and m_.get("scenario_files_sha256") == man_s.get("scenario_files_sha256"):
        rep_ok.append(d)
if rep_ok:
    from scipy.stats import binomtest
    rep = pd.concat([load_runs(d)[0] for d in rep_ok], ignore_index=True)
    rep_f = rep[rep.feasibility == "FEASIBLE"]
    src = ";".join(rep_ok)
    put("repNRuns", len(rep_ok), src)
    put("repRecords", len(rep), src)
    put("repSeeds", rep.seed.nunique(), src)
    rep_conds = [c for c in ORDER if c in set(rep.condition)]
    put("repNConditions", len(rep_conds), src)
    diffs = []
    for cond in rep_conds:
        g = rep_f[rep_f.condition == cond]
        lo, hi = cluster_bootstrap_ci(g, "final_success")
        put(f"rep{cond.replace('_', '')}", float(g.final_success.mean()), src, "pct")
        put(f"rep{cond.replace('_', '')}lo", lo, src, "pct")
        put(f"rep{cond.replace('_', '')}hi", hi, src, "pct")
        diffs.append(abs(float(g.final_success.mean()) - float(syn_f[syn_f.condition == cond].final_success.mean())))
    put("repMaxDiff", max(diffs), src + ";" + RUN_SYN, "pct")
    keyr = rep_f.scenario_id + "|" + rep_f.seed.astype(str)
    a = rep_f[rep_f.condition == "C_prescriptive"].assign(k=keyr).set_index("k").final_success
    b = rep_f[rep_f.condition == "G_direct_patch"].assign(k=keyr).set_index("k").final_success
    idx = a.index.intersection(b.index)
    ol, od = int((a.loc[idx] & ~b.loc[idx]).sum()), int((~a.loc[idx] & b.loc[idx]).sum())
    put("repOnlyLLM", ol, src)
    put("repOnlyDirect", od, src)
    put("repMcNemarP", float(binomtest(ol, ol + od).pvalue) if ol + od else 1.0, src, ".2f")
    pl, pd_ = ol + N["synOnlyLLMGdirectpatch"]["value"], od + N["synOnlyDirectGdirectpatch"]["value"]
    put("poolOnlyLLM", pl, src + ";" + RUN_SYN)
    put("poolOnlyDirect", pd_, src + ";" + RUN_SYN)
    put("poolMcNemarP", float(binomtest(pl, pl + pd_).pvalue) if pl + pd_ else 1.0, src + ";" + RUN_SYN, ".2f")
    # determinism of first plans at temperature 0: same scenario and seed in October and November
    def _first_plans(path):
        out = {}
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    r_ = json.loads(line)
                    if r_["condition"] == "A_llm_only":
                        out[(r_["scenario_id"], r_["seed"])] = json.dumps(r_.get("initial_plan"), sort_keys=True)
        return out
    fo = _first_plans(f"{RUN_SYN}/runs.jsonl")
    fn = {}
    for d in rep_ok:
        fn.update(_first_plans(f"{d}/runs.jsonl"))
    common = set(fo) & set(fn)
    put("repFirstCompared", len(common), src + ";" + RUN_SYN)
    put("repFirstSame", sum(fo[k] == fn[k] for k in common) / len(common), src + ";" + RUN_SYN, "pct0")
    put("repDates", "5--6 November 2026", src)

for run, tag in ((DYN_REAL_OLD, "dr"), (DYN_SYN_OLD, "ds")):      # v12.1: count reconciliation (review: 172 / 146 / 30 / 171)
    cr = pd.read_csv(f"{AUDIT}/{run}/count_reconciliation.csv")
    full = cr[cr.missing_units_vs_union == 0]
    put(f"{tag}Records", int(full.records.max()), f"{AUDIT}/{run}/count_reconciliation.csv")
    put(f"{tag}SeedZeroCases", int(cr.cases_seed0.max()), f"{AUDIT}/{run}/count_reconciliation.csv")
    put(f"{tag}Repeats", int(full.repeats_other_seeds.max()), f"{AUDIT}/{run}/count_reconciliation.csv")
    put(f"{tag}FeasibleRecords", int(full.feasible_records.max()), f"{AUDIT}/{run}/count_reconciliation.csv")
    put(f"{tag}UnknownAfterChange", int(full.records.max() - full.feasible_records.max()), f"{AUDIT}/{run}/count_reconciliation.csv")
    short_ = cr[cr.missing_units_vs_union > 0]
    put(f"{tag}ShortConditions", int(len(short_)), f"{AUDIT}/{run}/count_reconciliation.csv")
    put(f"{tag}ShortConditionNames", ", ".join(IDENT[c] for c in short_.condition) if len(short_) else "none", f"{AUDIT}/{run}/count_reconciliation.csv")
    put(f"{tag}ShortRecords", int(short_.records.min()) if len(short_) else int(full.records.max()), f"{AUDIT}/{run}/count_reconciliation.csv")
    put(f"{tag}ShortFeasible", int(short_.feasible_records.min()) if len(short_) else int(full.feasible_records.max()), f"{AUDIT}/{run}/count_reconciliation.csv")

for run, tag in ((DYN_REAL_OLD, "dr"), (DYN_SYN_OLD, "ds")):
    d = f"{AUDIT}/{run}"
    summ = json.load(open(f"{d}/audit_summary.json"))
    put(f"{tag}Cases", summ["n_cases"], d)
    put(f"{tag}Affected", summ["n_truly_affected"], d)
    put(f"{tag}NotAffected", summ["n_not_affected_but_counted"], d)
    put(f"{tag}NotAffectedShare", summ["n_not_affected_but_counted"] / summ["n_cases"], d, "pct0")
    put(f"{tag}SourceMissions", summ["selection"]["n_source_missions"], d)
    for ch, (na, nt) in summ["not_affected_by_change"].items():
        put(f"{tag}NotAff{ch.replace('_', '')}", na, d)
        put(f"{tag}Tot{ch.replace('_', '')}", nt, d)
    sel = summ["selection"].get("static_success_selected_vs_rest", {}).get("C_full")
    if sel:
        put(f"{tag}SelCfull", sel["static_success_selected"], d, "pct0")
        put(f"{tag}RestCfull", sel["static_success_not_selected"], d, "pct0")
    s = pd.read_csv(f"{d}/success_all_vs_truly_affected.csv").set_index("condition")
    for cond in s.index:
        k = cond.replace("_", "")
        put(f"{tag}All{k}", float(s.loc[cond, "all_success_rate"]), d, "pct")
        put(f"{tag}Aff{k}", float(s.loc[cond, "affected_success_rate"]), d, "pct")
        put(f"{tag}Affn{k}", int(s.loc[cond, "affected_n"]), d)
    put(f"{tag}AllN", int(s.loc["C_full", "all_n"]), d)
    put(f"{tag}AffN", int(s.loc["C_full", "affected_n"]), d)
    v = pd.read_csv(f"{d}/llm_vs_direct_on_affected_cases.csv")
    for _, x in v.iterrows():
        k = f"{tag}V{x.llm.replace('_', '')}{x.direct.replace('_', '')}"
        put(k + "n", int(x.n_cases), d)
        put(k + "llm", x.llm_success, d, "pct")
        put(k + "dir", x.direct_success, d, "pct")
        put(k + "onlyllm", int(x.only_llm), d)
        put(k + "onlydir", int(x.only_direct), d)
        put(k + "p", x.mcnemar_exact_p, d, "sci")
    # where the model wins over mechanical patching: change types
    cases = pd.read_csv(f"{d}/cases_reverified.csv")
    g = pd.read_csv(f"{d}/direct_controls_posthoc.csv")
    from src.utils.io import read_jsonl
    llm = pd.DataFrame([{"case_id": r["scenario_id"], "ok": bool(r["metrics"]["final_success"])} for r in read_jsonl(f"outputs/raw/{run}/runs.jsonl")
                        if r["seed"] == 0 and r["condition"] == "C_prescriptive"])
    mm = llm.merge(g[g.condition == "G_direct_patch"][["case_id", "final_success", "feasibility", "truly_affected"]], on="case_id").merge(cases[["case_id", "change"]], on="case_id")
    mm = mm[(mm.feasibility == "FEASIBLE") & mm.truly_affected]
    wins = mm[mm.ok & ~mm.final_success]
    put(f"{tag}WinsBudget", int(wins.change.isin(["reduce_energy", "tighten_distance"]).sum()), d)
    put(f"{tag}Wins", len(wins), d)
    bud = mm[mm.change.isin(["reduce_energy", "tighten_distance"])]
    put(f"{tag}BudgetN", len(bud), d)
    put(f"{tag}BudgetLLM", float(bud.ok.mean()) if len(bud) else 0.0, d, "pct0")
    put(f"{tag}BudgetDirect", float(bud.final_success.mean()) if len(bud) else 0.0, d, "pct0")

# dynamic table body (old logs: as published vs truly affected)
dl = []
for cond in ["A_llm_only", "D_resample", "X_binary_feedback", "X_evidence_regen", "C_full", "C_prescriptive", "E_classical", "F_astar"]:
    k = cond.replace("_", "")
    vals = [N[f"{t}{w}{k}"]["value"] for t in ("dr", "ds") for w in ("All", "Aff")]
    lab = LABEL[cond] + (" (stale plan)" if cond == "A_llm_only" else "")
    dl.append(f"{lab} & {100 * vals[0]:.1f} & \\textbf{{{100 * vals[1]:.1f}}} & {100 * vals[2]:.1f} & \\textbf{{{100 * vals[3]:.1f}}} \\\\")
for tag in ("dr", "ds"):
    pass
dl.append("\\midrule")
dl.append(f"G1$^\\dagger$ & -- & \\textbf{{{100 * N['drVCprescriptiveGdirectpatchdir']['value']:.1f}}} & -- & \\textbf{{{100 * N['dsVCprescriptiveGdirectpatchdir']['value']:.1f}}} \\\\")
open(f"{GEN}/tab_dynamic.tex", "w").write("\n".join(dl) + "\n")

# ------------------------------------------------------------------------------------------------------------------------------
# model-free dynamic study on the full benchmark (fixed code)
# ------------------------------------------------------------------------------------------------------------------------------
def latest(pattern):
    c = sorted(glob.glob(pattern))
    return c[-1] if c else None


for tag, sdir in (("mfr", "benchmark/real"), ("mfs", "benchmark/generated")):
    cand = []
    for d in glob.glob("outputs/raw/dyn-real_20261106-*"):
        man = json.load(open(f"{d}/manifest.json"))
        if man.get("provider") == "none" and man.get("completeness", {}).get("complete") and \
                any(s.startswith("real_") for s in man["case_selection"]["eligible"]) == (tag == "mfr"):
            cand.append(d)
    if not cand:
        continue
    d = sorted(cand)[-1]
    p = pd.read_csv(d.replace("/raw/", "/processed/") + "/per_run_dynamic.csv")
    p0 = p[p.detection_delay == 0]
    put(f"{tag}Run", os.path.basename(d), d)
    put(f"{tag}Missions", int(p.scenario_id.nunique()), d)
    put(f"{tag}Cases0", int(p0.case_id.nunique()), d)
    put(f"{tag}CasesAll", int(p.case_id.nunique()), d)
    f0 = p0[p0.feasibility == "FEASIBLE"]
    put(f"{tag}Feasible0", int(f0.case_id.nunique()), d)
    for cond in ("A_llm_only", "E_classical", "F_astar", "G_direct_patch"):
        put(f"{tag}{cond.replace('_', '')}", float(f0[f0.condition == cond].final_success.mean()), d, "pct")
    for ch in ("add_zone", "enlarge_zone", "tighten_distance", "reduce_energy"):
        g = f0[(f0.condition == "G_direct_patch") & (f0.change == ch)]
        put(f"{tag}G{ch.replace('_', '')}", float(g.final_success.mean()) if len(g) else 0.0, d, "pct0")
        put(f"{tag}N{ch.replace('_', '')}", int(g.case_id.nunique()), d)
    z = p[(p.condition == "A_llm_only") & p.change.isin(["add_zone", "enlarge_zone"]) & p.exposure.notna()]
    for dl_, nm in ((0.05, "five"), (0.15, "fifteen")):
        g = z[np.isclose(z.detection_delay, dl_)]
        put(f"{tag}Exposure{nm}", float(g.exposure.astype(bool).mean()) if len(g) else 0.0, d, "pct0")
        put(f"{tag}ExposureN{nm}", len(g), d)
    for dl_, nm in ((0.0, "zero"), (0.05, "five"), (0.15, "fifteen")):
        g = p[(p.condition == "G_direct_patch") & np.isclose(p.detection_delay, dl_) & (p.feasibility == "FEASIBLE")]
        put(f"{tag}GDelay{nm}", float(g.final_success.mean()) if len(g) else 0.0, d, "pct0")

# ------------------------------------------------------------------------------------------------------------------------------
# validation: oracle, geometry, energy sensitivity, simulation
# ------------------------------------------------------------------------------------------------------------------------------
ov = "outputs/review_v12/oracle_validation/oracle_validation.json"
if os.path.exists(ov):
    o = json.load(open(ov))
    put("oracleN", o["n_missions"], ov)
    put("oracleStored", o["n_stored"], ov)
    put("oracleRandom", o["n_random"], ov)
    put("oracleFeasible", o["status_counts"]["FEASIBLE"], ov)
    put("oracleInfeasible", o["status_counts"]["INFEASIBLE"], ov)
    put("oracleUnknown", o["status_counts"]["UNKNOWN"], ov)
    put("oracleUnknownRate", o["unknown_rate"], ov, "pct")
    put("oracleSoundViol", len(o["S1_violations_witness_fails_verifier"]) + len(o["S2_violations_infeasible_but_solution_found"]) + len(o["S3_violations_lower_bound_above_witness"]), ov)
    put("oracleIncomplete", len(o["unknown_but_independent_solver_succeeds"]), ov)
    put("oracleApproxMax", o["approximation_n12_vs_n96"]["max"], ov, "pct")
    put("oracleApproxMedian", o["approximation_n12_vs_n96"]["median"], ov, ".2%")
    put("oracleBracketMedian", o["bracket_ratio_ub_over_lb"]["median_rel_gap"], ov, "pct")
    csvp = ov.replace(".json", ".csv")
    oc = pd.read_csv(csvp)
    st = oc[oc.kind == "stored"]
    put("oracleStoredFeasible", int((st.status == "FEASIBLE").sum()), csvp)
    put("oracleStoredInfeasible", int((st.status == "INFEASIBLE").sum()), csvp)
    put("oracleStoredUnknown", int((st.status == "UNKNOWN").sum()), csvp)
gs_ = "outputs/review_v12/geometry_selftest.json"
if os.path.exists(gs_):
    g = json.load(open(gs_))
    put("geomRandom", g["random_scale_100"]["n_random_cases"] + g["random_scale_1e5"]["n_random_cases"], gs_)
    put("geomFN", g["random_scale_100"]["false_negatives"] + g["random_scale_1e5"]["false_negatives"], gs_)
    put("geomFP", g["random_scale_100"]["false_positives_beyond_grazing"] + g["random_scale_1e5"]["false_positives_beyond_grazing"], gs_)
    put("geomAdv", g["adversarial"]["n_adversarial_cases"], gs_)
    put("geomAdvFail", len(g["adversarial"]["failures"]), gs_)
    put("geomExact", g["exact_crosscheck"]["n_cases"], gs_)
    put("geomExactMissed", g["exact_crosscheck"]["float_missed_exact_intersection"], gs_)
    put("geomExactConservative", g["exact_crosscheck"]["float_conservative_within_tolerance"], gs_)
es = "outputs/review_v12/energy_sensitivity/oracle_energy_sensitivity.csv"
if os.path.exists(es):
    e = pd.read_csv(es)
    put("energyMissions", int(e.n_missions.iloc[0]), es)
    put("energyGrid", len(e), es)
    put("energyFixedMin", float(e.feasible_fixed_budget.min()), es, "pct0")
    put("energySlackMin", float(e.feasible_slack_preserved.min()), es, "pct0")
    pe = "outputs/review_v12/energy_sensitivity/plan_energy_robustness.csv"
    if os.path.exists(pe):
        q = pd.read_csv(pe)
        put("energyPlansN", int(q.n_successful_plans_with_energy_budget.max()), pe)
        put("energyPlansViolMax", float(q.fraction_violating_under_new_coefficients.max()), pe, "pct0")
        base = q[(q.k_v == 0.0) & (q.k_wp == 0.0)]
        dbl = q[(q.k_v == 2.0 * 30.0) & (q.k_wp == 0.0)] if False else q[(q.k_wp == 2.0)]
        put("energyPlansViolKwpTwo", float(q[q.k_wp == 2.0].fraction_violating_under_new_coefficients.max()), pe, "pct0")

for tag, d in (("simR0", "outputs/review_v12/simulation/real_margin0"), ("simR10", "outputs/review_v12/simulation/real_margin10"),
               ("simS0", "outputs/review_v12/simulation/syn_margin0"), ("simS5", "outputs/review_v12/simulation/syn_margin5")):
    f = f"{d}/executions.csv"
    if not os.path.exists(f):
        continue
    x = pd.read_csv(f)
    gap = json.load(open(f"{d}/claim_gap.json"))
    put(f"{tag}Flights", len(x), f)
    put(f"{tag}Plans", int(x.run_id.nunique()), f)
    put(f"{tag}Completed", float(x.mission_completed.mean()), f, "pct")
    put(f"{tag}LTwo", float(x.L2_executed_feasible.mean()), f, "pct")
    put(f"{tag}Intrusion", float(((x.intrusion_zones > 0) | x.collision).mean()), f, "pct")
    put(f"{tag}XtPninetyfive", float(x.cross_track_p95_m.median()), f, ".1f")
    put(f"{tag}DevMax", float(x.dev_max_m.median()), f, ".1f")
    put(f"{tag}VertMax", float(x.dev_vert_max_m.median()), f, ".1f")
    put(f"{tag}EnergyErr", float(x.energy_error_pct.median()), f, ".1f")
    put(f"{tag}MarginRejected", int(gap.get("plans_margin_not_admissible", 0)), f"{d}/claim_gap.json")
    for cond in ("C_prescriptive", "C_full", "E_classical", "G_direct_patch"):
        g = x[x.condition == cond]
        if len(g):
            put(f"{tag}{cond.replace('_', '')}LTwo", float(g.L2_executed_feasible.mean()), f, "pct")
    for w in ("calm", "crosswind"):
        g = x[x.wind == w]
        put(f"{tag}{w}LTwo", float(g.L2_executed_feasible.mean()), f, "pct")
        put(f"{tag}{w}Xt", float(g.cross_track_p95_m.median()), f, ".1f")

put("nGatedConditions", len([c for c in ORDER if c != "A_llm_only"]), "ORDER")
put("nGatedRuns", int(len(real_f[real_f.condition != "A_llm_only"]) + len(syn[syn.condition != "A_llm_only"])), "both runs", ",")

# ---- generated sentences (descriptive, numbers only) ----
es_txt = "Energy-sensitivity results were not generated (run \\texttt{scripts/energy\\_sensitivity.py})."
if os.path.exists(es):
    e = pd.read_csv(es)
    pe = "outputs/review_v12/energy_sensitivity/plan_energy_robustness.csv"
    q = pd.read_csv(pe) if os.path.exists(pe) else None
    kv = sorted(e.k_v.unique())
    kw = sorted(e.k_wp.unique())
    rel = "grid_mode" in e.columns and (e.grid_mode == "relative").all()
    kvs = ", ".join(f"{v:g}" for v in kv)
    if rel:
        grid = (f"$k_v$ scaled by $\\{{{kvs}\\}}$ and a per-waypoint overhead $k_w$ of "
                f"{', '.join(f'{100 * v:g}' for v in kw)}\\% of the mission's witness route length")
    else:
        grid = f"$k_v\\in\\{{{kvs}\\}}$ and $k_w\\in\\{{{', '.join(f'{v:g}' for v in kw)}\\}}$"
    # v12: every number of this sentence also as a macro, so the authors can write the sentence themselves
    put("enMissions", int(e.n_missions.iloc[0]), es)
    put("enFixedMin", float(e.feasible_fixed_budget.min()), es, "pct0")
    put("enFixedMax", float(e.feasible_fixed_budget.max()), es, "pct0")
    put("enSlackMin", float(e.feasible_slack_preserved.min()), es, "pct0")
    put("enKvMax", f"{max(kv):g}", es)
    put("enKwMaxPct", f"{100 * max(kw):g}", es)
    if q is not None and len(q):
        q0_, qk_ = q[q.k_wp == 0], q[q.k_wp > 0]
        put("enPlans", int(q.n_successful_plans_with_energy_budget.max()), pe)
        put("enFlipNoOverheadMax", float(q0_.fraction_violating_under_new_coefficients.max()), pe, "pct")
        put("enFlipOverheadMin", float(qk_.fraction_violating_under_new_coefficients.min()), pe, "pct0")
        put("enFlipOverheadMax", float(qk_.fraction_violating_under_new_coefficients.max()), pe, "pct0")
        put("enOverheadMinPct", f"{100 * qk_.k_wp.min():g}", pe)
        put("enOverheadMaxPct", f"{100 * qk_.k_wp.max():g}", pe)
    es_txt = (f"We re-ran the oracle on the {int(e.n_missions.iloc[0])} missions with an energy budget for every combination of {grid}. "
              f"With the budget held fixed, {100 * e.feasible_fixed_budget.min():.0f}--{100 * e.feasible_fixed_budget.max():.0f}\\% of these missions stayed feasible; with the budget "
              + ("rescaled to keep each mission's slack over its witness, all of them did." if e.feasible_slack_preserved.min() == e.feasible_slack_preserved.max() == 1.0 else
                 f"rescaled to keep each mission's slack over its witness, {100 * e.feasible_slack_preserved.min():.0f}--{100 * e.feasible_slack_preserved.max():.0f}\\% did."))
    if q is not None and len(q):
        q0 = q[q.k_wp == 0]
        qk = q[q.k_wp > 0]
        es_txt += (f" Re-verifying the {int(q.n_successful_plans_with_energy_budget.max())} released airport plans that carry an energy budget: without a waypoint overhead their verdict "
                   f"changed for at most {100 * q0.fraction_violating_under_new_coefficients.max():.1f}\\% of plans (climb weight up to {q0.k_v.max():g}$\\times$), but an overhead of "
                   f"{100 * qk.k_wp.min():g}--{100 * qk.k_wp.max():g}\\% of the route length per waypoint would have rejected "
                   f"{100 * qk.fraction_violating_under_new_coefficients.min():.0f}--{100 * qk.fraction_violating_under_new_coefficients.max():.0f}\\% of them, because detour-based plans have many waypoints. "
                   f"The energy verdict therefore depends on modeling choices in a way the zone verdict does not.")
open(f"{GEN}/energy_sentence.tex", "w").write(es_txt + "\n")

mg = []
for M_ in (15, 20):
    f = f"outputs/review_v12/simulation/real_margin{M_}/executions.csv"
    if os.path.exists(f):
        x = pd.read_csv(f)
        gap = json.load(open(f"outputs/review_v12/simulation/real_margin{M_}/claim_gap.json"))
        mg.append((M_, float(x.L2_executed_feasible.mean()), int(gap.get("plans_margin_not_admissible", 0)), int(x.run_id.nunique())))
mtxt = ", ".join(f"{M_}~m to {100 * v:.1f}\\%" + (f" ({r} plans could not be lifted within their budgets)" if r else "") for M_, v, r, _ in mg)
for M_, v, r, _ in mg:
    put(f"simRMargin{['Fifteen', 'Twenty'][[15, 20].index(M_)]}LTwo", v, f"outputs/review_v12/simulation/real_margin{M_}/executions.csv", "pct")
    put(f"simRMargin{['Fifteen', 'Twenty'][[15, 20].index(M_)]}NotLifted", r, f"outputs/review_v12/simulation/real_margin{M_}/claim_gap.json")
syn_txt = ""
f0 = "outputs/review_v12/simulation/syn_margin0/executions.csv"
if os.path.exists(f0):
    x = pd.read_csv(f0)
    x["near"] = (x.intrusion_zones > 0) | x.collision
    cm = set(x[x.condition == "C_full"].scenario_id)
    same = x[x.scenario_id.isin(cm)]
    put("synSimFlights", len(x), f0)
    put("synSimNear", float(x.near.mean()), f0, "pct")
    put("synSimNearVerifierDetours", float(x[x.condition.isin(["C_prescriptive", "G_direct_patch", "E_classical"])].near.mean()), f0, "pct")
    put("synSimNearCfull", float(x[x.condition == "C_full"].near.mean()), f0, "pct")
    put("synSimSameMissions", len(cm), f0)
    put("synSimSameNearE", float(same[same.condition == "E_classical"].near.mean()), f0, "pct")
    pm = same.groupby(["scenario_id", "condition"]).near.mean().unstack()
    put("synSimCfullLess", int((pm.C_full < pm.E_classical).sum()), f0)
    put("synSimCfullMore", int((pm.C_full > pm.E_classical).sum()), f0)
    syn_txt = (f" On the synthetic map (20~m per unit, a 5~kg quadrotor profile, \\NSynSimFlights{{}} flights of seed-0 plans) the picture changes: "
               f"\\NSynSimNear\\% of flights came within 1~m of a zone, almost all of them on plans whose detours the verifier or the planner had computed "
               f"(\\NSynSimNearVerifierDetours\\%), because shortest paths hug zone boundaries and the vehicle cuts corners. On the \\NSynSimSameMissions{{}} missions that \\cfull{{}} solved, its own "
               f"detours came that close in \\NSynSimNearCfull\\% of flights against \\NSynSimSameNearE\\% for the visibility-graph plan (fewer on \\NSynSimCfullLess{{}} missions, more on \\NSynSimCfullMore).")
    cs = "outputs/review_v12/simulation/clearance_study/clearance_summary.csv"
    if os.path.exists(cs):
        c = pd.read_csv(cs).sort_values("clearance_units")
        word = {0.0: "Default", 1.0: "Twenty", 2.0: "Forty"}            # 0 = default witness; 1 / 2 map units = 20 / 40 m
        for _, r in c.iterrows():
            w = word.get(float(r.clearance_units))
            if w is None:
                continue
            put(f"clr{w}Near", float(r.near_miss_or_collision_rate), cs, "pct")
            put(f"clr{w}LTwo", float(r.L2_rate), cs, "pct")
            put(f"clr{w}Adm", int(r.admissible_missions), cs)
            put(f"clr{w}Flights", int(r.flights), cs)
        put("clrMissions", int(c.missions.max()), cs)
        if {0.0, 1.0} <= set(c.clearance_units.astype(float)):
            syn_txt += (" We then re-planned all \\NClrMissions{} feasible synthetic missions with the visibility-graph planner, lifted every plan 5~m into its altitude "
                        "band, re-verified it against the unchanged mission and flew it under the same settings. With the default witness, whose legs may touch a "
                        "zone boundary, \\NClrDefaultNear\\% of flights came within 1~m of a zone and \\NClrDefaultLTwo\\% stayed within all modeled limits. "
                        "Requiring every leg to keep 20~m from every zone removed all near misses (\\NClrTwentyNear\\%) and raised the second rate to "
                        "\\NClrTwentyLTwo\\%. The planner still found a verified plan with this clearance for \\NClrTwentyAdm{} of the \\NClrMissions{} missions"
                        + (" (\\NClrFortyAdm{} with 40~m, again without near misses)." if 2.0 in set(c.clearance_units.astype(float)) else "."))
open(f"{GEN}/margin_sentence.tex", "w").write((f"lifting by {mtxt}." if mtxt else "") + syn_txt + "\n")

# ------------------------------------------------------------------------------------------------------------------------------
# v12.1 model-free additions: flight dynamics (turn radius, climb limits, smoothing), delayed map updates / state estimation, vehicle-parameter energy
# ------------------------------------------------------------------------------------------------------------------------------
for tag, sub in (("fdA", "airport"), ("fdS", "synthetic")):
    f_ = f"outputs/review_v12/flight_dynamics/{sub}/summary.json"
    if os.path.exists(f_):
        fd = json.load(open(f_, encoding="utf-8"))
        put(f"{tag}Radius", f"{fd['effective_turn_radius_m']:.0f}", f_)
        put(f"{tag}Plans", fd["plans"], f_)
        put(f"{tag}SmoothValid", fd["smoothed_valid_rate"], f_, "pct")
        put(f"{tag}SmoothInvalid", fd["plans"] - round(fd["smoothed_valid_rate"] * fd["plans"]), f_)
        put(f"{tag}ClimbMax", f"{fd['v_climb_max']:g}", f_)
        put(f"{tag}DescendMax", f"{fd['v_descend_max']:g}", f_)
        put(f"{tag}Margin", f"{fd['margin_m']:g}", f_)
        for mode, w in (("baseline", "Base"), ("dynamics", "Dyn"), ("dyn_margin", "DynMargin")):
            mm_ = fd["modes"].get(mode)
            if mm_:
                put(f"{tag}{w}Flights", mm_["flights"], f_)
                put(f"{tag}{w}LTwo", mm_["L2_executed_feasible"], f_, "pct")
                put(f"{tag}{w}Near", mm_["near_or_contact"], f_, "pct")
                put(f"{tag}{w}Entered", mm_["zone_entered"], f_, "pct")
                put(f"{tag}{w}Arrived", mm_["arrived"], f_, "pct")
                put(f"{tag}{w}Required", mm_["required_visited"], f_, "pct")
                put(f"{tag}{w}Xt", f"{mm_['cross_track_p95_median_m']:.1f}", f_)
f_ = "outputs/review_v12/delay_study/delay_summary.csv"
if os.path.exists(f_):
    dl_ = pd.read_csv(f_)
    W = {0.0: "Zero", 0.5: "Half", 1.0: "One", 2.0: "Two", 5.0: "Five", 15.0: "Fifteen", 30.0: "Thirty", 60.0: "Sixty"}
    for _, r in dl_.iterrows():
        pre = {"map_update_delay_s": "dlMap", "state_estimation_delay_s": "dlEst", "state_estimation_delay_uncompensated_s": "dlEstRaw"}[r.sweep] + W[float(r.value)]
        put(pre + "Success", float(r.classical_success), f_, "pct")
        put(pre + "Recovered", float(r.classical_recovered_affected) if r.classical_recovered_affected == r.classical_recovered_affected else 0.0, f_, "pct")
        put(pre + "Intrusion", float(r.classical_intrusion), f_, "pct")
        put(pre + "StaleIntrusion", float(r.stale_intrusion), f_, "pct")
        put(pre + "StaleSuccess", float(r.stale_success), f_, "pct")
        put(pre + "Episodes", int(r.episodes), f_)
f_ = "outputs/review_v12/estimation_delay/estimation_delay_summary.csv"
if os.path.exists(f_):
    ed = pd.read_csv(f_)
    for _, r in ed.iterrows():
        w = {0.0: "Zero", 0.5: "Half", 1.0: "One", 2.0: "Two", 5.0: "Five"}[float(r.tau_s)]
        put(f"ed{'Air' if r['set'] == 'airport' else 'Syn'}{w}", float(r.valid), f_, "pct")
        put(f"ed{'Air' if r['set'] == 'airport' else 'Syn'}{w}Cases", int(r.cases), f_)
f_ = "outputs/review_v12/vehicle_energy/vehicle_energy.csv"
if os.path.exists(f_):
    ve = pd.read_csv(f_)
    vs_ = ve.groupby("variant").rel_to_nominal.agg(["median", "max", "min"])
    put("veVariants", int(len(vs_) - 1), f_)
    put("vePlans", int(ve[ve.variant == "nominal"].shape[0]), f_)
    put("veRelMin", float(ve.rel_to_nominal.min()), f_, ".2f")
    put("veRelMax", float(ve.rel_to_nominal.max()), f_, ".2f")
    put("veExceed", int(ve.exceeds_battery.sum()), f_)
    put("veMaxUseNominal", float((ve[ve.variant == "nominal"].energy_wh / ve[ve.variant == "nominal"].usable_wh).max()), f_, "pct0")
    put("veMaxUseWorst", float((ve.energy_wh / ve.usable_wh).max()), f_, "pct0")
    for name, w in (("headwind 6 m/s", "Headwind"), ("payload full", "Payload"), ("lift/drag x0.8", "LiftDrag"), ("efficiency x0.9", "Eff"), ("speed x1.2", "SpeedUp")):
        if name in vs_.index:
            put(f"ve{w}Max", float(vs_.loc[name, "max"]), f_, ".2f")

# ------------------------------------------------------------------------------------------------------------------------------
# results of run_free.py (language interface, extra models, Bengaluru 3D set, mid-flight re-run, temperature check) -> Table IV + macros
# Source of truth: outputs/free_run/results_index.json (written by run_free.py). Without it, completed runs are looked up in outputs/.
# Only COMPLETE runs enter the paper (PAPER_INCLUDE_PARTIAL=1 also shows incomplete ones, marked); mock runs only with PAPER_INCLUDE_MOCK=1 (tests).
# ------------------------------------------------------------------------------------------------------------------------------
NL_ROOT = os.environ.get("PAPER_NL_ROOT", "outputs/nl_spec")
RAW_ROOT = os.environ.get("PAPER_RAW_ROOT", "outputs/raw")
INDEX = os.environ.get("PAPER_FREE_INDEX", "outputs/free_run/results_index.json")
PARTIAL = os.environ.get("PAPER_INCLUDE_PARTIAL") == "1"
idx = json.load(open(INDEX, encoding="utf-8")) if os.path.exists(INDEX) else {}
flags = {"nlsecond": False, "nlresults": False, "secondmodel": False, "georesults": False, "dynrerun": False, "tempresult": False, "extraresults": False}


def _usable(ent):
    if not ent or not ent.get("dir") or not os.path.exists(ent["dir"]):
        return False
    return bool(ent.get("complete")) or PARTIAL


def _mock_ok(man):
    return INCLUDE_MOCK or not man.get("is_mock")


def _short(model):
    return model.replace("_", "\\_")


rows_iv = []      # (label, n, {cond: rate}, partial)
COLS = ["A_llm_only", "X_binary_feedback", "C_full", "C_prescriptive", "G_direct_patch", "E_classical"]

# reference row: the main model on the airport set (Table II)
rows_iv.append((("Airport (Table~\\ref{tab:static})", man_r["model"]), int(real_f.scenario_id.nunique()),
                {c: float(real_f[real_f.condition == c].final_success.mean()) for c in COLS if c in set(real_f.condition)}, False))

# (a) language interface
spec_dir = idx.get("spec", {}).get("dir") if _usable(idx.get("spec")) else None
if spec_dir is None:                       # fallback: newest complete spec-extraction folder
    for d in sorted(glob.glob(f"{NL_ROOT}/*/summary.json")):
        s_ = json.load(open(d, encoding="utf-8"))
        man = json.load(open(os.path.join(os.path.dirname(d), "manifest.json"), encoding="utf-8"))
        if _mock_ok(man) and man.get("provider") != "truth" and (s_.get("complete") or PARTIAL):
            spec_dir = os.path.dirname(d)
if spec_dir and os.path.exists(os.path.join(spec_dir, "summary.json")):
    s_ = json.load(open(os.path.join(spec_dir, "summary.json"), encoding="utf-8"))
    man = json.load(open(os.path.join(spec_dir, "manifest.json"), encoding="utf-8"))
    if _mock_ok(man) and (man.get("provider") != "truth" or INCLUDE_MOCK):
        flags["nlresults"] = True
        put("nlModel", _short(man["model"]), spec_dir)
        put("nlN", s_["n"], spec_dir)
        for k in ("parsed", "all_fields_correct", "end_to_end_success"):
            put("nl" + k.replace("_", ""), s_[k]["rate"], spec_dir, "pct")
            put("nl" + k.replace("_", "") + "lo", s_[k]["wilson_lo"], spec_dir, "pct")
            put("nl" + k.replace("_", "") + "hi", s_[k]["wilson_hi"], spec_dir, "pct")
        recs = pd.read_json(os.path.join(spec_dir, "records.jsonl"), lines=True)
        for tag in ("Airport", "Synthetic"):            # both always defined (\\NNlEtoESynthetic is "--" if no synthetic mission was extracted)
            g = recs[(recs.source == "ourairports") == (tag == "Airport")]
            if len(g):
                put(f"nlEtoE{tag}", float(g.end_to_end_success.mean()), spec_dir, "pct")
            else:
                put(f"nlEtoE{tag}", "--", spec_dir)
            put(f"nlN{tag}", int(len(g)), spec_dir)
        if "fields" in recs:
            fa = recs[recs.parsed].fields.apply(pd.Series).mean()
            for f_, v in fa.items():
                put(f"nlField{str(f_).replace('_', '')}", float(v), spec_dir, "pct")
            worst = fa.sort_values().index[:1].tolist()
            if worst:
                put("nlWorstField", str(worst[0]).replace("_", " "), spec_dir)
                put("nlWorstFieldAcc", float(fa[worst[0]]), spec_dir, "pct")

ent2 = idx.get("spec_second")
if _usable(ent2) and os.path.exists(os.path.join(ent2["dir"], "summary.json")):
    s2 = json.load(open(os.path.join(ent2["dir"], "summary.json"), encoding="utf-8"))
    m2 = json.load(open(os.path.join(ent2["dir"], "manifest.json"), encoding="utf-8"))
    if _mock_ok(m2) and (m2.get("provider") != "truth" or INCLUDE_MOCK):
        flags["nlsecond"] = True
        put("nlTwoModel", _short(m2["model"]), ent2["dir"])
        put("nlTwoN", s2["n"], ent2["dir"])
        for k in ("parsed", "all_fields_correct", "end_to_end_success"):
            put("nlTwo" + k.replace("_", ""), s2[k]["rate"], ent2["dir"], "pct")


def _static_row(d, label, conds_expected):
    df_, man = load_runs(d)
    if not _mock_ok(man):
        return None
    df_ = df_[df_.feasibility == "FEASIBLE"]
    rates = {c: float(df_[df_.condition == c].final_success.mean()) for c in COLS if c in set(df_.condition)}
    key = df_.scenario_id + "|" + df_.seed.astype(str)
    a_ = df_[df_.condition == "C_prescriptive"].assign(k=key).set_index("k").final_success
    b_ = df_[df_.condition == "G_direct_patch"].assign(k=key).set_index("k").final_success
    ix = a_.index.intersection(b_.index)
    disc = (int((a_.loc[ix] & ~b_.loc[ix]).sum()), int((~a_.loc[ix] & b_.loc[ix]).sum()))
    unsafe = int((df_[df_.condition != "A_llm_only"].released & ~df_[df_.condition != "A_llm_only"].final_success).sum())
    return df_, man, rates, disc, unsafe


# (b) extra models on the airport set
seconds = [e for e in idx.get("second_models", []) if _usable(e)]
if not idx:                                  # fallback: any non-main, non-mock model run on the airport missions
    for mf in sorted(glob.glob(f"{RAW_ROOT}/*/manifest.json")):
        d = os.path.dirname(mf)
        man = json.load(open(mf, encoding="utf-8"))
        ids = man.get("scenario_ids") or []
        if (ids and all(x.startswith("real_") for x in ids) and man.get("model") != man_r["model"] and man.get("provider") not in ("none",)
                and man.get("kind") != "dynamic" and _mock_ok(man) and not d.endswith("__plus-direct")):
            seconds.append({"dir": d, "model": man["model"], "label": man["model"], "complete": True})
for k_, e in enumerate(seconds, 1):
    r_ = _static_row(e["dir"], e.get("label"), None)
    if r_ is None:
        continue
    df_, man, rates, disc, unsafe = r_
    w = ["One", "Two", "Three", "Four"][k_ - 1]
    flags["secondmodel"] = True
    put(f"xm{w}Model", _short(man["model"]), e["dir"])
    put(f"xm{w}Missions", int(df_.scenario_id.nunique()), e["dir"])
    for c, v in rates.items():
        put(f"xm{w}{c.replace('_', '')}", v, e["dir"], "pct")
    put(f"xm{w}OnlyLLM", disc[0], e["dir"])
    put(f"xm{w}OnlyDirect", disc[1], e["dir"])
    put(f"xm{w}Unsafe", unsafe, e["dir"])
    rows_iv.append((("Airport", man["model"]), int(df_.scenario_id.nunique()), rates, not e.get("complete", True)))

# (c) Bengaluru 3D set
geo_ent = idx.get("bengaluru")
if not _usable(geo_ent) and not idx:
    for mf in sorted(glob.glob(f"{RAW_ROOT}/*/manifest.json")):
        man = json.load(open(mf, encoding="utf-8"))
        ids = man.get("scenario_ids") or []
        if ids and all(x.startswith("blr_") for x in ids) and _mock_ok(man) and man.get("provider") != "none":
            geo_ent = {"dir": os.path.dirname(mf), "complete": True}
if _usable(geo_ent):
    r_ = _static_row(geo_ent["dir"], "bengaluru", None)
    if r_ is not None:
        df_, man, rates, disc, unsafe = r_
        flags["georesults"] = True
        put("geoModel", _short(man["model"]), geo_ent["dir"])
        put("geoMissions", int(df_.scenario_id.nunique()), geo_ent["dir"])
        for c, v in rates.items():
            put(f"geo{c.replace('_', '')}", v, geo_ent["dir"], "pct")
        put("geoUnsafe", unsafe, geo_ent["dir"])
        put("geoOnlyLLM", disc[0], geo_ent["dir"])
        put("geoOnlyDirect", disc[1], geo_ent["dir"])
        a0 = df_[df_.condition == "A_llm_only"]
        put("geoAunsafe", float((a0.released & ~a0.final_success).mean()), geo_ent["dir"], "pct")
        nb = [json.load(open(f, encoding="utf-8"))["metadata"]["n_obstacles"] for f in glob.glob("benchmark/bengaluru/*.json")]
        if nb:
            put("geoObstMin", min(nb), "benchmark/bengaluru")
            put("geoObstMax", max(nb), "benchmark/bengaluru")
        rows_iv.append((("Bengaluru 3D", man["model"]), int(df_.scenario_id.nunique()), rates, not geo_ent.get("complete", True)))

# (d) mid-flight re-runs with the model (random samples, corrected code): airport (prefix drr) and synthetic (prefix drs)
for key_, pre, lab_ in (("dynamic", "drr", "airport"), ("dynamic_synthetic", "drs", "synthetic")):
    dyn_ent = idx.get(key_)
    if not _usable(dyn_ent):
        continue
    with open(os.path.join(dyn_ent["dir"], "runs.jsonl"), encoding="utf-8") as fh:
        recs = [json.loads(l) for l in fh if l.strip()]
    man = json.load(open(os.path.join(dyn_ent["dir"], "manifest.json"), encoding="utf-8"))
    if not recs or not _mock_ok(man):
        continue

    def _succ(r):
        fv = r.get("final_verification") or {}
        return bool(r["metrics"].get("released")) and bool(fv.get("valid")) and bool(fv.get("success"))
    feas = [r for r in recs if r["scenario_meta"].get("feasibility") == "FEASIBLE"]
    rates = {}
    for c in COLS:
        g = [r for r in feas if r["condition"] == c]
        if g:
            rates[c] = sum(_succ(r) for r in g) / len(g)
    cases = len({r["scenario_id"] for r in feas})
    flags["dynrerun"] = True
    put(f"{pre}Model", _short(man["model"]), dyn_ent["dir"])
    put(f"{pre}Cases", cases, dyn_ent["dir"])
    put(f"{pre}Missions", len({(r.get("dynamic") or {}).get("orig_scenario") or r["scenario_id"].split("__")[0] for r in feas}), dyn_ent["dir"])
    for c, v in rates.items():
        put(f"{pre}{c.replace('_', '')}", v, dyn_ent["dir"], "pct")
    pres = {r["scenario_id"]: _succ(r) for r in feas if r["condition"] == "C_prescriptive"}
    g1 = {r["scenario_id"]: _succ(r) for r in feas if r["condition"] == "G_direct_patch"}
    common = set(pres) & set(g1)
    ol, od = sum(pres[k] and not g1[k] for k in common), sum(g1[k] and not pres[k] for k in common)
    put(f"{pre}OnlyLLM", ol, dyn_ent["dir"])
    put(f"{pre}OnlyDirect", od, dyn_ent["dir"])
    from scipy.stats import binomtest as _bt
    put(f"{pre}McNemarP", float(_bt(ol, ol + od).pvalue) if ol + od else 1.0, dyn_ent["dir"], "p")
    rows_iv.append(((f"Re-run, {lab_}", man["model"]), cases, rates, not dyn_ent.get("complete", True)))

# (e) temperature check
tc = idx.get("tempcheck")
if tc and tc.get("dir") and os.path.isfile(tc["dir"]):
    rep = json.load(open(tc["dir"], encoding="utf-8"))["report"]
    flags["tempresult"] = True
    v_ = rep.get("verdict", "").split(" (")[0]
    put("tempVerdict", v_.replace("_", " ").lower(), tc["dir"])
    n_t, d0_, d7_ = int(rep.get("n_per_temperature", 0)), int(rep.get("distinct_T0", 0)), int(rep.get("distinct_T0.7", 0))
    put("tempN", {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}.get(n_t, str(n_t)), tc["dir"])
    put("tempDistinctZero", d0_, tc["dir"])
    put("tempDistinctSeven", d7_, tc["dir"])
    # one sentence the paper prints after "A direct check (the same prompt N times at each temperature) ..." (American spelling, no verdict codes)
    put("tempSentence", {
        "HONOURED": f"gave {d7_} distinct replies at 0.7 against {d0_} at 0, so the temperature was applied",
        "INCONCLUSIVE-LIKELY-IGNORED": "gave identical replies at both temperatures, so we do not attribute D's results to temperature",
        "NOT-HONOURED-AT-0": f"gave {d0_} distinct replies at 0 and {d7_} at 0.7, so temperature had no visible effect and we do not attribute D's results to it",
    }.get(v_, "was inconclusive, so we do not attribute D's results to temperature"), tc["dir"])

flags["extraresults"] = len(rows_iv) > 1
short_c = {"A_llm_only": "A", "C_full": "\\cfull", "C_prescriptive": "\\cpre", "G_direct_patch": "G1", "E_classical": "E"}
TAB_COLS = list(short_c)          # X_binary appears only in the first row, which repeats Table II
lines = ["\\begin{tabular}{@{}llr" + "r" * len(TAB_COLS) + "@{}}", "\\toprule",
         "Mission set & Model & $n$ & " + " & ".join(short_c[c] for c in TAB_COLS) + "\\\\", "\\midrule"]
models_iv = []                                   # M1 = the main model, M2 = the extra model (legend under the table keeps the column narrow)
for (set_, model_), n_, rates, partial in rows_iv:
    if model_ not in models_iv:
        models_iv.append(model_)
    cells = [f"{100 * rates[c]:.1f}" if c in rates else "--" for c in TAB_COLS]
    star = "$^\\ast$" if partial else ""
    lines.append(f"{set_}{star} & M{models_iv.index(model_) + 1} & {n_} & " + " & ".join(cells) + "\\\\")
lines += ["\\bottomrule", "\\end{tabular}\\\\[2pt]"]
legend = "; ".join(f"M{i + 1}: {_short(m_)}" for i, m_ in enumerate(models_iv))
if any(r[3] for r in rows_iv):
    legend += "; $^\\ast$run not complete"
lines.append("\\parbox{\\columnwidth}{\\scriptsize " + legend + ".}")
open(f"{GEN}/tab_extra.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")
for fn in ("optional_language_interface.tex", "optional_nl_only.tex", "optional_second_model.tex", "optional_geo.tex"):
    open(f"{GEN}/{fn}", "w", encoding="utf-8").write("% superseded by tab_extra.tex and the macros (v12)\n")
with open(f"{GEN}/flags.tex", "w", encoding="utf-8") as fh:
    for k, v in flags.items():
        fh.write(f"\\newif\\if{k}\\{k}{'true' if v else 'false'}\n")

# ------------------------------------------------------------------------------------------------------------------------------
# figure: success by condition
# ------------------------------------------------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.linewidth": 0.6})
show = ["A_llm_only", "D_resample", "X_binary_feedback", "X_localized", "C_full", "X_evidence_regen", "C_prescriptive", "G_direct_patch", "G_direct_full_witness", "E_classical"]
short = {k: v.replace("\\_", "_") for k, v in IDENT.items()}      # the same identifiers as Table II
t = tab.set_index("condition").loc[show]
fig, ax = plt.subplots(figsize=(3.45, 1.95))
x = np.arange(len(show))
w = 0.38
cols = ["#4C72B0" if c.startswith(("A_", "D_", "X_", "C_")) else "#999999" for c in show]
ax.bar(x - w / 2, 100 * t.real, w, yerr=[100 * (t.real - t.real_lo), 100 * (t.real_hi - t.real)], color=cols, edgecolor="black", linewidth=0.4, capsize=1.5, error_kw={"elinewidth": 0.5}, label="Airport (n=53)")
ax.bar(x + w / 2, 100 * t.syn, w, yerr=[100 * (t.syn - t.syn_lo), 100 * (t.syn_hi - t.syn)], color=cols, edgecolor="black", linewidth=0.4, hatch="////", capsize=1.5, error_kw={"elinewidth": 0.5}, alpha=0.6, label="Synthetic (112$\\times$3)")
ax.set_xticks(x)
ax.set_xticklabels([short[c] for c in show], rotation=35, ha="right")
ax.set_ylabel("Final success (%)")
ax.set_ylim(0, 128)
ax.set_yticks([0, 20, 40, 60, 80, 100])
ax.axvline(6.5, color="black", lw=0.5, ls=":")
ax.text(3.0, 116, "LLM repair conditions", ha="center", fontsize=6.3)
ax.text(8.0, 116, "no LLM after plan 1", ha="center", fontsize=6.3)
ax.legend(fontsize=5.8, loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, ncol=2)
ax.grid(axis="y", lw=0.3, alpha=0.5)
fig.tight_layout(pad=0.2)
fig.savefig(f"{FIG}/fig_success.pdf")
plt.close(fig)

# ------------------------------------------------------------------------------------------------------------------------------
json.dump(N, open(f"{GEN}/numbers.json", "w"), indent=1, default=str)
out_lines = ["% generated by scripts/make_paper_numbers.py -- do not edit"]
seen = set()
for name, s_ in TEX:
    mn = macro_name(name)
    if mn in seen:
        raise SystemExit(f"duplicate macro {mn} ({name}); macros.tex left unchanged")
    seen.add(mn)
    import re as _re
    s_ = _re.sub(r"(?<!\\)%", r"\\%", s_)       # a bare % would start a LaTeX comment
    out_lines.append(f"\\newcommand{{{mn}}}{{{s_}}}")
with open(f"{GEN}/macros.tex", "w") as fh:          # written only after the whole set is known to be consistent
    fh.write("\n".join(out_lines) + "\n")
print(f"{len(N)} numbers -> {GEN}/numbers.json, macros.tex; tables tab_static.tex, tab_dynamic.tex; figure {FIG}/fig_success.pdf")

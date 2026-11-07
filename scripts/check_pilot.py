"""Check a PILOT run against the sanity gates and print a shareable report (also saved as pilot_report.txt next to the run).
Usage: python scripts/check_pilot.py outputs/raw/<run folder>        (works for static runs; dynamic runs: see --dynamic)"""
import argparse, collections, json, os, sys

ap = argparse.ArgumentParser(); ap.add_argument("run_dir"); args = ap.parse_args()
RUN = args.run_dir
man = json.load(open(os.path.join(RUN, "manifest.json")))
path = os.path.join(RUN, "runs.jsonl")
runs = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
out = []
P = out.append
gate = lambda ok, text: P(f"  [{'PASS' if ok else 'FAIL'}] {text}")

P(f"PILOT REPORT  run={man['experiment_run_id']}  provider={man.get('provider')}  model={man.get('model')}  prompt={man.get('prompt_version')}")
P(f"seeds={man.get('seeds')}  scenarios={man.get('n_scenarios')}  runs={len(runs)}  started={man.get('started')}  finished={man.get('finished')}")
P("\nGATE A: run integrity")
gate(not man.get("is_mock"), "real model (not mock)")
gate(len(runs) > 0, f"runs saved ({len(runs)})")
gate(not man.get("stopped_reason"), f"run finished cleanly (stopped_reason={str(man.get('stopped_reason'))[:90]})")
gate(man.get("prompt_version") == "plan-v3", f"prompt_version == plan-v3 (got {man.get('prompt_version')})")
gate(man.get("env", {}).get("git_commit") not in (None, ""), f"git commit recorded ({man.get('env', {}).get('git_commit')}): run 'git init; git add -A; git commit' first")
if not runs:
    P("\nNo runs to analyse. Fix GATE A first."); txt = "\n".join(out); print(txt); open(os.path.join(RUN, "pilot_report.txt"), "w").write(txt); sys.exit(0)

calls = [c for r in runs for it in r["iterations"] for c in it.get("calls", [])]
perr = sum(1 for c in calls if c.get("parse_error")); norm = sum(1 for c in calls if c.get("normalized"))
lat = [c["latency_s"] for c in calls if c.get("latency_s")]
P("\nGATE B: format reliability")
gate(len(calls) == 0 or perr / len(calls) <= 0.10, f"parse-error rate {perr}/{len(calls)} = {100*perr/max(1,len(calls)):.0f}% (limit 10%)")
P(f"  info: format-normalized calls = {norm}/{len(calls)} (report this number in the paper); mean latency = {sum(lat)/max(1,len(lat)):.1f}s; total calls = {len(calls)}")
errs = collections.Counter(c["parse_error"][:70] for c in calls if c.get("parse_error"))
for e, n in errs.most_common(3):
    P(f"    parse error x{n}: {e}")

rs = [c["temperature"] for r in runs if r["condition"] == "D_resample" for it in r["iterations"][1:] for c in it.get("calls", [])]
P("\nGATE C: design validity")
gate(all(abs(t - 0.7) < 1e-9 for t in rs) if rs else True, f"D_resample samples at temperature 0.7 (seen: {sorted(set(rs)) or 'no resample calls'})")
bad_oracle = [r["scenario_id"] + "/" + r["condition"] for r in runs if r["metrics"].get("oracle_inconsistent")]
gate(not bad_oracle, f"oracle consistent (no valid plan on an INFEASIBLE mission): {bad_oracle[:3]}")
for cond in ("E_classical", "F_astar"):
    rr = [r for r in runs if r["condition"] == cond and r["feasibility"] and (r["feasibility"] if isinstance(r["feasibility"], str) else r["feasibility"].get("status")) == "FEASIBLE"]
    if rr:
        ok = sum(1 for r in rr if r["metrics"].get("final_success"))
        gate(ok == len(rr), f"{cond} solves every FEASIBLE mission ({ok}/{len(rr)}); if not, the verifier/oracle disagree: BUG")

P("\nPER-SCENARIO RESULTS  (final success: Y/n; calls = LLM calls used; iter0 = valid on the FIRST attempt)")
conds = [c for c in man["conditions"] if c in {r["condition"] for r in runs}]
by = {(r["scenario_id"], r["condition"]): r for r in runs}
sids = sorted({r["scenario_id"] for r in runs})
P(f"  {'scenario':26s}" + "".join(f"{c[:14]:>15s}" for c in conds))
for s in sids:
    row = f"  {s:26s}"
    for c in conds:
        r = by.get((s, c))
        if not r: row += f"{'-':>15s}"; continue
        m = r["metrics"]; ok = "Y" if m.get("final_success") else "n"
        row += f"{(ok + '/' + str(m.get('n_llm_calls', 0)) + 'c' + ('*' if m.get('released') and not m.get('final_success') else '')):>15s}"
    P(row)
P("  (* = an INVALID plan was released as if it were fine: 'unsafe release')")

P("\nSUMMARY BY CONDITION (feasible missions only)")
P(f"  {'condition':20s}{'n':>4s}{'1st-try ok':>12s}{'final ok':>10s}{'unsafe rel.':>13s}{'mean calls':>12s}")
summ = {}
for c in conds:
    rr = [r for r in runs if r["condition"] == c and (r["feasibility"] if isinstance(r["feasibility"], str) else (r["feasibility"] or {}).get("status")) == "FEASIBLE"]
    if not rr: continue
    f = lambda k: sum(1 for r in rr if r["metrics"].get(k))
    unsafe = sum(1 for r in rr if r["metrics"].get("released") and not r["metrics"].get("final_success"))
    summ[c] = (len(rr), f("initial_success"), f("final_success"))
    P(f"  {c:20s}{len(rr):4d}{f('initial_success'):12d}{f('final_success'):10d}{unsafe:13d}{sum(r['metrics'].get('n_llm_calls', 0) for r in rr)/len(rr):12.1f}")
inf = [r for r in runs if (r["feasibility"] if isinstance(r["feasibility"], str) else (r["feasibility"] or {}).get("status")) == "INFEASIBLE" and r["condition"] not in ("E_classical", "F_astar")]
if inf:
    ver = [r for r in inf if r["condition"] != "A_llm_only"]
    rel_a = sum(1 for r in inf if r["condition"] == "A_llm_only" and r["metrics"].get("released"))
    rel_v = sum(1 for r in ver if r["metrics"].get("released"))
    P(f"  INFEASIBLE missions: A_llm_only released {rel_a} plan(s) (expected: it never checks); verified conditions released {rel_v}/{len(ver)} (ideal: 0)")

P("\nINTERPRETATION HINTS")
sa = summ.get("A_llm_only"); sc = summ.get("C_full")
if sa:
    P(f"  A_llm_only first-try success {sa[1]}/{sa[0]}: " + ("TOO EASY: the model solves nearly everything unaided; add harder scenarios or a weaker model" if sa[1] >= 0.9 * sa[0]
      else "TOO HARD: almost nothing is valid without repair; check prompt/format before concluding" if sa[1] <= 0.1 * sa[0] else "good range: there is room for repair to help"))
if sa and sc:
    P(f"  Repair gain (C_full final - A_llm_only first try): {sc[2] - sa[1]:+d} of {sc[0]} missions")
P("  Pilot-scale numbers (n = 8 feasible missions) are for debugging only: do NOT put them in the paper.")
txt = "\n".join(out); print(txt)
open(os.path.join(RUN, "pilot_report.txt"), "w", encoding="utf-8").write(txt)

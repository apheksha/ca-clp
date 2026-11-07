#!/usr/bin/env python
"""Independent cross-check of the paper (v12). Recomputes key numbers from the RAW files with code that does not use src/evaluation/*, compares them with
paper/generated/numbers.json, and checks the descriptive facts that appear as plain text in paper.tex (dataset composition, coefficients, budgets).
Exit code 1 on any mismatch.   python scripts/check_paper_numbers.py"""
import glob, json, math, os, re, sys
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
N = {k: v["value"] for k, v in json.load(open("paper/generated/numbers.json")).items()}
TEX = open("paper/paper.tex", encoding="utf-8").read()
bad, ok = [], 0


def check(name, got, want, tol=1e-9):
    global ok
    if isinstance(want, float) or isinstance(got, float):
        good = abs(float(got) - float(want)) <= tol
    else:
        good = got == want
    if good:
        ok += 1
    else:
        bad.append(f"{name}: independent={got!r} paper={want!r}")


def jl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def success(r):
    fv = r.get("final_verification") or {}
    return bool(r["metrics"].get("released")) and bool(fv.get("valid")) and bool(fv.get("success"))


# ---- static runs: per-condition success counted directly ----
for tag, run in (("real", "outputs/raw/real_20261025-194958_3b2644_seed0-only__plus-direct"), ("syn", "outputs/raw/real_20261023-181728_55def8__plus-direct")):
    k, n = Counter(), Counter()
    unsafe = Counter()
    zone_first = Counter()
    for r in jl(f"{run}/runs.jsonl"):
        if r["scenario_meta"]["feasibility"] != "FEASIBLE":
            continue
        c = r["condition"]
        n[c] += 1
        k[c] += success(r)
        unsafe[c] += bool(r["metrics"].get("released")) and not success(r)
        v0 = r["iterations"][0]["verification"] or {}
        zone_first[c] += any(v["family"] == "no_fly_zone" for v in v0.get("violations", []))
    for c in n:
        check(f"{tag} {c} success", k[c] / n[c], N[f"{tag}{c.replace('_', '')}"])
    check(f"{tag} A unsafe", unsafe["A_llm_only"] / n["A_llm_only"], N[f"{tag}Aunsafe"])
    check(f"{tag} A zone first plans", zone_first["A_llm_only"] / n["A_llm_only"], N[f"{tag}Azone"])
    check(f"{tag} gated unsafe", sum(v for c, v in unsafe.items() if c != "A_llm_only"), 0)
    if tag == "real":
        check("real missions", n["C_full"], N["realMissions"])
    else:
        check("synthetic runs per condition", n["C_full"], N["synRuns"])

# ---- replication runs (5-6 Nov 2026) ----
rk, rn = Counter(), Counter()
pres, g1 = {}, {}
for run in sorted(glob.glob("outputs/raw/real_2026110[56]-*")):
    man = json.load(open(f"{run}/manifest.json"))
    if man.get("provider") != "gemini" or man.get("data_sources") != ["synthetic"]:
        continue
    for r in jl(f"{run}/runs.jsonl"):
        if r["scenario_meta"]["feasibility"] != "FEASIBLE":
            continue
        rn[r["condition"]] += 1
        rk[r["condition"]] += success(r)
        if r["condition"] == "C_prescriptive":
            pres[(r["scenario_id"], r["seed"])] = success(r)
        if r["condition"] == "G_direct_patch":
            g1[(r["scenario_id"], r["seed"])] = success(r)
for c in rn:
    check(f"replication {c}", rk[c] / rn[c], N[f"rep{c.replace('_', '')}"])
check("replication only-LLM", sum(pres[k] and not g1[k] for k in pres), N["repOnlyLLM"])
check("replication only-G1", sum(g1[k] and not pres[k] for k in pres), N["repOnlyDirect"])
check("replication runs per condition", rn["C_full"], 336)

# ---- dynamic audit: re-verify stale plans independently of dynamic_audit.py ----
from src.models.schemas import Mission, FlightPlan
from src.verifier.verifier import verify
for tag, run in (("dr", "dyn-real_20261025-141536_7aae72_seed0-only"), ("ds", "dyn-real_20261025-173134_6d86b6")):
    aff, cases = {}, set()
    succ = defaultdict(lambda: [0, 0])
    recs = list(jl(f"outputs/raw/{run}/runs.jsonl"))
    for r in recs:
        cid = r["scenario_id"]
        if cid not in aff:
            m = Mission.from_dict(r["dynamic"]["mission"])
            p = FlightPlan.from_dict(r["initial_plan"])
            p.mission_id = m.mission_id
            aff[cid] = not verify(m, p).success
    for r in recs:
        if r["scenario_meta"]["feasibility"] == "FEASIBLE" and aff[r["scenario_id"]]:
            s = succ[r["condition"]]
            s[0] += success(r)
            s[1] += 1
    check(f"{tag} cases", len(aff), N[f"{tag}Cases"])
    check(f"{tag} not affected", sum(not v for v in aff.values()), N[f"{tag}NotAffected"])
    for c, (a, b) in succ.items():
        check(f"{tag} affected {c}", a / b, N[f"{tag}Aff{c.replace('_', '')}"])

# ---- simulation ----
import csv
for tag, d in (("simR0", "real_margin0"), ("simR10", "real_margin10")):
    rows = list(csv.DictReader(open(f"outputs/review_v12/simulation/{d}/executions.csv")))
    l2 = sum(r["L2_executed_feasible"] == "True" for r in rows) / len(rows)
    check(f"{tag} L2", l2, N[f"{tag}LTwo"])
    check(f"{tag} flights", len(rows), N[f"{tag}Flights"])
    check(f"{tag} intrusions", sum(int(r["intrusion_zones"]) > 0 or r["collision"] == "True" for r in rows), 0)

rows = list(csv.DictReader(open("outputs/review_v12/simulation/syn_margin0/executions.csv")))
near = [int(r["intrusion_zones"]) > 0 or r["collision"] == "True" for r in rows]
check("synthetic sim flights", len(rows), N["synSimFlights"])
check("synthetic sim near-miss rate", sum(near) / len(rows), N["synSimNear"])
rows = list(csv.DictReader(open("outputs/review_v12/simulation/clearance_study/clearance_executions.csv")))
for c, w in (("0.0", "Default"), ("1.0", "Twenty"), ("2.0", "Forty")):
    fl = [r for r in rows if r["clearance_units"] == c and r["admissible"] == "True"]
    check(f"clearance {w} flights", len(fl), N[f"clr{w}Flights"])
    check(f"clearance {w} near misses", sum(int(float(r["intrusion_zones"])) > 0 or r["collision"] == "True" for r in fl) / len(fl), N[f"clr{w}Near"])
    check(f"clearance {w} L2", sum(r["L2_executed_feasible"] == "True" for r in fl) / len(fl), N[f"clr{w}LTwo"])
    check(f"clearance {w} solvable missions", len({r["scenario_id"] for r in fl}), N[f"clr{w}Adm"])
check("clearance study covers all feasible synthetic missions", len({r["scenario_id"] for r in rows}), N["clrMissions"])
check("phrase present: 20 m leg clearance", "keep 20~m from every zone" in TEX, True)

# ---- v12.1 model-free additions, recomputed from the per-flight / per-case files ----
import pandas as _pd
fl = _pd.read_csv("outputs/review_v12/flight_dynamics/airport/flights.csv")
fl = fl[fl.admissible == True]  # noqa: E712
check("FD airport arrived (dynamics)", float(fl[fl["mode"] == "dynamics"].arrived.mean()), N["fdADynArrived"])
check("FD airport arrived (baseline)", float(fl[fl["mode"] == "baseline"].arrived.mean()), N["fdABaseArrived"])
fs = _pd.read_csv("outputs/review_v12/flight_dynamics/synthetic/flights.csv")
fs = fs[fs.admissible == True]  # noqa: E712
check("FD synthetic near (dynamics)", float(fs[fs["mode"] == "dynamics"].near_or_contact.mean()), N["fdSDynNear"])
gs = _pd.read_csv("outputs/review_v12/flight_dynamics/synthetic/smoothed_geometry.csv")
check("FD synthetic smoothed valid", float(gs.smoothed_valid.mean()), N["fdSSmoothValid"])
ed = _pd.read_csv("outputs/review_v12/estimation_delay/estimation_delay.csv")
ed = ed[ed.planned == True]  # noqa: E712
check("estimation delay airport 5 s", float(ed[(ed["set"] == "airport") & (ed.tau_s == 5.0)].valid_from_true_position.mean()), N["edAirFive"])
ve = _pd.read_csv("outputs/review_v12/vehicle_energy/vehicle_energy.csv")
check("vehicle energy: no plan over battery", int((ve.energy_wh > ve.usable_wh).sum()), N["veExceed"])
for c, rr in (("C_full", "real"), ("C_prescriptive", "real")):
    check(f"Table II withheld {c}", 1 - N[f"{rr}{c.replace('_', '')}"] - N[f"{rr}{c.replace('_', '')}unsaferate"], N[f"{rr}{c.replace('_', '')}withheld"], 1e-9)

# ---- oracle / geometry ----
o = json.load(open("outputs/review_v12/oracle_validation/oracle_validation.json"))
check("oracle n", o["n_missions"], N["oracleN"])
check("oracle unsound", len(o["S1_violations_witness_fails_verifier"]) + len(o["S2_violations_infeasible_but_solution_found"]), 0)

# ---- descriptive facts written as plain text in paper.tex ----
real = [json.load(open(f)) for f in sorted(glob.glob("benchmark/real/*.json"))]
syn = [json.load(open(f)) for f in sorted(glob.glob("benchmark/generated/*.json"))]
dc = Counter(s["difficulty"] for s in real)
check("Table I: 18 easy/18 medium/17 hard", (dc["real_easy"], dc["real_medium"], dc["real_hard"]), (18, 18, 17))
check("Table I: zones per airport mission 2-6", (min(len(s["mission"]["no_fly_zones"]) for s in real), max(len(s["mission"]["no_fly_zones"]) for s in real)), (2, 6))
radii = {z["radius"] for s in real for z in s["mission"]["no_fly_zones"]}
check("Table I: zone radii 8/5 km", sorted(radii), [5000.0, 8000.0])
check("regions = 9", len({s["metadata"]["region"] for s in real}), 9)
for reg in ("chicago", "dubai", "tokyo", "london", "sydney"):
    check(f"region {reg} present", reg in {s["metadata"]["region"] for s in real}, True)
sl = {s["difficulty"]: s["metadata"]["distance_slack"] for s in real}
check("distance slack 1.40/1.20/1.08", (sl["real_easy"], sl["real_medium"], sl["real_hard"]), (1.4, 1.2, 1.08))
es = {s["difficulty"]: s["metadata"]["energy_slack"] for s in real}
check("energy slack -/1.30/1.12", (es["real_easy"], es["real_medium"], es["real_hard"]), (None, 1.3, 1.12))
bands = {s["difficulty"]: s["mission"]["max_altitude"] - s["mission"]["min_altitude"] for s in real}
check("altitude bands 100/70/40 m", (bands["real_easy"], bands["real_medium"], bands["real_hard"]), (100.0, 70.0, 40.0))
check("ceiling <= 120 m", max(s["mission"]["max_altitude"] for s in real), 120.0)
check("hard missions >= 2 blocked legs", min(s["metadata"]["n_blocked_legs"] for s in real if s["difficulty"] == "real_hard") >= 2, True)
check("k_v = 30 for airport energy missions", {s["mission"]["energy_k_v"] for s in real if s["mission"].get("max_energy")}, {30.0})
check("k_v = 1 for synthetic missions with an energy budget", {s["mission"]["energy_k_v"] for s in syn if s["mission"].get("max_energy") and "energy" in s["mission"]["active"]}, {1.0})
check("k_w = 0 everywhere", {s["mission"]["energy_k_wp"] for s in real + syn}, {0.0})
sc = Counter(s["difficulty"] for s in syn)
check("synthetic families", (sc["easy"] + sc["medium"] + sc["hard"], sc["corridor"], sc["combo"], sc["infeasible"]), (60, 10, 42, 6))
check("synthetic zones 0-7", (min(len(s["mission"]["no_fly_zones"]) for s in syn), max(len(s["mission"]["no_fly_zones"]) for s in syn)), (0, 6))
dsl = [s["metadata"]["distance_slack"] for s in syn if s["metadata"].get("distance_slack")]
check("synthetic distance slack within 1.04-1.80", (min(dsl) >= 1.04, max(dsl) <= 1.80), (True, True))
check("full-height zones in all experiments", all(z.get("z_min") is None and z.get("z_max") is None for s in real + syn for z in s["mission"]["no_fly_zones"]), True)
m = json.load(open("outputs/raw/real_20261025-194958_3b2644_seed0-only/manifest.json"))
check("prompt plan-v3", m["prompt_version"], "plan-v3")
check("temperature 0", m["planner_config"]["temperature"], 0.0)
check("D temperature 0.7", m["conditions"]["D_resample"]["temperature"], 0.7)
check("3 repair rounds", m["conditions"]["C_full"]["max_iterations"], 3)
check("run dates 23-26 Oct", (m["started"][:10], json.load(open("outputs/raw/real_20261023-181728_55def8/manifest.json"))["started"][:10]), ("2026-10-25", "2026-10-23"))
from src.benchmark.feasibility import CIRCLE_SIDES, LB_CIRCLE_SIDES
check("12-gon / 24-gon", (CIRCLE_SIDES, LB_CIRCLE_SIDES), (12, 24))
check("2.35% bound stated in the paper", "2.35\\%" in TEX and abs(100 * (math.tan(math.pi / 12) / (math.pi / 12) - 1) - 2.35) < 0.01, True)
dyn = json.load(open("outputs/raw/dyn-real_20261025-141536_7aae72_seed0-only/manifest.json"))["experiment_config"]["dynamic"]
check("dynamic progress 35/65%", dyn["progress"], [0.35, 0.65])
for phrase in ("Holm correction over ten comparisons", "2000 resamples", "20{,}000 draws", "200 cells", "0.75 cell", "1.25--1.8 times", "1.03 times"):
    check(f"phrase present: {phrase}", phrase in TEX, True)
import inspect
from src.dynamic import changes
src_ = inspect.getsource(changes)
check("enlarge factors 1.25-1.8", "(1.8, 1.5, 1.25)" in src_, True)
check("distance cut 1.03", "witness_distance * 1.03" in src_, True)
from src.planner import astar
check("A* 200 cells", inspect.signature(astar.astar_plan).parameters["resolution"].default, 200)
check("A* inflation 0.75", "0.75 * cell" in inspect.getsource(astar), True)

print(f"{ok} checks passed, {len(bad)} failed")
for b in bad:
    print("  MISMATCH", b)
sys.exit(1 if bad else 0)

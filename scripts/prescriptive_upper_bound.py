"""LLM-free sanity check of the PRESCRIPTIVE feedback: simulate a PERFECT instruction follower.
Start from a naive straight-line plan (ignores zones), then each round apply EXACTLY what the feedback's suggested_detour says.
If the feedback is correct, nearly every mission should be valid after 1-3 rounds. Anything unsolved = a bug or a genuinely hard geometry case.
Usage: python scripts/prescriptive_upper_bound.py [benchmark dir]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.schemas import Mission, FlightPlan, Waypoint
from src.benchmark.generator import load_scenarios
from src.benchmark.feasibility import assess, visiting_order
from src.verifier.verifier import verify
from src.feedback.feedback import build_feedback

def naive(m):
    alt = (m.min_altitude + m.max_altitude) / 2
    ids = visiting_order(m) or [r.id for r in m.required_waypoints]
    by = {r.id: r for r in m.required_waypoints}
    w = [Waypoint(i, by[i].x, by[i].y, alt, k + 1) for k, i in enumerate(ids)]
    return FlightPlan(m.mission_id, w, Waypoint("DEST", m.destination[0], m.destination[1], alt, len(w) + 1))

def follow(plan, fb, counter):
    wps = [Waypoint(w.id, w.x, w.y, w.altitude, 0) for w in plan.ordered()]
    ops = []
    for v in fb.payload["violations"]:
        d = v.get("suggested_detour")
        if not d: continue
        if d["action"] == "insert":
            pos = -1 if d["between"][0] == "START" else next(i for i, w in enumerate(wps) if w.id == d["between"][0])
            ops.append((pos, 0, d["new_waypoints_in_order"]))                       # insert AFTER pos
        else:
            pos = next((i for i, w in enumerate(wps) if w.id == d["waypoint"]), None)
            if pos is not None: ops.append((pos, 1, d["with_waypoints_in_order"]))  # replace pos
    for pos, kind, new in sorted(ops, key=lambda o: -o[0]):
        nw = []
        for p in new:
            counter[0] += 1; nw.append(Waypoint(f"D{counter[0]}", p["x"], p["y"], p["altitude"], 0))
        wps[pos + 1:pos + 1] = nw if kind == 0 else []
        if kind == 1: wps[pos:pos + 1] = nw
    wps = [Waypoint(w.id, w.x, w.y, w.altitude, i + 1) for i, w in enumerate(wps)]
    return FlightPlan(plan.mission_id, wps, plan.destination)

src = sys.argv[1] if len(sys.argv) > 1 else "benchmark/generated"
tot = solved = naive_ok = nfz_clear = nfz_needed = 0; rounds = {}; bad = []
for s in load_scenarios(src):
    m = Mission.from_dict(s["mission"])
    if assess(m).status != "FEASIBLE": continue
    tot += 1; plan = naive(m); cnt = [0]; done = None
    for rd in range(0, 4):
        res = verify(m, plan)
        if res.success: done = rd; break
        if rd == 3: break
        fb = build_feedback(m, plan, res, "prescriptive", "targeted")
        new = follow(plan, fb, cnt)
        if [ (w.x, w.y) for w in new.ordered()] == [(w.x, w.y) for w in plan.ordered()]: break      # no actionable suggestion -> stuck
        plan = new
    fin = {v.type for v in verify(m, plan).all_violations()}
    nfz_clear += not any(t.startswith('NFZ') for t in fin)
    if done is None: bad.append((s["scenario_id"], sorted({v.type for v in verify(m, plan).all_violations()})))
    else: solved += 1; rounds[done] = rounds.get(done, 0) + 1
print(f"feasible missions: {tot} | solved by a perfect follower: {solved} | rounds needed: {dict(sorted(rounds.items()))}")
print(f"NFZ violations fully removed by following the detours: {nfz_clear}/{tot}  (the rest of the unsolved cases fail on ENERGY/DISTANCE, which a detour-only follower does not address)")
print(f"unsolved ({len(bad)}):", bad[:6])

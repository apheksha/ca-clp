"""Kinematic plausibility audit of saved FINAL plans that passed the verifier (no API calls).
Reports maximum heading change at waypoints, steepest climb/descent angle, and estimated flight time, using the mission's speed_h/speed_v.
usage: python tools/kinematic_audit.py <runs.jsonl> [--max-turn 120] [--max-slope 30]"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow `python tools/<x>.py` from the repo root
import math, argparse, collections
from tools._common import runs, mission_map, FlightPlan

ap = argparse.ArgumentParser(); ap.add_argument("runs"); ap.add_argument("--max-turn", type=float, default=120.0); ap.add_argument("--max-slope", type=float, default=30.0)
a = ap.parse_args()
rows = runs(a.runs); mm = mission_map(rows)
agg = collections.defaultdict(list)
for r in rows:
    if not (r.get("final_plan") and r["metrics"].get("final_success")):
        continue
    m = mm[r["scenario_id"]]
    p = FlightPlan.from_dict(r["final_plan"])
    pts = [(m.start[0], m.start[1], m.start_altitude)] + [(w.x, w.y, w.altitude) for w in p.ordered()] + [(p.destination.x, p.destination.y, p.destination.altitude)]
    turn = slope = 0.0; L = 0.0; asc = 0.0
    for i in range(1, len(pts)):
        dx, dy, dz = pts[i][0] - pts[i-1][0], pts[i][1] - pts[i-1][1], pts[i][2] - pts[i-1][2]
        h = math.hypot(dx, dy); L += h; asc += max(0, dz)
        if h > 0:
            slope = max(slope, math.degrees(math.atan2(abs(dz), h)))
        if i >= 2:
            px, py = pts[i-1][0] - pts[i-2][0], pts[i-1][1] - pts[i-2][1]
            if math.hypot(px, py) > 0 and h > 0:
                c = (px * dx + py * dy) / (math.hypot(px, py) * h); turn = max(turn, math.degrees(math.acos(max(-1, min(1, c)))))
    t = L / m.speed_h + asc / m.speed_v
    agg[r["condition"]].append((turn, slope, t))
print(f"condition,n,max_turn_deg(median),turn>{a.max_turn:g}%,max_slope_deg(median),slope>{a.max_slope:g}%,flight_time_min(median)")
for c, v in agg.items():
    n = len(v); med = lambda i: sorted(x[i] for x in v)[n // 2]
    print(f"{c},{n},{med(0):.0f},{100*sum(x[0] > a.max_turn for x in v)/n:.1f},{med(1):.1f},{100*sum(x[1] > a.max_slope for x in v)/n:.1f},{med(2)/60:.1f}")

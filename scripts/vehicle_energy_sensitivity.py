#!/usr/bin/env python
"""Energy of the released airport plans under different VEHICLE parameters (v12, no model calls) - the review asked for tests with different speed, payload,
climb/descent and power parameters, not only the proxy coefficients.

Physical model of src/models/energy.py (rotor/airframe power, climb and descent rates, hover at waypoints), long-range fixed-wing profile. One factor is varied at a
time around the profile: cruise speed x0.8/x1.2, payload 0 / half / full payload_max, climb and descent rate x0.5/x2, drag (lift-to-drag ratio x0.8/x1.2 for the fixed wing), efficiency x0.9/x1.1,
avionics power x2, plus calm air vs a 6 m/s headwind on the whole route. Reported per variant: median and max energy relative to the nominal profile, and the share of
plans whose energy exceeds the usable battery (i.e. plans the proxy budget accepted that this vehicle could not fly).

  python scripts/vehicle_energy_sensitivity.py --out outputs/review_v12/vehicle_energy
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from src.benchmark.generator import load_scenarios
from src.models.energy import plan_energy
from src.models.schemas import Mission, FlightPlan
from src.models.vehicle import get_vehicle
from src.utils.io import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="outputs/raw/real_20261025-194958_3b2644_seed0-only__plus-direct")
ap.add_argument("--scenario-dir", default="benchmark/real")
ap.add_argument("--vehicle", default="long_range_fixed_wing")
ap.add_argument("--out", default="outputs/review_v12/vehicle_energy")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
base = get_vehicle(a.vehicle)
P = {k: getattr(base, k) for k in ("v_cruise", "v_max", "v_climb_max", "v_descend_max", "drag_area_m2", "efficiency", "avionics_w", "payload_max_kg", "lift_to_drag")}
DRAG = (("lift/drag x0.8", {"lift_to_drag": P["lift_to_drag"] * 0.8}), ("lift/drag x1.2", {"lift_to_drag": P["lift_to_drag"] * 1.2})) if base.kind == "fixed_wing" else \
       (("drag x0.7", {"drag_area_m2": P["drag_area_m2"] * 0.7}), ("drag x1.3", {"drag_area_m2": P["drag_area_m2"] * 1.3}))
variants = {"nominal": {}, "speed x0.8": {"v_cruise": P["v_cruise"] * 0.8}, "speed x1.2": {"v_cruise": min(P["v_cruise"] * 1.2, P["v_max"])},
            "payload half": {"_payload": P["payload_max_kg"] / 2}, "payload full": {"_payload": P["payload_max_kg"]},
            "climb/descent x0.5": {"v_climb_max": P["v_climb_max"] * 0.5, "v_descend_max": P["v_descend_max"] * 0.5},
            "climb/descent x2": {"v_climb_max": P["v_climb_max"] * 2, "v_descend_max": P["v_descend_max"] * 2},
            **dict(DRAG),
            "efficiency x0.9": {"efficiency": P["efficiency"] * 0.9}, "efficiency x1.1": {"efficiency": min(1.0, P["efficiency"] * 1.1)},
            "avionics x2": {"avionics_w": P["avionics_w"] * 2}, "headwind 6 m/s": {"_wind": True}}
missions = {s["scenario_id"]: Mission.from_dict(s["mission"]) for s in load_scenarios(a.scenario_dir)}
plans = []
for r in read_jsonl(os.path.join(a.run, "runs.jsonl")):
    fv = r.get("final_verification") or {}
    if r["condition"] in ("C_full", "C_prescriptive", "G_direct_patch", "E_classical") and r["metrics"].get("released") and fv.get("valid") and fv.get("success") and r.get("final_plan"):
        plans.append((r["scenario_id"], r["condition"], FlightPlan.from_dict(r["final_plan"])))
rows = []
for sid, cond, plan in plans:
    m0 = missions[sid]
    for name, ch in variants.items():
        veh = {"base": a.vehicle, **{k: v for k, v in ch.items() if not k.startswith("_")}}
        d = m0.to_dict()
        d.update(vehicle=veh, payload_kg=ch.get("_payload", 0.0), unit_m=1.0, energy_model="physical", max_energy=1e9)
        if ch.get("_wind"):        # headwind against the start->destination direction
            dx, dy = m0.destination[0] - m0.start[0], m0.destination[1] - m0.start[1]
            n = (dx * dx + dy * dy) ** 0.5 or 1.0
            d["wind"] = [-6.0 * dx / n, -6.0 * dy / n]
        m = Mission.from_dict(d)
        e = plan_energy(m, plan)
        rows.append({"scenario_id": sid, "condition": cond, "variant": name, "energy_wh": e["energy_wh"], "usable_wh": e["usable_wh"], "time_s": e["time_s"]})
df = pd.DataFrame(rows)
nom = df[df.variant == "nominal"].set_index(["scenario_id", "condition"]).energy_wh
df["rel_to_nominal"] = df.apply(lambda x: x.energy_wh / nom[(x.scenario_id, x.condition)], axis=1)
df["exceeds_battery"] = df.energy_wh > df.usable_wh
df.to_csv(os.path.join(a.out, "vehicle_energy.csv"), index=False)
summ = df.groupby("variant").agg(plans=("energy_wh", "size"), median_rel=("rel_to_nominal", "median"), max_rel=("rel_to_nominal", "max"),
                                 exceeds_battery=("exceeds_battery", "mean")).reset_index()
summ.to_csv(os.path.join(a.out, "vehicle_energy_summary.csv"), index=False)
print(summ.round(3).to_string(index=False))
json.dump({"vehicle": a.vehicle, "n_plans": len(plans), "summary": summ.to_dict(orient="records")}, open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8"), indent=1)

"""Event-detection + online re-planning experiment (what the paper's dynamic study could not do: events are SENSED, not injected).

For eligible scenarios a verified reference plan is flown in the simulator; at t_appear a zone appears / is enlarged on the remaining route; the UAV detects it
(onboard sensor or broadcast), re-plans with the chosen replanner and continues. Reports detection delay, re-plan verification, intrusions and end-to-end mission success.
Replanners: none (stale plan kept), classical (visibility-graph witness), llm (CA-CLP loop; needs provider)."""
from __future__ import annotations
import csv
import json
import os
import random
from dataclasses import replace
from src.benchmark.generator import load_scenarios
from src.benchmark.feasibility import assess
from src.dynamic.changes import comfortable_plan, CHANGES
from src.dynamic.detection import events_from_change
from src.dynamic.online import run_online_episode, classical_replanner, loop_replanner
from src.models.schemas import Mission
from src.simulator.batch import prepare_mission
from src.simulator.sim import SimConfig


def run_online_experiment(scenario_dir: str, out_dir: str, replanners: dict, n_scenarios: int = 10, vehicle="generic_quad_large", unit_m: float = 20.0,
                          changes=("add_zone", "enlarge_zone"), progress=0.4, delivery_modes=("onboard", "broadcast"), seeds=(0,), selection_seed: int = 2024,
                          appear_frac: float = 0.1, sensor_range_m: float = 120.0, onboard_delay_s: float = 1.0, broadcast_latency_s: float = 5.0, replan_latency_s: float = 5.0,
                          loc_delay_s: float = 0.0, compensate_loc_delay: bool = True) -> str:
    os.makedirs(out_dir, exist_ok=True)
    scen = [s for s in load_scenarios(scenario_dir) if s["metadata"]["feasibility"] == "FEASIBLE"]
    random.Random(f"online|{selection_seed}").shuffle(scen)
    rows, used, baselines = [], 0, {}
    # NOTE: the zone GEOMETRY is generated for the route after `progress` of the flight, but the event HAPPENS (t_appear) at `appear_frac` of the planned time, so the UAV has
    # time to detect and react. Set appear_frac close to `progress` to study late discovery (the zone then appears right in front of the vehicle: collisions become unavoidable).
    for s in scen:
        if used >= n_scenarios:
            break
        m0 = Mission.from_dict(s["mission"])
        m = prepare_mission(m0, vehicle, unit_m, None)
        m.active = [a for a in m.active if a != "energy"]            # proxy energy budget has no meaning in metres/Wh: distance and zones are the study variables
        m.max_distance = m.max_distance * 1.6                          # leave room for a detour at the larger physical scale
        plan = comfortable_plan(m)
        if plan is None:
            continue
        used += 1
        for ch in changes:
            for dm in delivery_modes:
                for sd in seeds:
                    rng = random.Random(f"{s['scenario_id']}|{ch}|{progress}|online")
                    t_plan = _planned_time(m, plan)
                    ev = events_from_change(m, plan, progress, ch, rng, t_appear=appear_frac * t_plan, delivery=dm)
                    if ev is None:
                        continue
                    key_b = (s["scenario_id"], sd)
                    if key_b not in baselines:    # event-free baseline: how often is the VERIFIED reference plan itself flown without incident? (also the reference for 'extra' distance/energy)
                        b = run_online_episode(m, plan, [], lambda a, b_, c: None, sim_cfg=SimConfig(dt=0.5, record_every=4, seed=sd, gust_sigma=0.5, loc_delay_s=loc_delay_s, compensate_loc_delay=compensate_loc_delay), seed=sd)
                        baselines[key_b] = _episode_metrics(b, m, plan, None)
                        rows.append({"scenario_id": s["scenario_id"], "change": "none", "delivery": "-", "replanner": "no_event_baseline", "seed": sd, "n_replans": 0, "replan_verified": False,
                                     **baselines[key_b]})
                    for rname, rp in replanners.items():
                        res = run_online_episode(m, plan, [ev], rp if rp is not None else (lambda a, b, c: None), sim_cfg=SimConfig(dt=0.5, record_every=4, seed=sd, gust_sigma=0.5, loc_delay_s=loc_delay_s, compensate_loc_delay=compensate_loc_delay), seed=sd,
                                                 sensor_range_m=sensor_range_m, onboard_delay_s=onboard_delay_s, broadcast_latency_s=broadcast_latency_s, replan_latency_s=replan_latency_s)
                        em = _episode_metrics(res, m, plan, ev.t_appear)
                        bl = baselines[key_b]
                        rows.append({"scenario_id": s["scenario_id"], "change": ch, "delivery": dm, "replanner": rname, "seed": sd, "n_replans": res["n_replans"],
                                     "replan_verified": any(x.get("replan_verified") for x in res["replans"]),
                                     "extra_distance_m": em["distance_flown_m"] - bl["distance_flown_m"], "extra_duration_s": em["duration_s"] - bl["duration_s"],
                                     "extra_energy_wh": em["energy_used_wh"] - bl["energy_used_wh"], **em})
    keys = sorted({k for r in rows for k in r})
    with open(os.path.join(out_dir, "online_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    # --- recovery summary: an episode is AFFECTED when keeping the stale plan ('none') fails on the same case; recovery = a re-planner succeeds on affected episodes ---
    from src.evaluation.report import wilson
    none_fail = {(r["scenario_id"], r["change"], r["delivery"], r["seed"]) for r in rows if r["replanner"] == "none" and not r["mission_success"]}
    summ = {}
    for r in rows:
        if r["replanner"] in ("no_event_baseline",):
            continue
        g = summ.setdefault((r["replanner"], r["delivery"]), {"n": 0, "affected": 0, "recovered": 0, "success": 0, "detected": 0, "intr": 0, "dd": [], "al": [], "xd": [], "xe": [], "xt": [], "pen": []})
        g["n"] += 1
        g["success"] += int(r["mission_success"])
        g["detected"] += int(bool(r["detected"]))
        g["intr"] += int(r["intrusion_zones"] > 0 or r["status"] == "collision")
        aff = (r["scenario_id"], r["change"], r["delivery"], r["seed"]) in none_fail
        g["affected"] += int(aff)
        g["recovered"] += int(aff and r["mission_success"])
        for k_, v_ in (("dd", "detection_delay_s"), ("al", "activation_latency_s"), ("xd", "extra_distance_m"), ("xe", "extra_energy_wh"), ("xt", "extra_duration_s")):
            if r.get(v_) is None:
                continue
            if k_.startswith("x") and not r["mission_success"]:
                continue          # 'extra' cost vs the event-free flight is only meaningful when the episode actually completed
            g[k_].append(r[v_])
        g["pen"].append(r.get("max_penetration_m") or 0.0)
    mean = lambda v: (sum(v) / len(v)) if v else None
    out_rows = []
    for (rn, dm), g in sorted(summ.items()):
        lo, hi = wilson(g["success"], g["n"])
        rlo, rhi = wilson(g["recovered"], g["affected"]) if g["affected"] else (None, None)
        out_rows.append({"replanner": rn, "delivery": dm, "n_episodes": g["n"], "mission_success_rate": g["success"] / g["n"], "success_ci_lo": lo, "success_ci_hi": hi, "detect_rate": g["detected"] / g["n"],
                         "intrusion_rate": g["intr"] / g["n"], "max_penetration_m_max": max(g["pen"], default=0.0), "n_affected_episodes": g["affected"], "recovery_rate_on_affected": (g["recovered"] / g["affected"]) if g["affected"] else None,
                         "recovery_ci_lo": rlo, "recovery_ci_hi": rhi, "mean_detection_delay_s": mean(g["dd"]), "mean_activation_latency_s": mean(g["al"]), "mean_extra_distance_m": mean(g["xd"]),
                         "mean_extra_duration_s": mean(g["xt"]), "mean_extra_energy_wh": mean(g["xe"])})
    with open(os.path.join(out_dir, "recovery_summary.csv"), "w", newline="") as f:
        if out_rows:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
    base = [r for r in rows if r["replanner"] == "no_event_baseline"]
    json.dump({f"{x['replanner']}|{x['delivery']}": {"n": x["n_episodes"], "success_rate": x["mission_success_rate"], "detect_rate": x["detect_rate"], "intrusions": round(x["intrusion_rate"] * x["n_episodes"])} for x in out_rows}
              | {"no_event_baseline": {"n": len(base), "success_rate": (sum(b["mission_success"] for b in base) / len(base)) if base else None}},
              open(os.path.join(out_dir, "online_summary.json"), "w"), indent=1)
    return out_dir


def _episode_metrics(res: dict, m: Mission, plan, ev_t_appear: float | None) -> dict:
    """Recovery-related, measurable outcomes of one online episode."""
    from src.simulator.study import intrusion_stats, deviation_stats, planned_polyline_m
    tr = res["trajectory"]
    it = intrusion_stats(res["truth_mission"], tr, res["truth_zones"])
    act = [x.get("activated_t") for x in res["replans"] if x.get("activated_t") is not None]
    det = res["detections"][0] if res["detections"] else None
    return {"distance_flown_m": res["evaluation"]["distance_flown_m"], "duration_s": res["evaluation"]["duration_s"], "energy_used_wh": tr.energy_used_wh,
            "intrusion_zones": it["n_zones_entered"], "max_penetration_m": it["max_penetration_m"], "cross_track_max_m": max(tr.cross_track_m) if tr.cross_track_m else None,
            "detection_delay_s": det["detection_delay_s"] if det else None, "detected": det is not None,
            "activation_latency_s": (act[0] - ev_t_appear) if (act and ev_t_appear is not None) else None, "arrived": res["evaluation"]["arrived"],
            "all_required_visited": all(res["evaluation"]["required_visited"].values()), "mission_success": res["mission_success"], "status": res["status"]}


def _planned_time(m: Mission, plan) -> float:
    from src.models.vehicle import get_vehicle
    v = get_vehicle(m.vehicle)
    pts = [m.start] + [(w.x, w.y) for w in plan.ordered()] + [(plan.destination.x, plan.destination.y)]
    L = sum(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 for a, b in zip(pts, pts[1:])) * m.unit_m
    return L / (v.v_cruise * 0.7) + len(plan.waypoints) * v.hover_time_per_waypoint_s

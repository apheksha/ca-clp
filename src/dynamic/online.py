"""Online mission execution with event detection and AUTOMATIC re-planning.

Loop: the simulator flies the current plan -> sensors/broadcast detect events -> the sensed map is updated -> the remaining mission is rebuilt from the ESTIMATED
state (estimated position, visited waypoints, odometer distance, battery use) -> a pluggable `replanner` returns a repaired plan (verifier-gated) -> the new plan
is activated after the re-planning latency (the vehicle keeps flying the stale plan meanwhile) -> execution continues.
The run ends with a ground-truth evaluation (zones/limits as they really are): `mission_success` requires arrival, no collision, no zone intrusion and budgets kept.

Replanners:
  classical_replanner()        verifier-checked visibility-graph witness (no model)
  loop_replanner(planner,cfg)  the LLM verify-and-repair loop (CA-CLP) started from the stale remaining plan
Any callable (remaining_mission, stale_plan, seed) -> FlightPlan | None can be plugged in."""
from __future__ import annotations
import copy
import math
import time
from dataclasses import replace
from src.dynamic.detection import Event, EventEnvironment, SensedMap
from src.models.schemas import Mission, FlightPlan, Waypoint
from src.simulator.sensors import ZoneEventSensor
from src.simulator.sim import FlightSimulator, SimConfig
from src.simulator.evaluation import evaluate_execution
from src.verifier.verifier import verify
from src.benchmark.feasibility import assess


def predicted_start(sim: FlightSimulator, advance_m: float):  # noqa: E302
    """Where the vehicle will be `advance_m` metres further along its CURRENT (stale) path: the new plan must start there, because the vehicle keeps flying while the
    re-plan is computed. Returns (position (x,y,z) metres, ids of intermediate waypoints passed on the way, distance really advanced)."""
    start = tuple(sim.est.p)
    passed: list[str] = []
    rem = advance_m
    nodes = sim.path[sim.k + 1:]
    for idx, (nid, pos) in enumerate(nodes):
        d = math.dist(start, pos)
        last = idx == len(nodes) - 1
        if rem >= d and not last:
            rem -= d
            start = tuple(pos)
            passed.append(nid)
        else:
            f = min(1.0, rem / d) if d > 1e-9 else 0.0
            if last:
                f = min(f, 0.9)           # never plan to start AT the destination
            start = tuple(start[i] + f * (pos[i] - start[i]) for i in range(3))
            rem = 0.0
            break
    return start, passed


def remaining_mission(m: Mission, sim: FlightSimulator, sensed: SensedMap, odometer_m: float, energy_used_wh: float, advance_m: float = 0.0) -> tuple[Mission, FlightPlan]:
    """Mission + stale plan for the part of the flight still to do, as the UAV currently believes it (estimated position, sensed zones, remaining budgets).
    advance_m: plan from the position the vehicle will have reached after that much further flight (re-planning latency); waypoints passed meanwhile count as visited."""
    u = m.unit_m
    (ex, ey, ez), passed = predicted_start(sim, advance_m) if advance_m > 0 else (tuple(sim.est.p), [])
    odometer_m = odometer_m + advance_m
    visited = set(sim.visited) | set(passed)
    left = [r for r in m.required_waypoints if r.id not in visited]
    ids_left = {r.id for r in left}
    ordering = [p for p in m.ordering if p[0] in ids_left and p[1] in ids_left]
    nodes = [n for n in sim.path[sim.k + 1:] if n[0] not in passed]
    wps = [Waypoint(i, p[0] / u, p[1] / u, p[2], k + 1) for k, (i, p) in enumerate(nodes[:-1])]
    dest = nodes[-1][1]
    stale = FlightPlan(m.mission_id, wps, Waypoint("DEST", dest[0] / u, dest[1] / u, dest[2], len(wps) + 1)).validate()
    max_energy = None
    if m.max_energy is not None and sensed.max_energy is not None:
        max_energy = (sensed.max_energy - energy_used_wh) if m.energy_model == "physical" else sensed.max_energy
    rm = replace(m, start=[ex / u, ey / u], start_altitude=ez, required_waypoints=left, ordering=ordering, no_fly_zones=list(sensed.zones.values()),
                 max_distance=max(sensed.max_distance - odometer_m / u, 1e-3), max_energy=max_energy,
                 active=list(m.active) + (["no_fly_zone"] if "no_fly_zone" not in m.active and sensed.zones else []))
    return rm.validate(), stale


def classical_replanner(clearance_m: float | None = None, edge_clearance_m: float = 0.0):
    """clearance_m: extra clearance (metres) around zones for the witness route. The default (tiny) clearance verifies fine but is NOT execution-robust: a vehicle that cuts
    corners / drifts in wind hits a zone it merely 'touches'. Use >= a few tracking-error widths (e.g. 30 m) when the plan will really be flown."""
    def f(rem: Mission, stale: FlightPlan, seed: int):
        a = assess(rem, clear=(clearance_m / rem.unit_m) if clearance_m else None, edge_clearance=edge_clearance_m / rem.unit_m)   # v12: legs keep edge_clearance_m
        return a.witness_plan if a.status == "FEASIBLE" else None
    f.name = "classical"
    return f


def loop_replanner(planner, cfg):
    from src.repair.loop import run_condition
    from src.planner.planner import PlannerOutput

    def f(rem: Mission, stale: FlightPlan, seed: int):
        o = run_condition(rem, planner, cfg, seed, initial=PlannerOutput(stale, [], None))
        return o.final_plan if (o.released and o.success) else None
    f.name = f"loop:{cfg.name}"
    return f


def run_online_episode(mission: Mission, plan: FlightPlan, events: list[Event], replanner, vehicle=None, sim_cfg: SimConfig | None = None, seed: int = 0,
                       sensor_range_m: float | None = None, onboard_delay_s: float = 1.0, p_detect: float = 0.9, broadcast_latency_s: float = 5.0,
                       replan_latency_s: float = 5.0, use_measured_latency: bool = False, max_replans: int = 4) -> dict:
    cfg = sim_cfg or SimConfig(seed=seed)
    env = EventEnvironment(mission, events)
    sim = FlightSimulator(mission, plan, vehicle, cfg, truth_zones_fn=env.truth_zones)
    veh = sim.veh
    sensed = SensedMap(mission)
    zsens = ZoneEventSensor(sensor_range_m or veh.sensor_range_m, onboard_delay_s, p_detect, None, seed)
    ev_by_zone = {e.zone.id: e for e in events if e.zone is not None}
    pending_limit = [e for e in events if e.kind.startswith("limit")]
    pending_replan: dict | None = None
    log, n_replans, odo, last = [], 0, 0.0, tuple(sim.p)
    triggers: list[Event] = []
    seen: set[str] = set()          # events already detected (each event is handled once)
    while sim.step():
        odo += math.dist(last, sim.p)
        last = tuple(sim.p)
        t = sim.t
        # ---- sensing ----
        existing = [z for z in env.truth_zones(t) if z.id in ev_by_zone and t >= ev_by_zone[z.id].t_appear]
        for zid in zsens.poll(t, sim.est.p, existing, env.appear_times(), mission.unit_m, cfg.dt):
            e = ev_by_zone[zid]
            if e.event_id in seen:
                continue
            seen.add(e.event_id)
            sensed.apply(e, t, "onboard")
            triggers.append(e)
        for e in list(pending_limit) + [x for x in events if x.zone is not None and x.delivery == "broadcast"]:
            if e.event_id not in seen and t >= e.t_appear + broadcast_latency_s:
                seen.add(e.event_id)
                sensed.apply(e, t, "broadcast")
                triggers.append(e)
                if e in pending_limit:
                    pending_limit.remove(e)
        # ---- apply a finished re-plan after its latency ----
        if pending_replan and t >= pending_replan["t_active"]:
            if pending_replan["plan"] is not None:
                sim.replace_plan(pending_replan["plan"], pending_replan["mission"], via=pending_replan["via"])
                log[-1]["activated_t"] = round(t, 2)
            pending_replan = None
        # ---- trigger re-planning ----
        if triggers and pending_replan is None and n_replans < max_replans:
            batch, triggers = triggers, []
            adv = replan_latency_s * max(math.hypot(sim.v[0], sim.v[1]), 0.5 * veh.v_cruise)
            rem, stale = remaining_mission(mission, sim, sensed, odo, sim.energy, advance_m=adv)
            if verify(rem, stale).success:
                log.append({"t": round(t, 2), "events": [e.event_id for e in batch], "action": "none_needed (stale plan still valid under the sensed map)"})
                continue
            t0 = time.perf_counter()
            try:
                new = replanner(rem, stale, seed + n_replans)
            except Exception as ex:   # a failing planner must not crash the flight; the vehicle keeps its stale plan
                new, err = None, repr(ex)
            else:
                err = None
            wall = time.perf_counter() - t0
            ok = bool(new is not None and verify(rem, new).success)       # only a verified plan may be activated
            n_replans += 1
            lat = wall if use_measured_latency else replan_latency_s
            log.append({"t": round(t, 2), "events": [e.event_id for e in batch], "replanner": getattr(replanner, "name", "custom"), "replan_verified": ok, "wall_s": round(wall, 3),
                        "latency_applied_s": round(lat, 2), "error": err})
            pending_replan = {"t_active": t + lat, "plan": new if ok else None, "mission": rem, "via": (rem.start[0] * mission.unit_m, rem.start[1] * mission.unit_m, rem.start_altitude)}
    t_end = sim.t
    truth = env.truth_mission(t_end)
    ev = evaluate_execution(truth, plan, sim.res, truth_zones=env.truth_zones(t_end), margin_m=0.0)
    entered = {z: tt for z, tt in ev["zone_intrusions"].items()}
    return {"mission_success": bool(ev["executed_feasible"]), "status": sim.res.status, "evaluation": ev, "truth_zones": env.truth_zones(t_end), "truth_mission": truth, "replans": log, "detections": sensed.detected,
            "zone_intrusions": entered, "n_replans": n_replans, "events": [{"id": e.event_id, "kind": e.kind, "t_appear": e.t_appear} for e in events],
            "undetected_events": [e.event_id for e in events if not any(d["event"] == e.event_id for d in sensed.detected)], "trajectory": sim.res}

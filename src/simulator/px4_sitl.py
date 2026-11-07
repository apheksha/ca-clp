"""PX4 SITL bridge (MAVLink): upload a verified plan to a RUNNING PX4 software-in-the-loop instance (Gazebo or jMAVSim), fly it in AUTO.MISSION, log the real flight
(position, battery, mission progress) and convert the log into the same TrajectoryResult that src/simulator/evaluation.py evaluates for the built-in simulator.

STATUS (be explicit): this module is written against the MAVLink common message set and the PX4 mission protocol, but it was NOT run against a live PX4/Gazebo
instance in this repository's environment (no PX4, Gazebo or network available). The pure functions (mission item construction, log conversion) are unit-tested; the
live `run_px4_mission` needs `pip install pymavlink` and a running SITL (docs/PX4_GAZEBO.md). Expect to adapt connection strings/parameters on first use.

Typical use:
    docker run --rm -it -p 14540:14540/udp jonasvautherin/px4-gazebo-headless      # or: make px4_sitl gazebo
    python run.py px4 --scenario benchmark/generated/easy_000.json --plan plan.json --origin 47.397742 8.545594 --unit-m 20
"""
from __future__ import annotations
import csv
import math
import time
from src.models.schemas import Mission, FlightPlan
from src.simulator.export_px4 import to_qgc_plan, local_to_latlon
from src.simulator.sim import TrajectoryResult
from src.geodata.projection import LocalFrame


def mission_items(mission: Mission, plan: FlightPlan, origin_lat: float, origin_lon: float, unit_m: float = 1.0, cruise_speed: float = 12.0) -> list[dict]:
    """MAVLink MISSION_ITEM_INT field dicts (seq, frame, command, params, x=lat*1e7, y=lon*1e7, z=alt) for TAKEOFF, waypoints, LAND."""
    q = to_qgc_plan(mission, plan, origin_lat, origin_lon, unit_m, cruise_speed)
    items = []
    for i, it in enumerate(q["mission"]["items"]):
        p = it["params"]
        items.append({"seq": i, "frame": it["frame"], "command": it["command"], "current": 1 if i == 0 else 0, "autocontinue": 1,
                      "param1": p[0] or 0, "param2": p[1] or 0, "param3": p[2] or 0, "param4": 0 if p[3] is None else p[3],
                      "x": int(round(p[4] * 1e7)), "y": int(round(p[5] * 1e7)), "z": float(p[6])})
    return items


def log_to_trajectory(rows: list[dict], mission: Mission, origin_lat: float, origin_lon: float, battery_wh: float | None = None, status: str = "arrived") -> TrajectoryResult:
    """rows: [{"t", "lat", "lon", "rel_alt", ("battery_remaining_pct")}] from a PX4 flight -> TrajectoryResult in the mission's metre frame."""
    fr = LocalFrame(origin_lat, origin_lon)
    tr = TrajectoryResult(unit_m=mission.unit_m, vehicle="px4_sitl", status=status)
    prev = None
    used = 0.0
    for r in rows:
        x, y = fr.to_xy(r["lat"], r["lon"])
        tr.t.append(float(r["t"]))
        tr.pos.append((x, y, float(r["rel_alt"])))
        tr.est.append((x, y, float(r["rel_alt"])))
        if prev is None:
            tr.vel.append((0.0, 0.0, 0.0))
        else:
            dt = max(r["t"] - prev["t"], 1e-6)
            tr.vel.append(((x - prev["x"]) / dt, (y - prev["y"]) / dt, (r["rel_alt"] - prev["z"]) / dt))
        if battery_wh is not None and r.get("battery_remaining_pct") is not None:
            used = battery_wh * (1 - r["battery_remaining_pct"] / 100.0)
        tr.energy_wh.append(used)
        tr.power_w.append(0.0)
        tr.cross_track_m.append(0.0)
        prev = {"t": r["t"], "x": x, "y": y, "z": r["rel_alt"]}
    return tr


def save_log_csv(rows: list[dict], path: str) -> str:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["t", "lat", "lon", "rel_alt", "battery_remaining_pct", "mission_seq"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in w.fieldnames})
    return path


def run_px4_mission(mission: Mission, plan: FlightPlan, origin_lat: float, origin_lon: float, unit_m: float = 1.0, connection: str = "udp:127.0.0.1:14540",
                    timeout_s: float = 1800.0, dry_run: bool = False, say=print) -> list[dict]:
    """Upload + fly + log. Returns the log rows. dry_run=True only builds and prints the mission items (no connection)."""
    items = mission_items(mission, plan, origin_lat, origin_lon, unit_m)
    if dry_run:
        say(f"[px4] dry run: {len(items)} mission items")
        return []
    try:
        from pymavlink import mavutil
    except ImportError as e:
        raise RuntimeError("pymavlink is required for PX4 SITL (pip install pymavlink)") from e
    m = mavutil.mavlink_connection(connection)
    m.wait_heartbeat(timeout=30)
    say(f"[px4] heartbeat from system {m.target_system}")
    m.mav.mission_clear_all_send(m.target_system, m.target_component)
    m.mav.mission_count_send(m.target_system, m.target_component, len(items), 0)
    sent = 0
    while sent < len(items):
        req = m.recv_match(type=["MISSION_REQUEST_INT", "MISSION_REQUEST"], blocking=True, timeout=10)
        if req is None:
            raise RuntimeError("PX4 did not request mission items (timeout)")
        it = items[req.seq]
        m.mav.mission_item_int_send(m.target_system, m.target_component, it["seq"], it["frame"], it["command"], it["current"], it["autocontinue"],
                                    it["param1"], it["param2"], it["param3"], it["param4"], it["x"], it["y"], it["z"], 0)
        sent = max(sent, req.seq + 1)
    ack = m.recv_match(type="MISSION_ACK", blocking=True, timeout=10)
    say(f"[px4] mission upload ack: {ack}")
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    m.mav.command_long_send(m.target_system, m.target_component, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0, 1, 4, 4, 0, 0, 0, 0)   # custom mode AUTO.MISSION (PX4 main=4, sub=4)
    rows, t0, seq, bat = [], time.time(), 0, None
    while time.time() - t0 < timeout_s:
        msg = m.recv_match(type=["GLOBAL_POSITION_INT", "BATTERY_STATUS", "MISSION_CURRENT", "MISSION_ITEM_REACHED"], blocking=True, timeout=2)
        if msg is None:
            continue
        ty = msg.get_type()
        if ty == "BATTERY_STATUS":
            bat = msg.battery_remaining
        elif ty == "MISSION_CURRENT":
            seq = msg.seq
        elif ty == "MISSION_ITEM_REACHED" and msg.seq >= len(items) - 1:
            break
        elif ty == "GLOBAL_POSITION_INT":
            rows.append({"t": round(time.time() - t0, 2), "lat": msg.lat / 1e7, "lon": msg.lon / 1e7, "rel_alt": msg.relative_alt / 1000.0, "battery_remaining_pct": bat, "mission_seq": seq})
    return rows

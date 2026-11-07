"""Export a verified plan as a QGroundControl / PX4 `.plan` mission file so it can be flown in PX4 SITL (Gazebo) or any MAVLink simulator.
The execution itself is NOT done here (needs PX4/Gazebo); this removes the manual step between 'verified polyline' and 'simulated flight'.

Map coordinates are local ENU (x east, y north) in `unit_m` metres per unit around (origin_lat, origin_lon). Altitudes are relative to home (m).
Local->geodetic uses an equirectangular approximation (error << 1 % over tens of km), the same projection family as src/benchmark/real_airspace.py."""
from __future__ import annotations
import json
import math
from src.models.schemas import Mission, FlightPlan

EARTH_R = 6371008.8


def local_to_latlon(x: float, y: float, origin_lat: float, origin_lon: float, unit_m: float = 1.0) -> tuple[float, float]:
    lat = origin_lat + math.degrees(y * unit_m / EARTH_R)
    lon = origin_lon + math.degrees(x * unit_m / (EARTH_R * math.cos(math.radians(origin_lat))))
    return lat, lon


def to_qgc_plan(mission: Mission, plan: FlightPlan, origin_lat: float, origin_lon: float, unit_m: float = 1.0, cruise_speed: float = 12.0) -> dict:
    items, jid = [], 1
    def wp(cmd, x, y, alt, params=None):
        nonlocal jid
        lat, lon = local_to_latlon(x, y, origin_lat, origin_lon, unit_m)
        items.append({"type": "SimpleItem", "command": cmd, "doJumpId": jid, "frame": 3, "autoContinue": True,
                      "params": params or [0, 0, 0, None, lat, lon, alt]})
        jid += 1
    sx, sy = mission.start_xy
    slat, slon = local_to_latlon(sx, sy, origin_lat, origin_lon, unit_m)
    wp(22, sx, sy, plan.ordered()[0].altitude if plan.waypoints else plan.destination.altitude, [15, 0, 0, None, slat, slon, plan.destination.altitude])   # TAKEOFF
    for w in plan.ordered():
        wp(16, w.x, w.y, w.altitude, [0, 0, 0, None, *local_to_latlon(w.x, w.y, origin_lat, origin_lon, unit_m), w.altitude])
    wp(16, plan.destination.x, plan.destination.y, plan.destination.altitude, [0, 0, 0, None, *local_to_latlon(plan.destination.x, plan.destination.y, origin_lat, origin_lon, unit_m), plan.destination.altitude])
    wp(21, plan.destination.x, plan.destination.y, 0, [0, 0, 0, None, *local_to_latlon(plan.destination.x, plan.destination.y, origin_lat, origin_lon, unit_m), 0])   # LAND
    return {"fileType": "Plan", "version": 1, "groundStation": "uav-llm-closed-loop",
            "mission": {"version": 2, "firmwareType": 12, "vehicleType": 2, "cruiseSpeed": cruise_speed, "hoverSpeed": 5,
                        "plannedHomePosition": [slat, slon, 0], "items": items},
            "geoFence": {"circles": [], "polygons": [], "version": 2}, "rallyPoints": {"points": [], "version": 2}}


def save_qgc_plan(path: str, *args, **kw) -> str:
    with open(path, "w") as f:
        json.dump(to_qgc_plan(*args, **kw), f, indent=1)
    return path

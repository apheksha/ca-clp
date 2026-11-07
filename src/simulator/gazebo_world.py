"""Generate a Gazebo (SDF 1.9) world for a mission: ground plane, one translucent prism per restricted zone / obstacle (circle -> cylinder, polygon -> extruded polyline,
vertical extent respected), start/destination/waypoint markers. Terrain is NOT exported as a heightmap (needs a raster image + licence-compatible DEM); fly over flat ground
or add the DEM by hand. Not run against Gazebo here (not installed); the file is plain SDF and validated structurally in tests."""
from __future__ import annotations
from xml.sax.saxutils import escape
from src.models.schemas import Mission


def _prism(z, unit_m: float, z_default_top: float = 500.0) -> str:
    z0 = 0.0 if z.z_min is None else z.z_min
    z1 = z_default_top if z.z_max is None else z.z_max
    h = max(z1 - z0, 0.5)
    pose_z = z0 + h / 2
    mat = "<material><ambient>1 0.1 0.1 0.35</ambient><diffuse>1 0.1 0.1 0.35</diffuse></material>"
    if z.kind == "circle":
        cx, cy = z.center[0] * unit_m, z.center[1] * unit_m
        geom = f"<cylinder><radius>{z.radius * unit_m:.3f}</radius><length>{h:.3f}</length></cylinder>"
        pose = f"{cx:.3f} {cy:.3f} {pose_z:.3f} 0 0 0"
    else:
        pts = "".join(f"<point>{v[0] * unit_m:.3f} {v[1] * unit_m:.3f}</point>" for v in z.vertices)
        geom = f"<polyline>{pts}<height>{h:.3f}</height></polyline>"
        pose = f"0 0 {z0:.3f} 0 0 0"
    return (f'<model name="{escape(z.id)}"><static>true</static><link name="l"><visual name="v"><geometry>{geom}</geometry>{mat}</visual>'
            f"<collision name=\"c\"><geometry>{geom}</geometry></collision><pose>{pose}</pose></link></model>")


def world_sdf(mission: Mission, name: str = "uav_mission", origin_lat: float = 47.397742, origin_lon: float = 8.545594) -> str:
    u = mission.unit_m
    zones = "".join(_prism(z, u) for z in mission.no_fly_zones)
    marks = []
    for nm, p in [("START", mission.start)] + [(r.id, r.xy) for r in mission.required_waypoints] + [("DEST", mission.destination)]:
        marks.append(f'<model name="mark_{escape(nm)}"><static>true</static><pose>{p[0] * u:.3f} {p[1] * u:.3f} 0.5 0 0 0</pose><link name="l"><visual name="v"><geometry><sphere><radius>2</radius></sphere></geometry>'
                     f"<material><ambient>0.1 0.4 1 1</ambient></material></visual></link></model>")
    return f"""<?xml version="1.0"?>
<sdf version="1.9"><world name="{escape(name)}">
  <spherical_coordinates><surface_model>EARTH_WGS84</surface_model><latitude_deg>{origin_lat}</latitude_deg><longitude_deg>{origin_lon}</longitude_deg><elevation>0</elevation></spherical_coordinates>
  <include><uri>model://ground_plane</uri></include><include><uri>model://sun</uri></include>
  {zones}
  {''.join(marks)}
</world></sdf>
"""


def write_world(mission: Mission, path: str, **kw) -> str:
    open(path, "w").write(world_sdf(mission, **kw))
    return path

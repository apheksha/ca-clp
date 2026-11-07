from src.models.schemas import Mission, RequiredWaypoint, NoFlyZone, FlightPlan, Waypoint


def mk_mission(**kw) -> Mission:
    d = dict(mission_id="M1", description="test", start=[0, 0], destination=[100, 0],
             required_waypoints=[RequiredWaypoint("A", 30, 0), RequiredWaypoint("B", 70, 0)],
             min_altitude=40, max_altitude=80, max_distance=150,
             no_fly_zones=[], ordering=[["A", "B"]])
    d.update(kw)
    return Mission(**d).validate()


def mk_plan(pts, dest=(100, 0, 60), mid="M1") -> FlightPlan:
    wps = [Waypoint(i, x, y, a, k + 1) for k, (i, x, y, a) in enumerate(pts)]
    return FlightPlan(mid, wps, Waypoint("DEST", dest[0], dest[1], dest[2], len(wps) + 1)).validate()


GOOD = [("A", 30, 0, 60), ("B", 70, 0, 60)]

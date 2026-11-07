"""Planned-vs-actual trajectory plots (top view + altitude profile) with zones, for the simulation study."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon as MplPolygon


def plot_planned_vs_actual(mission, plan, traj, path: str, title: str = "", truth_zones=None) -> str:
    u = mission.unit_m
    fig, (ax, az) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for z in (truth_zones if truth_zones is not None else mission.no_fly_zones):
        if z.kind == "circle":
            ax.add_patch(Circle((z.center[0] * u, z.center[1] * u), z.radius * u, color="tab:red", alpha=0.25))
        else:
            ax.add_patch(MplPolygon([(v[0] * u, v[1] * u) for v in z.vertices], closed=True, color="tab:red", alpha=0.25))
    pts = plan.route_points(mission)
    px, py = [p[1][0] * u for p in pts], [p[1][1] * u for p in pts]
    ax.plot(px, py, "k--", lw=1.2, label="planned (verified)")
    ax.plot([p[0] for p in traj.pos], [p[1] for p in traj.pos], color="tab:blue", lw=1.4, label="flown (true)")
    ax.plot([p[0] for p in traj.est], [p[1] for p in traj.est], color="tab:green", lw=0.6, alpha=0.6, label="estimated")
    for n_, p in pts[1:-1]:
        ax.plot(p[0] * u, p[1] * u, "ko", ms=3)
    ax.plot(px[0], py[0], "g^", ms=8)
    ax.plot(px[-1], py[-1], "r*", ms=10)
    if traj.collision:
        c = traj.collision["position_m"]
        ax.plot(c[0], c[1], "rx", ms=12, mew=2, label="collision")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.legend(loc="best", fontsize=7); ax.set_title(title or "top view", fontsize=9)
    # altitude vs. time
    az.plot(traj.t, [p[2] for p in traj.pos], color="tab:blue", label="altitude (true)")
    if mission.is_active("altitude"):
        az.axhspan(mission.min_altitude, mission.max_altitude, color="tab:green", alpha=0.12, label="allowed band")
    az2 = az.twinx()
    az2.plot(traj.t, traj.cross_track_m, color="tab:orange", lw=0.8, label="cross-track")
    az.set_xlabel("time (s)"); az.set_ylabel("altitude (m)"); az2.set_ylabel("cross-track error (m)")
    az.legend(loc="upper left", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

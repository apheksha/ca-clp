"""2D visualization drawn from real mission/plan/verifier objects only."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from src.models.schemas import Mission, FlightPlan, VerificationResult


def _draw(ax, m: Mission, plan: FlightPlan | None, res: VerificationResult | None, title: str = ""):
    for z in m.no_fly_zones:
        patch = (Circle(z.center, z.radius, fc="#e45756", ec="#a02020", alpha=0.35) if z.kind == "circle"
                 else Polygon(z.vertices, closed=True, fc="#e45756", ec="#a02020", alpha=0.35))
        ax.add_patch(patch)
        c = z.centroid()
        ax.text(c[0], c[1], z.id, ha="center", va="center", fontsize=7, color="#601010")
    bad_segs = set()
    bad_wps = set()
    if res:
        for v in res.violations:
            if v.segment:
                bad_segs.add(tuple(v.segment))
            if v.waypoint_id:
                bad_wps.add(v.waypoint_id)
    if plan:
        pts = plan.route_points(m)
        for (na, pa), (nb, pb) in zip(pts, pts[1:]):
            bad = (na, nb) in bad_segs
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="#d62728" if bad else "#1f77b4", lw=2.2 if bad else 1.5,
                    ls="--" if bad else "-", zorder=3)
        req = {r.id for r in m.required_waypoints}
        for w in plan.ordered():
            ax.plot(w.x, w.y, "o", ms=6, mfc="#2ca02c" if w.id in req else "white", mec="#d62728" if w.id in bad_wps else "#1f4e79", mew=2 if w.id in bad_wps else 1, zorder=4)
            ax.annotate(f"{w.id}\n{w.altitude:g}m", (w.x, w.y), textcoords="offset points", xytext=(4, 5), fontsize=6.5)
        d = plan.destination
        ax.plot(d.x, d.y, "*", ms=11, c="#d62728" if "DEST" in bad_wps else "#9467bd", zorder=5)
    for r in m.required_waypoints:
        ax.plot(r.x, r.y, "x", c="#2ca02c", ms=5, zorder=2)
    ax.plot(*m.start, "s", c="black", ms=7, zorder=5)
    ax.annotate("START", m.start, textcoords="offset points", xytext=(4, -10), fontsize=7)
    ax.annotate("DEST", m.destination, textcoords="offset points", xytext=(-4, 8), fontsize=7)
    pts = [m.start_xy, m.dest_xy] + [r.xy for r in m.required_waypoints]
    for z in m.no_fly_zones:
        c, rr = z.centroid(), z.bounding_radius()
        pts += [(c[0] - rr, c[1] - rr), (c[0] + rr, c[1] + rr)]
    if plan:
        pts += [w.xy for w in plan.waypoints]
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    pad = 0.06 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    sub = ""
    if res:
        dist_txt = f"{res.route_distance:.1f}/{m.max_distance:g}" if m.is_active("distance") else f"{res.route_distance:.1f}"
        en_txt = f" E={res.energy:.0f}/{m.max_energy:g}" if (m.is_active("energy") and m.max_energy is not None) else ""
        sub = f"\nvalid={res.valid} preserved={res.mission_preserved} viol={res.score} dist={dist_txt}{en_txt}"
    ax.set_title(title + sub, fontsize=8)


def plot_plan(m: Mission, plan: FlightPlan | None, res: VerificationResult | None = None, title: str = "", path: str | None = None):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    _draw(ax, m, plan, res, title or m.mission_id)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200)
        plt.close(fig)
    return fig


def plot_history(m: Mission, iterations: list[dict], path: str | None = None, suptitle: str = ""):
    """iterations: list of RepairIteration dicts (plan + verification as stored in raw results)."""
    shown = [it for it in iterations if it.get("plan")]
    n = max(1, len(shown))
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    from src.verifier.verifier import verify
    for ax in axes.ravel():
        ax.axis("off")
    for ax, it in zip(axes.ravel(), shown):
        ax.axis("on")
        plan = FlightPlan.from_dict(it["plan"])
        res = verify(m, plan)  # re-verified deterministically for drawing
        _draw(ax, m, plan, res, f"iter {it['iteration']} [{it['outcome']}]")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=170)
        plt.close(fig)
    return fig


def plot_dynamic(rec: dict, path: str, tag: str = ""):
    """Objective-4 figure from a stored dynamic run: original route (flown + abandoned remainder), the change, and the repaired remaining route."""
    from src.models.schemas import Mission
    d = rec["dynamic"]
    m_new = Mission.from_dict(d["mission"])
    m_old = Mission.from_dict(d["original_mission"])
    old = FlightPlan.from_dict(d["original_plan"])
    final = FlightPlan.from_dict(rec["final_plan"]) if rec["final_plan"] else None
    res = verify_plan(m_new, final) if final else None
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    _draw(ax, m_new, final, res, f"{rec['scenario_id']}  [{rec['condition']} -> {rec['final_status']}]")
    old_ids = {z.id: z for z in m_old.no_fly_zones}
    for z in m_new.no_fly_zones:
        if z.id not in old_ids or abs(z.area() - old_ids[z.id].area()) > 1e-6:       # highlight the changed / new zone
            patch = (Circle(z.center, z.radius, fc="none", ec="#e08a00", lw=2.2, ls="--") if z.kind == "circle" else Polygon(z.vertices, closed=True, fc="none", ec="#e08a00", lw=2.2, ls="--"))
            ax.add_patch(patch)
    pts = old.route_points(m_old)
    k = d["state"]["position"]
    flown = [p for n_, p in pts if n_ in d["state"]["visited"]]
    ax.plot([pts[0][1][0]] + [p[0] for p in flown] + [k[0]], [pts[0][1][1]] + [p[1] for p in flown] + [k[1]], color="#555", lw=2.5, label="flown")
    ax.plot([k[0]] + [p[0] for n_, p in pts if n_ not in d["state"]["visited"] and n_ != "START"], [k[1]] + [p[1] for n_, p in pts if n_ not in d["state"]["visited"] and n_ != "START"],
            color="#999", lw=1.2, ls=":", label="abandoned original route")
    ax.plot(*k, "D", c="#e08a00", ms=8, zorder=6)
    ax.legend(fontsize=7, loc="lower left")
    allp = [p_ for _, p_ in pts] + [tuple(k)]
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, min(p_[0] for p_ in allp) - 2), max(x1, max(p_[0] for p_ in allp) + 2))
    ax.set_ylim(min(y0, min(p_[1] for p_ in allp) - 2), max(y1, max(p_[1] for p_ in allp) + 2))
    if tag:
        ax.text(0.5, 0.01, tag, transform=ax.transAxes, ha="center", color="red", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)


def verify_plan(m, plan):
    from src.verifier.verifier import verify
    return verify(m, plan)

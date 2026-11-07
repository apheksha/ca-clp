"""MOCK LLM: a seeded, heuristic stand-in for a language model. IT IS NOT A REAL LLM.

It exists only to exercise the pipeline offline. Its error/repair-success probabilities are ASSUMPTIONS set in
config (mock:), so any experiment result produced with it is a pipeline test, not evidence about real LLMs.
Every record produced with it carries is_mock=True and provider='mock'."""
from __future__ import annotations
import json
import math
import random
import re
from src.models.schemas import Mission, NoFlyZone
from src.planner.llm import LLMClient, LLMResponse
from src.utils import geometry as G

DEFAULTS = dict(order_error=0.15, drop_required=0.04, ignore_nfz=0.6, altitude_error=0.25, coord_error=0.03, malformed=0.04,
                p_fix={"localized": 0.60, "evidence": 0.85, "prescriptive": 0.85}, p_binary_resample=0.10,
                p_collateral={"regenerate": 0.25, "targeted": 0.05}, p_regen_fresh=0.10, p_move_locked=0.10)

PROMPT_ONLY_KEYS = ("ground_elevation_samples", "terrain_note")   # written by prompts.mission_spec for terrain missions; not Mission fields


def _tag(text: str, name: str):
    m = re.search(rf"\[{name}\](.*?)\[/{name}\]", text, re.S)
    return json.loads(m.group(1)) if m else None


def _word(text: str, name: str):
    m = re.search(rf"\[{name}\]\s*(\w+)", text)
    return m.group(1) if m else None


class MockLLM(LLMClient):
    name = "mock"
    is_mock = True
    model = "mock-heuristic-v1"

    def __init__(self, params: dict | None = None):
        self.p = {**DEFAULTS, **(params or {})}

    def config(self):
        return {"provider": "mock", "model": self.model, "is_mock": True, "params": self.p,
                "WARNING": "MOCK heuristic stand-in; not a language model"}

    # ---- helpers working on dict plans ----
    @staticmethod
    def _zone(m: Mission, zid):
        return m.zone(zid)

    def _detour(self, m: Mission, a, b, z: NoFlyZone, rng, precise: bool, ev: dict | None):
        c = z.centroid()
        R = z.bounding_radius()
        if precise and ev and "closest_point_on_segment" in ev:
            cp = ev["closest_point_on_segment"]
            ux, uy = cp[0] - c[0], cp[1] - c[1]
        else:
            dx, dy = b[0] - a[0], b[1] - a[1]
            ux, uy = -dy, dx
            if rng.random() < 0.5:
                ux, uy = -ux, -uy
        n = math.hypot(ux, uy) or 1.0
        off = (R * 1.03 + 2.5) * 1.15 if precise else R * 1.1
        return (c[0] + ux / n * off, c[1] + uy / n * off)

    def _fresh_plan(self, m: Mission, rng: random.Random):
        p = self.p
        order = [r for r in m.required_waypoints]
        if len(order) >= 2 and rng.random() < p["order_error"]:
            i = rng.randrange(len(order) - 1)
            order[i], order[i + 1] = order[i + 1], order[i]
        if len(order) > 2 and rng.random() < p["drop_required"]:
            order.pop(rng.randrange(len(order)))
        pts = []
        for r in order:
            x, y = r.x, r.y
            if rng.random() < p["coord_error"]:
                x += rng.choice([-1, 1]) * max(3.0, 0.03 * max(abs(m.dest_xy[0] - m.start_xy[0]), 1.0))
            pts.append({"id": r.id, "x": x, "y": y})
        route = [("START", m.start_xy)] + [(q["id"], (q["x"], q["y"])) for q in pts] + [("DEST", m.dest_xy)]
        if rng.random() >= p["ignore_nfz"]:
            k = 0
            seq = []
            for i in range(len(route) - 1):
                a, b = route[i][1], route[i + 1][1]
                for z in m.no_fly_zones:
                    if z.intersects_segment(a, b):
                        d = self._detour(m, a, b, z, rng, False, None)
                        k += 1
                        seq.append((i, {"id": f"W{k}", "x": d[0], "y": d[1]}))
                        break
            for i, w in sorted(seq, key=lambda s: -s[0]):
                pts.insert(i, w)  # insert before original index i (after route[i])
        alt = lambda: (rng.choice([m.min_altitude - rng.uniform(5, 20), m.max_altitude + rng.uniform(5, 20)])
                       if rng.random() < p["altitude_error"] else rng.uniform(m.min_altitude, m.max_altitude))
        for q in pts:
            q["altitude"] = alt()
        return {"waypoints": pts, "destination": {"x": m.destination[0], "y": m.destination[1], "altitude": alt()}}

    def _repair(self, m: Mission, plan: dict, fb: dict, level: str, mode: str, rng: random.Random):
        p = self.p
        if level == "binary" or fb is None or "violations" not in fb:
            return self._fresh_plan(m, rng)  # binary signal carries no location: only resampling is possible
        if mode == "regenerate" and rng.random() < p["p_regen_fresh"]:
            return self._fresh_plan(m, rng)
        precise = level in ("evidence", "prescriptive")
        pf = p["p_fix"][level]
        wps = [dict(w) for w in plan["waypoints"]]
        dest = dict(plan["destination"])
        find = lambda i: dest if i == "DEST" else next((w for w in wps if w["id"] == i), None)
        req = {r.id: r for r in m.required_waypoints}
        implicated = set()
        mid = (m.min_altitude + m.max_altitude) / 2
        nxt = [sum(1 for w in wps if w["id"].startswith("W")) + 1]
        for v in fb["violations"] + fb.get("mission_requirement_violations", []):
            implicated.update(x for x in [v.get("waypoint_id")] + (v.get("segment") or []) if x)
            if rng.random() > pf:
                continue
            t = v["type"]
            w = find(v.get("waypoint_id")) if v.get("waypoint_id") else None
            if t in ("ALTITUDE_BELOW_MIN", "ALTITUDE_ABOVE_MAX") and w:
                w["altitude"] = (min(max(w["altitude"], m.min_altitude + 1), m.max_altitude - 1) if precise else mid)
            elif t == "NFZ_WAYPOINT" and w:
                if v["waypoint_id"] in req and rng.random() > p["p_move_locked"]:
                    continue
                z = self._zone(m, v["zone_id"])
                c = z.centroid()
                dx, dy = w["x"] - c[0], w["y"] - c[1]
                n = math.hypot(dx, dy) or 1.0
                off = z.bounding_radius() * (1.03 if precise else 1.01) + (2.5 if precise else 1.0)
                w["x"], w["y"] = c[0] + dx / n * off, c[1] + dy / n * off
            elif t == "NFZ_SEGMENT":
                a_id, b_id = v["segment"]
                z = self._zone(m, v["zone_id"])
                pts = {"START": m.start_xy, "DEST": (dest["x"], dest["y"])}
                pts.update({q["id"]: (q["x"], q["y"]) for q in wps})
                d = self._detour(m, pts[a_id], pts[b_id], z, rng, precise, v.get("evidence"))
                idx = 0 if a_id == "START" else next(i for i, q in enumerate(wps) if q["id"] == a_id) + 1
                while f"W{nxt[0]}" in {q["id"] for q in wps}:
                    nxt[0] += 1
                wps.insert(idx, {"id": f"W{nxt[0]}", "x": d[0], "y": d[1], "altitude": mid})
            elif t == "DISTANCE_EXCEEDED":
                extras = [q for q in wps if q["id"] not in req]
                if extras:
                    pts_ = [m.start_xy] + [(q["x"], q["y"]) for q in wps] + [(dest["x"], dest["y"])]
                    def gain(q):
                        i = wps.index(q) + 1
                        return G.dist(pts_[i - 1], pts_[i]) + G.dist(pts_[i], pts_[i + 1]) - G.dist(pts_[i - 1], pts_[i + 1])
                    victim = max(extras, key=gain) if precise else rng.choice(extras)
                    wps.remove(victim)
            elif t == "ENERGY_EXCEEDED":
                for q in wps + [dest]:
                    q["altitude"] = (m.min_altitude + 1 if precise else max(m.min_altitude + 1, q["altitude"] * 0.8))
                extras = [q for q in wps if q["id"] not in req]
                if extras and not precise:
                    wps.remove(rng.choice(extras))
            elif t == "ORDER_VIOLATION":
                rid = [q for q in wps if q["id"] in req]
                ids = [q["id"] for q in rid]
                for b, a in m.ordering:
                    if b in ids and a in ids and ids.index(b) > ids.index(a):
                        ib, ia = wps.index(find(b)), wps.index(find(a))
                        wps[ib], wps[ia] = wps[ia], wps[ib]
                        ids = [q["id"] for q in wps if q["id"] in req]
            elif t == "MISSING_REQUIRED_WAYPOINT":
                r = req[v["waypoint_id"]]
                rank = [x.id for x in m.required_waypoints]
                pos = sum(1 for q in wps if q["id"] in rank and rank.index(q["id"]) < rank.index(r.id) and q["id"] in req)
                req_idx = [i for i, q in enumerate(wps) if q["id"] in req]
                at = (req_idx[pos - 1] + 1) if 0 < pos <= len(req_idx) else (req_idx[0] if req_idx else len(wps))
                wps.insert(at, {"id": r.id, "x": r.x, "y": r.y, "altitude": mid})
            elif t == "REQUIRED_WAYPOINT_MOVED" and w:
                w["x"], w["y"] = req[v["waypoint_id"]].xy
            elif t == "DESTINATION_CHANGED":
                dest["x"], dest["y"] = m.dest_xy
        if wps and rng.random() < p["p_collateral"][mode]:
            others = [q for q in wps if q["id"] not in implicated] or wps
            q = rng.choice(others)
            q["altitude"] = rng.choice([m.min_altitude - rng.uniform(3, 12), m.max_altitude + rng.uniform(3, 12), mid])
        return {"waypoints": wps, "destination": dest}

    def generate(self, system, user, temperature, max_tokens, seed):
        spec = {k: v for k, v in _tag(user, "MISSION_SPEC").items() if k not in PROMPT_ONLY_KEYS}   # v12: terrain missions add prompt-only keys
        m = Mission.from_dict({**spec, "description": "mock"})
        role = _word(user, "ROLE")
        rng = random.Random(f"{seed}|{m.mission_id}|{role}|{_word(user, 'FEEDBACK_LEVEL')}|{_word(user, 'REPAIR_MODE')}")
        if role == "repair":
            plan = self._repair(m, _tag(user, "CURRENT_PLAN"), _tag(user, "FEEDBACK"), _word(user, "FEEDBACK_LEVEL"), _word(user, "REPAIR_MODE"), rng)
        else:
            plan = self._fresh_plan(m, rng)
        out = {"waypoints": [{"id": w["id"], "x": round(w["x"], 2), "y": round(w["y"], 2), "altitude": round(w["altitude"], 1)} for w in plan["waypoints"]],
               "destination": {k: round(v, 2) for k, v in plan["destination"].items()}}
        text = json.dumps(out)
        if rng.random() < self.p["malformed"]:
            text = rng.choice(["Here is the plan: " + text, text[:-1], text.replace('"altitude"', '"alt"', 1)])
        return LLMResponse(text, self.model, len(user) // 4, len(text) // 4, 0.0, True, tokens_estimated=True)

"""FollowerLLM: a deterministic DRY-RUN stand-in (provider 'follower', is_mock=True). It is NOT a language model and its results are NOT evidence.
Purpose: exercise the whole closed loop end to end with no API. Behaviour (fixed, no randomness):
  initial plan : straight line through the required waypoints at mid altitude (ignores zones and limits)
  repair       : applies EXACTLY the 'suggested_detour' entries of the feedback (prescriptive level); with any other level it returns the plan unchanged.
So with prescriptive feedback the loop must succeed wherever a detour fixes the violations, and must fail elsewhere; the test-suite asserts that."""
from __future__ import annotations
import json
import re
from src.models.schemas import Mission
from src.planner.llm import LLMClient, LLMResponse


def _tag(text, name):
    m = re.search(rf"\[{name}\](.*?)\[/{name}\]", text, re.S)
    return json.loads(m.group(1)) if m else None


class FollowerLLM(LLMClient):
    name = "follower"
    is_mock = True
    model = "follower-dryrun-v1"

    def config(self):
        return {"provider": self.name, "model": self.model, "is_mock": True, "WARNING": "DRY-RUN stand-in, not a language model"}

    def generate(self, system, user, temperature, max_tokens, seed):
        spec = _tag(user, "MISSION_SPEC")
        from src.planner.mock_llm import PROMPT_ONLY_KEYS
        m = Mission.from_dict({**{k: v for k, v in spec.items() if k not in PROMPT_ONLY_KEYS}, "description": "dry-run"})
        if "[ROLE]repair" in user:
            plan = _tag(user, "CURRENT_PLAN")
            fb = _tag(user, "FEEDBACK") or {}
            wps = list(plan["waypoints"])
            n = 0
            ops = []
            for v in fb.get("violations", []):
                d = v.get("suggested_detour")
                if not d:
                    continue
                if d["action"] == "insert":
                    pos = -1 if d["between"][0] == "START" else next((i for i, w in enumerate(wps) if w["id"] == d["between"][0]), None)
                    if pos is not None:
                        ops.append((pos, "insert", d["new_waypoints_in_order"]))
                else:
                    pos = next((i for i, w in enumerate(wps) if w["id"] == d["waypoint"]), None)
                    if pos is not None:
                        ops.append((pos, "replace", d["with_waypoints_in_order"]))
            used = {w["id"] for w in wps}
            for pos, kind, new in sorted(ops, key=lambda o: -o[0]):
                nw = []
                for q in new:
                    while f"D{n}" in used:
                        n += 1
                    used.add(f"D{n}")
                    nw.append({"id": f"D{n}", "x": q["x"], "y": q["y"], "altitude": q["altitude"]})
                if kind == "insert":
                    wps[pos + 1:pos + 1] = nw
                else:
                    wps[pos:pos + 1] = nw
            out = {"waypoints": wps, "destination": plan["destination"]}
        else:
            alt = round((m.min_altitude + m.max_altitude) / 2, 1)
            out = {"waypoints": [{"id": r.id, "x": r.x, "y": r.y, "altitude": alt} for r in m.required_waypoints],
                   "destination": {"x": m.destination[0], "y": m.destination[1], "altitude": alt}}
        text = json.dumps(out)
        return LLMResponse(text, self.model, len(user) // 4, len(text) // 4, 0.0, True, tokens_estimated=True)

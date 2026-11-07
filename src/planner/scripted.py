"""ScriptedLLM: returns pre-written replies in order. Used for tests and the hand-written demo cases.
It is NOT a language model and its output must never be reported as LLM output."""
import json
from src.planner.llm import LLMClient, LLMResponse


class ScriptedLLM(LLMClient):
    name = "scripted"
    is_mock = True
    model = "scripted-fixture"

    def __init__(self, replies: list):
        self.replies = list(replies)
        self.i = 0
        self.prompts: list[str] = []

    def generate(self, system, user, temperature, max_tokens, seed):
        self.prompts.append(user)
        r = self.replies[min(self.i, len(self.replies) - 1)]
        self.i += 1
        text = r if isinstance(r, str) else json.dumps(r)
        return LLMResponse(text, self.model, len(user) // 4, len(text) // 4, 0.0, True, tokens_estimated=True)


def plan_json(pts, dest):
    return {"waypoints": [{"id": i, "x": x, "y": y, "altitude": a} for i, x, y, a in pts],
            "destination": {"x": dest[0], "y": dest[1], "altitude": dest[2]}}

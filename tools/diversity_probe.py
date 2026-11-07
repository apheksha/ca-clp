"""Temperature-diversity probe (a few real API calls): do repeated samples at temperature T return different plans?
usage: python tools/diversity_probe.py --provider gemini --model <id> [--temperature 0.7 1.0] [--nonce] [--samples 5] [--missions 4]
Costs samples x missions x len(temperatures) calls (default 5 x 4 x 2 = 40)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # allow `python tools/<x>.py` from the repo root
import argparse, json, glob, yaml
from tools._common import mission_of, runs
from src.planner.llm import make_client
from src.planner.planner import LLMPlanner
from src.benchmark.generator import load_scenarios
from src.models.schemas import Mission

ap = argparse.ArgumentParser(); ap.add_argument("--provider", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--temperature", type=float, nargs="*", default=[0.7, 1.0]); ap.add_argument("--nonce", action="store_true")
ap.add_argument("--samples", type=int, default=5); ap.add_argument("--missions", type=int, default=4); ap.add_argument("--scenario-dir", default="benchmark/generated")
a = ap.parse_args()
cfg = yaml.safe_load(open("config/default.yaml")); cfg.update(provider=a.provider, model=a.model)
pl = LLMPlanner(make_client(cfg), 0.0, cfg.get("max_tokens", 2000))
for s in load_scenarios(a.scenario_dir)[:a.missions]:
    m = Mission.from_dict(s["mission"])
    for T in a.temperature:
        sig = []
        for k in range(a.samples):
            o = pl.initial_plan(m, 0, k, temperature=T, nonce=a.nonce)
            sig.append(json.dumps(o.plan.to_dict(), sort_keys=True) if o.plan else "PARSE_FAIL")
        print(f"{s['scenario_id']:<34} T={T:<4} nonce={a.nonce} distinct plans {len(set(sig))}/{a.samples}")

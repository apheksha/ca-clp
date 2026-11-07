#!/usr/bin/env python
"""LLM as the language interface: free-text mission -> structured spec -> classical planner -> verify against the TRUE mission (src/nl/spec_extraction.py).
ONE model call per mission (53 airport missions + 118 synthetic = 171 calls for everything), so it fits a free-tier daily quota.

  python scripts/spec_extraction_eval.py --provider gemini --model gemini-3.5-flash-lite --scenario-dir benchmark/real benchmark/generated
  python scripts/spec_extraction_eval.py --provider groq --model <id> --scenario-dir benchmark/real --max-calls 60
  python scripts/spec_extraction_eval.py ... --resume outputs/nl_spec/<folder>        # continue after a quota stop
  python scripts/spec_extraction_eval.py --provider truth --scenario-dir benchmark/real  # pipeline sanity check: echoes the true spec (NOT a model)
Writes records.jsonl, manifest.json, summary.json, summary_by_source.csv, field_accuracy.csv in the output folder."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
from src.benchmark.generator import load_scenarios
from src.models.schemas import Mission
from src.nl.spec_extraction import SPEC_SYSTEM, spec_prompt, evaluate_reply, truth_spec, wilson
from src.planner.llm import make_client, LLMError, LLMResponse, stop_info
from src.utils.io import append_jsonl, read_jsonl, env_manifest


from src.utils.dotenv import load_dotenv   # v12: shared, BOM/quote tolerant


class TruthEcho:
    """Sanity-check client: returns the TRUE specification as JSON. Not a language model; shows the pipeline's upper bound (should be 100 %)."""
    name, model, is_mock = "truth", "truth-echo", True

    def __init__(self, missions):
        self.by_prompt = missions

    def generate(self, system, user, temperature, max_tokens, seed):
        m = self.by_prompt[user]
        t = truth_spec(m)
        t["zones"] = [{"id": z.id, "kind": z.kind, "center": z.center, "radius": z.radius, "vertices": z.vertices, "z_min": z.z_min, "z_max": z.z_max} for z in t["zones"]]
        return LLMResponse(json.dumps(t), self.model, None, None, 0.0, True)


def summarize(recs):
    import pandas as pd
    df = pd.DataFrame([{**{k: r[k] for k in ("scenario_id", "source", "parsed", "all_fields_correct", "end_to_end_success")}, **{f"f_{k}": v for k, v in (r.get("fields") or {}).items()}} for r in recs])
    out = {"n": len(df)}
    for col in ("parsed", "all_fields_correct", "end_to_end_success"):
        k = int(df[col].sum()); lo, hi = wilson(k, len(df))
        out[col] = {"k": k, "rate": k / len(df) if len(df) else None, "wilson_lo": lo, "wilson_hi": hi}
    by = []
    for src, g in df.groupby("source"):
        row = {"source": src, "n": len(g)}
        for col in ("parsed", "all_fields_correct", "end_to_end_success"):
            k = int(g[col].sum()); lo, hi = wilson(k, len(g))
            row.update({f"{col}_rate": k / len(g), f"{col}_lo": lo, f"{col}_hi": hi})
        by.append(row)
    fields = [c for c in df.columns if c.startswith("f_")]
    fa = df[df.parsed][fields].mean().rename(lambda c: c[2:]).reset_index().rename(columns={"index": "field", 0: "accuracy"}) if fields else pd.DataFrame()
    return out, pd.DataFrame(by), fa


def main(argv=None):
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--scenario-dir", nargs="*", default=["benchmark/real"])
    ap.add_argument("--max-calls", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--resume")
    ap.add_argument("--out-root", default="outputs/nl_spec")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--config", default="config/default.yaml", help="provider settings (e.g. config/default_groq.yaml for Groq: reasoning effort)")
    a = ap.parse_args(argv)
    scen = [(d, s) for d in a.scenario_dir for s in load_scenarios(d) if s["metadata"].get("feasibility") == "FEASIBLE"]
    if not scen:
        raise SystemExit(f"no feasible scenarios in {a.scenario_dir}")
    if a.limit:
        scen = scen[:a.limit]
    missions = {s["scenario_id"]: Mission.from_dict(s["mission"]) for _, s in scen}
    if a.provider == "truth":
        client = TruthEcho({spec_prompt(m): m for m in missions.values()})
    else:
        cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
        cfg.update(provider=a.provider, model=a.model)
        client = make_client(cfg)
    out = a.resume or os.path.join(a.out_root, f"{client.name}_{client.model.replace('/', '_')}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "records.jsonl")
    done = {r["scenario_id"] for r in read_jsonl(path)} if os.path.exists(path) else set()
    mpath = os.path.join(out, "manifest.json")
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if os.path.exists(mpath):        # resume: keep the original manifest (start time, environment); refuse to mix models in one folder
        with open(mpath, encoding="utf-8") as fh:
            man = json.load(fh)
        if (man.get("provider"), man.get("model")) != (client.name, client.model):
            raise SystemExit(f"{out} belongs to {man.get('provider')}/{man.get('model')}, not {client.name}/{client.model}")
        man.setdefault("sessions", []).append({"started": now})
    else:
        man = {"provider": client.name, "model": client.model, "is_mock": client.is_mock, "scenario_dirs": a.scenario_dir, "n_scenarios": len(scen), "system_prompt": SPEC_SYSTEM,
               "temperature": 0.0, "max_tokens": a.max_tokens, "env": env_manifest(), "started": now, "sessions": [{"started": now}],
               "note": "one call per mission; end_to_end_success = oracle witness planned on the EXTRACTED mission is valid and mission-preserving for the TRUE mission"}
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    calls, stop, stop_struct = 0, None, None
    try:
        for d, s in scen:
            sid = s["scenario_id"]
            if sid in done:
                continue
            if a.max_calls is not None and calls >= a.max_calls:
                stop = f"call budget reached ({calls})"
                break
            m = missions[sid]
            prompt = spec_prompt(m)
            t0 = time.perf_counter()
            r = client.generate(SPEC_SYSTEM, prompt, 0.0, a.max_tokens, 0)
            calls += 1
            ev = evaluate_reply(r.text, m)
            rec = {"scenario_id": sid, "source": s["metadata"].get("source", "synthetic"), "difficulty": s["difficulty"], "nl_style": s["metadata"].get("nl_style"),
                   "provider": client.name, "model": client.model, "prompt": prompt, "raw_response": r.text, "prompt_tokens": r.prompt_tokens, "completion_tokens": r.completion_tokens,
                   "latency_s": round(time.perf_counter() - t0, 3), **ev}
            append_jsonl(path, rec)
            print(f"[{len(done) + calls}/{len(scen)}] {sid:<32} parsed={ev['parsed']} fields_ok={ev.get('all_fields_correct')} e2e={ev.get('end_to_end_success')}", flush=True)
    except LLMError as e:
        stop = f"provider error: {e}"
        stop_struct = stop_info(e)
    except KeyboardInterrupt:
        stop = "interrupted"
    man["stop"] = stop_struct or stop_info(None, stop)
    man["sessions"][-1].update(finished=time.strftime("%Y-%m-%dT%H:%M:%S%z"), calls=calls, stopped_reason=stop)
    with open(mpath, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    recs = read_jsonl(path) if os.path.exists(path) else []
    if recs:
        s, by, fa = summarize(recs)
        s.update(stopped_reason=stop, stop=man["stop"], complete=len({r["scenario_id"] for r in recs}) == len(scen))
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(s, fh, indent=1)
        by.to_csv(os.path.join(out, "summary_by_source.csv"), index=False)
        fa.to_csv(os.path.join(out, "field_accuracy.csv"), index=False)
        print(json.dumps(s, indent=1))
    if stop:
        print(f"[stopped] {stop}. Continue with: python scripts/spec_extraction_eval.py --provider {a.provider} --model {a.model} --scenario-dir {' '.join(a.scenario_dir)} --resume {out}")
    print("output:", out)
    return out


if __name__ == "__main__":
    main()

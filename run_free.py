#!/usr/bin/env python
"""ONE command for every experiment that is still missing from the paper. Free tier only (one Gemini API key).

    python run_free.py              run (or continue) everything; waits for free daily quotas to reset and carries on
    python run_free.py --status     what is done, what is left
    python run_free.py --dry-run    the whole pipeline offline with stand-ins (no key, no calls); writes only to outputs_dryrun/

Safe to stop (Ctrl-C, closed laptop, power cut) and to start again: every finished model call is on disk and is never repeated.
At the end it rebuilds the paper's numbers and writes outputs/free_run/RESULTS_TO_SEND.zip.

Stages (config/free_run.yaml):
  preflight      Python/packages, .env key, unit tests, model ids, one test call per model
  tempcheck      12 calls: does the provider honour temperature? (review: the resampling baseline)
  spec           ~171 calls: the model as language interface (text -> specification -> classical planner -> verifier)   [review point D]
  bengaluru      <=210 calls: the 30 missions built from OpenStreetMap buildings + SRTM terrain (3D obstacles, terrain)
  second_models  <=530 calls per model: other models on the 53 airport-anchored missions                               [review: one model only]
  dynamic        <=492 calls: mid-flight re-planning with the model on a random sample, corrected code                   [review: dynamic runs]
  groq           optional non-Google model (off by default; slow on the free tier)
  paper          make_paper_numbers.py + check_paper_numbers.py (+ latexmk if installed) + results zip
"""
from __future__ import annotations
import os
import subprocess
import sys

if __name__ == "__main__" and not sys.flags.utf8_mode and os.environ.get("RUN_FREE_CHILD") != "1":
    # Windows opens text files in the ANSI code page by default; run everything in UTF-8 mode (OSM names, model replies).
    sys.exit(subprocess.call([sys.executable, "-X", "utf8"] + sys.argv, env=dict(os.environ, PYTHONUTF8="1", RUN_FREE_CHILD="1")))

import argparse
import datetime as _dt
import glob
import importlib.util
import json
import re
import shutil
import time
import traceback
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import yaml  # noqa: E402
from src.utils.dotenv import load_dotenv  # noqa: E402

D = "config/default.yaml"
EXP = "config/experiments.yaml"
DYN = "config/experiments_dynamic_rerun.yaml"
DYN_SYN = "config/experiments_dynamic_rerun_synthetic.yaml"
BENGALURU = ["A_llm_only", "C_full", "C_prescriptive", "G_direct_patch", "E_classical"]
SECOND = ["A_llm_only", "C_full", "C_prescriptive", "G_direct_patch"]      # v12.1: lean - the comparison the paper needs (RQ1/RQ2), no binary ablation
SPEC_DIRS = ["benchmark/real", "benchmark/generated"]
# tests that guard the experimental logic itself (verifier, oracle, repair loop, mission sets, mid-flight changes, client quota handling): a failure here stops the run
CORE_TESTS = {"tests.test_verifier", "tests.test_loop", "tests.test_benchmark", "tests.test_dynamic", "tests.test_dynamic_stale", "tests.test_robustness",
              "tests.test_v12_free_run", "tests.test_v12_audit_fixes"}
WORST = {"tempcheck": 12, "spec": 171, "bengaluru": 30 * 7, "second": 53 * 7, "dynamic": 82 * 6, "dynamic_synthetic": 63 * 6, "spec_second": 171}


# ------------------------------------------------------------------------------------------------------------------------------ utilities
class Tee:
    def __init__(self, stream, path):
        self.stream, self.f = stream, open(path, "a", encoding="utf-8")

    def write(self, s):
        try:
            self.stream.write(s)
        except UnicodeEncodeError:
            self.stream.write(s.encode("ascii", "replace").decode())
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.stream.flush()
        self.f.flush()


def now_s() -> float:
    return time.time()


def stamp(t: float | None = None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t or now_s()))


def say(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def keep_awake():
    """Ask Windows not to sleep while this process runs (the screen may still turn off). No effect elsewhere."""
    if os.name == "nt":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
            say("Windows will not go to sleep while this runs (keep the charger connected).")
        except Exception:
            pass


def next_pacific_midnight() -> float | None:
    """Gemini free-tier daily quotas reset at midnight Pacific time."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo("America/Los_Angeles")
    except Exception:
        return None
    n = _dt.datetime.now(tz)
    nxt = (n + _dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return nxt.timestamp()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def records(path):
    from src.utils.io import read_jsonl
    return read_jsonl(path) if os.path.exists(path) else []


# ------------------------------------------------------------------------------------------------------------------------------ state
class Ctx:
    def __init__(self, dry: bool, quick: bool, base: str | None = None):
        self.dry, self.quick = dry, quick
        self.base = base or ("outputs_dryrun" if dry else "outputs")
        self.home = os.path.join(self.base, "free_run")
        os.makedirs(self.home, exist_ok=True)
        self.raw = os.path.join(self.base, "raw")
        self.nl = os.path.join(self.base, "nl_spec")
        self.tc = os.path.join(self.base, "tempcheck")
        self.state_path = os.path.join(self.home, "state.json")
        self.state = json.load(open(self.state_path, encoding="utf-8")) if os.path.exists(self.state_path) else \
            {"created": stamp(), "dry_run": dry, "models": {}, "jobs": {}}
        self.cfg = yaml.safe_load(open("config/free_run.yaml", encoding="utf-8"))

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=1)
        os.replace(tmp, self.state_path)
        self.write_index()

    def job(self, name) -> dict:
        return self.state["jobs"].setdefault(name, {"status": "pending", "dir": None, "sessions": 0, "history": []})

    def write_index(self):
        """outputs/free_run/results_index.json: which run folder holds which result. scripts/make_paper_numbers.py reads it."""
        J, M = self.state["jobs"], self.state["models"]
        ent = lambda k: {"dir": J[k].get("dir"), "complete": J[k]["status"] == "done", "status": J[k]["status"], **M.get(J[k].get("model_key", ""), {})} if k in J else None
        idx = {"dry_run": self.dry, "main_model": M.get("main"), "tempcheck": ent("tempcheck"), "spec": ent("spec"), "bengaluru": ent("bengaluru"),
               "dynamic": ent("dynamic"), "dynamic_synthetic": ent("dynamic_synthetic"), "groq": ent("groq"), "spec_second": ent("spec_second"),
               "second_models": [dict(ent(k), label=J[k].get("label")) for k in J if k.startswith("second:")]}
        with open(os.path.join(self.home, "results_index.json"), "w", encoding="utf-8") as fh:
            json.dump(idx, fh, indent=1)


# ------------------------------------------------------------------------------------------------------------------------------ models
def make(provider, model):
    from src.planner.llm import make_client
    cfg = yaml.safe_load(open(D, encoding="utf-8"))
    cfg.update(provider=provider, model=model)
    return make_client(cfg)


def resolve_auto(requested: str, available: list[str]) -> list[str]:
    """Candidate ids for 'auto-gemma' / 'auto-flash', best first."""
    def ver(i):
        m = re.search(r"-(\d+(?:\.\d+)?)", i)
        return float(m.group(1)) if m else 0.0
    if requested == "auto-gemma":
        c = [i for i in available if re.match(r"^gemma-\d", i) and i.endswith("-it") and not re.search(r"-e\d+b", i)]
        size = lambda i: float((re.search(r"(\d+)b", i) or [0, 0])[1])
        return sorted(c, key=lambda i: (ver(i), size(i)), reverse=True)
    if requested == "auto-flash":
        c = [i for i in available if re.match(r"^gemini-\d+(\.\d+)?-flash(-\d+)?$", i)]
        return sorted(c, key=lambda i: ver(i), reverse=True) + \
            sorted([i for i in available if re.match(r"^gemini-\d+(\.\d+)?-flash-preview", i)], key=ver, reverse=True)
    return [requested]


def test_call(provider, model) -> tuple[str, str]:
    """('ok' | 'quota' | 'auth' | 'bad_model' | 'error', message). One tiny call."""
    from src.planner.llm import LLMError, stop_info
    try:
        c = make(provider, model)
        r = c.generate("You are a test endpoint. Answer with JSON only.", 'Return exactly {"ok": true}', 0.0, 64, 0)
        return ("ok", r.text.strip()[:80]) if r.text.strip() else ("error", "empty reply")
    except LLMError as e:
        k = stop_info(e)["kind"]
        return ({"daily_quota": "quota", "rate_limit": "quota", "auth": "auth", "bad_request": "bad_model"}.get(k, "error"), str(e)[:300])


# ------------------------------------------------------------------------------------------------------------------------------ preflight
def preflight(ctx: Ctx, skip_tests: bool) -> bool:
    say("=== preflight ===")
    ok = True
    if sys.version_info < (3, 10):
        say(f"[FAIL] Python {sys.version.split()[0]}: 3.10 or newer is needed.")
        return False
    for mod in ("numpy", "pandas", "scipy", "matplotlib", "yaml"):
        try:
            __import__(mod)
        except ImportError:
            say(f"[FAIL] package {mod} missing: run  pip install -r requirements.txt")
            ok = False
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    if free_gb < 2:
        say(f"[FAIL] only {free_gb:.1f} GB free disk space; 2 GB are needed.")
        ok = False
    for d in ("benchmark/real", "benchmark/generated", "benchmark/bengaluru"):
        n = len(glob.glob(os.path.join(d, "*.json")))
        if n == 0:
            say(f"[FAIL] {d} is empty: unzip the project again.")
            ok = False
    if not ok:
        return False
    from src.utils.io import code_fingerprint
    fp = code_fingerprint()["code_sha256"]
    if ctx.state.get("code_sha256") and ctx.state["code_sha256"] != fp:
        say("[note] the code changed since this run state was created; finished results are kept, new calls use the new code (recorded per run).")
    ctx.state["code_sha256"] = fp
    if not skip_tests and ctx.state.get("tests_passed_for") != fp:
        say("running the test suite (about 1-2 minutes) ...")
        r = subprocess.run([sys.executable, "-X", "utf8", "-m", "unittest", "discover", "-s", "tests", "-t", "."], capture_output=True, text=True, timeout=1800,
                           env=dict(os.environ, PYTHONUTF8="1"))
        log = os.path.join(ctx.home, "preflight_tests.log")
        open(log, "w", encoding="utf-8").write(r.stdout + r.stderr)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        if r.returncode != 0:
            bad = sorted(set(re.findall(r"^(?:FAIL|ERROR): \S+ \((tests\.[A-Za-z0-9_]+)", r.stdout + r.stderr, re.M)))
            core = [b for b in bad if b in CORE_TESTS] or ([] if bad else ["(could not tell which tests failed)"])
            if core:
                say(f"[FAIL] core unit tests failed ({', '.join(core)}). Nothing was sent to any model. Please send {log} to get it fixed.")
                return False
            say(f"[warn] tests outside the experiment core failed ({', '.join(bad)}); the experiments are not affected. Please send {log} later.")
        else:
            say(f"[ok] tests: {' | '.join(tail)}")
        ctx.state["tests_passed_for"] = fp
    if ctx.dry:
        for key in model_keys(ctx):
            ctx.state["models"][key] = {"provider": "mock", "model": "mock-heuristic-v1", "requested": "dry-run stand-in", "label": label_of(ctx, key)}
        ctx.save()
        say("[ok] dry run: every model is replaced by the offline stand-in; no key is needed.")
        return True
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        say("[FAIL] GEMINI_API_KEY is not set. Create a free key at https://aistudio.google.com/apikey and put this line in the file .env next to run_free.py:\n"
            "          GEMINI_API_KEY=your-key-here")
        return False
    available = None
    for key in model_keys(ctx):
        if key in ctx.state["models"]:
            m = ctx.state["models"][key]
            say(f"[ok] {key}: {m['provider']}/{m['model']} (fixed earlier)")
            continue
        req = model_req(ctx, key)
        prov, want = req["provider"], req["model"]
        if prov == "groq" and not os.environ.get("GROQ_API_KEY"):
            say("[skip] groq: GROQ_API_KEY not set")
            continue
        cands = [want]
        if want.startswith("auto-"):
            if available is None:
                try:
                    available = make("gemini", "listing").list_models()
                    say(f"[ok] your key can use {len(available)} models")
                except Exception as e:
                    say(f"[FAIL] could not list models ({e}). Write exact model ids instead of auto-... in config/free_run.yaml.")
                    return False
            cands = resolve_auto(want, available)
            if not cands:
                say(f"[skip] {key}: no model matches {want} for your key")
                continue
        chosen = None
        for c in cands[:4]:
            st, msg = test_call(prov, c)
            if st == "error" and ("network error" in msg or "HTTP 599" in msg):
                say(f"[FAIL] no connection to {prov} ({msg[:160]}). Check the internet connection, VPN or firewall, then start the same command again.")
                return False
            if st in ("ok", "quota"):
                chosen = c
                say(f"[ok] {key}: {prov}/{c} ({'test call answered' if st == 'ok' else 'quota already used today; will wait'})")
                break
            if st == "auth":
                say(f"[FAIL] the {prov} key was rejected: {msg}")
                return False
            say(f"[try next] {prov}/{c}: {msg}")
        if chosen is None:
            if key == "main":
                say(f"[FAIL] the main model {prov}/{want} does not answer. Check the id in Google AI Studio and put it in config/free_run.yaml (main_model).")
                return False
            say(f"[skip] {key}: no working model")
            continue
        ctx.state["models"][key] = {"provider": prov, "model": chosen, "requested": want, "label": req.get("label", key)}
        ctx.save()
    return True


def model_keys(ctx):
    keys = ["main"] + [f"second:{m.get('label', m['model'])}" for m in (ctx.cfg.get("second_models") or [])]
    if ctx.cfg.get("groq_model") and "groq" in ctx.cfg.get("stages", []):
        keys.append("groq")
    return keys


def model_req(ctx, key):
    if key == "main":
        return ctx.cfg["main_model"]
    if key == "groq":
        return ctx.cfg["groq_model"]
    lab = key.split(":", 1)[1]
    return next(m for m in ctx.cfg["second_models"] if m.get("label", m["model"]) == lab)


def label_of(ctx, key):
    return "main" if key == "main" else model_req(ctx, key).get("label", key)


# ------------------------------------------------------------------------------------------------------------------------------ jobs
def plan_jobs(ctx):
    st = ctx.cfg.get("stages", [])
    jobs = []
    if "tempcheck" in st and not ctx.dry:
        jobs.append(("tempcheck", "main"))
    if "spec" in st:
        jobs.append(("spec", "main"))
    if "dynamic" in st:                      # first: it replaces the dynamic LLM runs made with the old (buggy) code
        jobs.append(("dynamic", "main"))
    if "bengaluru" in st:
        jobs.append(("bengaluru", "main"))
    if "dynamic_synthetic" in st:
        jobs.append(("dynamic_synthetic", "main"))
    if "second_models" in st:
        jobs += [(k, k) for k in model_keys(ctx) if k.startswith("second:") and k in ctx.state["models"]]
    if "spec_second" in st:                  # the language-interface test with the extra model (one call per mission)
        sec = [k for k in model_keys(ctx) if k.startswith("second:") and k in ctx.state["models"]]
        if sec:
            jobs.append(("spec_second", sec[0]))
    if "groq" in st and "groq" in ctx.state["models"]:
        jobs.append(("groq", "groq"))
    return jobs


def best_resume_dir(ctx, current, prov, model, sdir):
    """The run folder to continue: the one with the MOST saved runs for this provider/model and mission set (a Ctrl-C at the wrong moment can leave the
    stage without a recorded folder, and a restart would otherwise begin a new, empty run)."""
    from src.benchmark.generator import load_scenarios
    want = {s["scenario_id"] for s in load_scenarios(sdir)}
    best, best_n = current, (len(records(os.path.join(current, "runs.jsonl"))) if current and os.path.isdir(current) else -1)
    for mf in glob.glob(os.path.join(ctx.raw, "*", "manifest.json")):
        try:
            man = json.load(open(mf, encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if man.get("provider") != prov or man.get("model") != model or set(man.get("scenario_ids") or []) - want or man.get("kind") == "dynamic":
            continue
        d = os.path.dirname(mf)
        n = len(records(os.path.join(d, "runs.jsonl")))
        if n > best_n:
            best, best_n = d, n
    if best != current:
        say(f"[resume] continuing the run with the most saved results: {best} ({best_n} runs)")
    return best


def run_job(ctx, name, mkey) -> dict:
    """Returns {'kind': 'done' | 'daily_quota' | 'rate_limit' | 'auth' | 'bad_request' | 'provider_error' | 'interrupted' | 'other', ...}."""
    from src.planner.llm import LLMError, stop_info
    m = ctx.state["models"][mkey]
    J = ctx.job(name)
    J["model_key"] = mkey
    J["label"] = m.get("label")
    if mkey != "main" and not os.environ.get("CA_CLP_HTTP_TIMEOUT"):
        os.environ["CA_CLP_HTTP_TIMEOUT"] = "300"        # the extra (larger) model often needs more than 90 s per reply on the free tier
    J["sessions"] += 1
    limit = 3 if ctx.quick else None
    prov, model = m["provider"], m["model"]
    if name == "tempcheck":
        from src.planner.temperature_check import run_tempcheck
        try:
            ok = run_tempcheck(prov, model, n=6, say=lambda *a: say(*a), out_root=ctx.tc, raise_errors=True)
        except LLMError as e:
            return stop_info(e)
        files = sorted(glob.glob(os.path.join(ctx.tc, f"tempcheck_{prov}_{model.replace('/', '_')}_*.json")))
        J["dir"] = files[-1] if files else None
        return {"kind": "done"} if ok and files else {"kind": "other", "message": "tempcheck wrote no file"}
    if name in ("spec", "spec_second"):
        spec = load_module(os.path.join("scripts", "spec_extraction_eval.py"), "spec_extraction_eval")
        argv = ["--provider", "truth" if ctx.dry else prov, "--model", model, "--scenario-dir", *SPEC_DIRS, "--out-root", ctx.nl]
        if prov == "groq":
            argv += ["--config", "config/default_groq.yaml"]
        if J.get("dir"):
            argv += ["--resume", J["dir"]]
        if limit:
            argv += ["--limit", str(limit)]
        out = spec.main(argv)
        J["dir"] = out
        s = json.load(open(os.path.join(out, "summary.json"), encoding="utf-8")) if os.path.exists(os.path.join(out, "summary.json")) else {}
        if s.get("complete"):
            return {"kind": "done"}
        man = json.load(open(os.path.join(out, "manifest.json"), encoding="utf-8"))
        return man.get("stop") or {"kind": "other", "message": "incomplete"}
    if name in ("dynamic", "dynamic_synthetic"):
        from src.experiments.dynamic_runner import run_dynamic_experiment
        cfg_, sdir_ = (DYN, "benchmark/real") if name == "dynamic" else (DYN_SYN, "benchmark/generated")
        d = run_dynamic_experiment(D, cfg_, prov, model, raw_root=ctx.raw, scenario_dir=[sdir_], resume_dir=J.get("dir"),
                                   progress=lambda *a: say(*a), limit_cases=limit)
        J["dir"] = d
        man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
        if (man.get("completeness") or {}).get("complete"):
            return {"kind": "done"}
        return man.get("stop") or {"kind": "other", "message": man.get("stopped_reason") or "incomplete"}
    # static experiments: Bengaluru set, extra models on the airport set
    from src.experiments.runner import run_experiment
    from src.benchmark.generator import load_scenarios
    sdir, conds = ("benchmark/bengaluru", BENGALURU) if name == "bengaluru" else ("benchmark/real", SECOND)
    dcfg = D if name == "bengaluru" else ("config/default_groq.yaml" if prov == "groq" else "config/default_extra_models.yaml")     # extra models: larger reply cap (reasoning models), same prompts
    J["dir"] = best_resume_dir(ctx, J.get("dir"), prov, model, sdir)
    d = run_experiment(dcfg, EXP, prov, model, raw_root=ctx.raw, scenario_dir=sdir, seeds=[0], conditions=conds, resume_dir=J.get("dir"),
                       limit=limit, progress=lambda *a: say(*a))
    J["dir"] = d
    scen = load_scenarios(sdir)[:limit] if limit else load_scenarios(sdir)
    got = {(r["scenario_id"], r["condition"], r["seed"]) for r in records(os.path.join(d, "runs.jsonl"))}
    need = {(s["scenario_id"], c, 0) for s in scen for c in conds}
    if need <= got:
        return {"kind": "done"}
    man = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    st = man.get("stop") or {"kind": "other", "message": man.get("stopped_reason")}
    if st.get("kind") in (None, "none"):
        st = {"kind": "other", "message": f"{len(need - got)} runs missing"}
    return st


def paper_job(ctx) -> bool:
    say("=== paper numbers ===")
    env = dict(os.environ, PYTHONUTF8="1", PAPER_FREE_INDEX=os.path.join(ctx.home, "results_index.json"))
    if ctx.dry:
        env.update(PAPER_INCLUDE_MOCK="1", PAPER_RAW_ROOT=ctx.raw, PAPER_NL_ROOT=ctx.nl, PAPER_GEN_DIR=os.path.join(ctx.base, "paper_generated"),
                   PAPER_FIG_DIR=os.path.join(ctx.base, "paper_figures"))
    ok = True
    for script in ["scripts/make_paper_numbers.py"] + ([] if ctx.dry else ["scripts/check_paper_numbers.py"]):
        r = subprocess.run([sys.executable, "-X", "utf8", script], capture_output=True, text=True, env=env, timeout=3600)
        out = (r.stdout + r.stderr).strip().splitlines()
        say(f"{script}: {'ok' if r.returncode == 0 else 'FAILED'} | " + " | ".join(out[-2:]))
        if r.returncode != 0:
            open(os.path.join(ctx.home, os.path.basename(script) + ".log"), "w", encoding="utf-8").write(r.stdout + r.stderr)
            ok = False
    if not ctx.dry and shutil.which("latexmk"):
        r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "paper.tex"], cwd="paper", capture_output=True, text=True, timeout=600)
        say(f"latexmk: {'paper/paper.pdf rebuilt' if r.returncode == 0 else 'failed (compile on Overleaf instead)'}")
    elif not ctx.dry:
        say("LaTeX is not installed here: upload the folder paper/ to Overleaf and compile there (the numbers are already updated in paper/generated/).")
    pack(ctx)
    return ok


def pack(ctx):
    z = os.path.join(ctx.home, "RESULTS_TO_SEND.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ("state.json", "results_index.json", "log.txt", "preflight_tests.log"):
            p = os.path.join(ctx.home, f)
            if os.path.exists(p):
                zf.write(p, os.path.join("free_run", f))
        for j in ctx.state["jobs"].values():
            d = j.get("dir")
            if not d or not os.path.exists(d):
                continue
            if os.path.isfile(d):
                zf.write(d, os.path.relpath(d, ctx.base))
                continue
            for p in glob.glob(os.path.join(d, "**", "*"), recursive=True):
                if os.path.isfile(p) and not p.endswith(".png"):
                    zf.write(p, os.path.relpath(p, ctx.base))
        gen = os.path.join(ctx.base, "paper_generated") if ctx.dry else os.path.join("paper", "generated")
        for p in glob.glob(os.path.join(gen, "*")):
            zf.write(p, os.path.join("paper_generated", os.path.basename(p)))
    say(f"results packed: {z} ({os.path.getsize(z) / 1e6:.1f} MB)")


# ------------------------------------------------------------------------------------------------------------------------------ main loop
def status(ctx):
    print(f"\nstate: {ctx.state_path}  (created {ctx.state.get('created')})")
    for k, m in ctx.state.get("models", {}).items():
        print(f"  model {k:<28} {m['provider']}/{m['model']}")
    for name, mk in plan_jobs(ctx) or [(n, j.get("model_key")) for n, j in ctx.state["jobs"].items()]:
        j = ctx.state["jobs"].get(name, {"status": "pending"})
        extra = f" until {stamp(j['blocked_until'])}" if j.get("status") == "waiting" and j.get("blocked_until") else ""
        print(f"  {name:<34} {j['status']:<9}{extra}  {j.get('dir') or ''}")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="offline stand-ins, writes to outputs_dryrun/ only")
    ap.add_argument("--quick", action="store_true", help="with --dry-run: 3 missions per stage")
    ap.add_argument("--no-wait", action="store_true", help="stop at the first exhausted daily quota instead of waiting")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--only", help="comma-separated job names, e.g. spec,bengaluru")
    ap.add_argument("--out-base", help=argparse.SUPPRESS)        # tests: write everything below this folder
    a = ap.parse_args()
    ctx = Ctx(a.dry_run, a.quick, a.out_base)
    if a.out_base:                      # tests: keep even the oracle cache inside the given folder
        os.environ["CA_CLP_ORACLE_CACHE"] = os.path.join(a.out_base, "oracle_cache")
    sys.stdout = Tee(sys.stdout, os.path.join(ctx.home, "log.txt"))
    sys.stderr = Tee(sys.stderr, os.path.join(ctx.home, "log.txt"))
    if a.status:
        status(ctx)
        return 0
    say(f"run_free.py started ({'DRY RUN, offline' if ctx.dry else 'real run'}); log: {os.path.join(ctx.home, 'log.txt')}")
    keep_awake()
    if not preflight(ctx, a.skip_tests):
        ctx.save()
        say("Preflight failed: nothing was run. Fix the [FAIL] line above and start the same command again.")
        return 2
    jobs = plan_jobs(ctx)
    if a.only:
        keep = {k.strip() for k in a.only.split(",")}
        if keep & {"second_models", "second"}:
            keep |= {j[0] for j in jobs if j[0].startswith("second:")}
        jobs = [j for j in jobs if j[0] in keep]
        if not jobs:
            say(f"--only {a.only}: no matching stage (names: {', '.join(n for n, _ in plan_jobs(ctx))}, second_models)")
            return 2
    say("plan (worst-case model calls; usually fewer):")
    for n, mk in jobs:
        m = ctx.state["models"][mk]
        say(f"   {n:<32} {m['provider']}/{m['model']:<28} <= {WORST.get(n.split(':')[0], WORST['second'])} calls   [{ctx.job(n)['status']}]")
    # a new session always tries again at once: the quota may have reset, and a refused call costs nothing
    ctx.state["model_blocked_until"] = {}
    for n_, _ in jobs:
        if ctx.job(n_)["status"] == "failed":            # a new session = the user fixed the cause (e.g. a new key): try again, resuming
            ctx.job(n_)["status"] = "pending"
        if ctx.job(n_)["status"] == "waiting":
            ctx.job(n_)["status"], ctx.job(n_)["blocked_until"] = "pending", None
    wait = ctx.cfg.get("wait_for_quota", True) and not a.no_wait
    t_begin = now_s()
    while True:
        mblock = ctx.state.setdefault("model_blocked_until", {})     # a daily quota belongs to a MODEL: every stage that uses it waits
        for name, mk in jobs:
            J = ctx.job(name)
            if J["status"] in ("done", "failed"):
                continue
            if (mblock.get(mk) or 0) > now_s() or (J.get("blocked_until") or 0) > now_s():
                J["status"], J["blocked_until"] = "waiting", max(mblock.get(mk) or 0, J.get("blocked_until") or 0)
                continue
            m = ctx.state["models"][mk]
            say(f"=== {name}  ({m['provider']}/{m['model']}) ===")
            J["status"] = "running"
            ctx.save()
            try:
                res = run_job(ctx, name, mk)
            except KeyboardInterrupt:
                res = {"kind": "interrupted"}
            except Exception as e:      # a bug must not lose the other stages: record it and carry on
                res = {"kind": "other", "message": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-3000:]}
            J["history"].append({"when": stamp(), **{k: v for k, v in res.items() if k != "traceback"}})
            kind = res.get("kind")
            if kind == "done":
                J["status"], J["blocked_until"] = "done", None
                say(f"[done] {name}")
            elif kind == "interrupted":
                J["status"] = "pending"
                ctx.save()
                say("Stopped by you. Everything finished so far is saved; run the same command again to continue.")
                return 130
            elif kind in ("daily_quota", "rate_limit"):
                ra = res.get("retry_after_s") or 0
                if kind == "daily_quota" and m["provider"] == "gemini" and next_pacific_midnight():
                    until = max(next_pacific_midnight() + 300, now_s() + ra + 60)       # Gemini free quotas reset at midnight Pacific time
                elif kind == "daily_quota":
                    until = now_s() + max(3600, ra + 60)                                   # unknown reset time: try again every hour
                else:
                    until = now_s() + max(300, ra + 30)
                mblock[mk] = until
                J["status"], J["blocked_until"] = "waiting", until
                say(f"[quota] {m['model']}: {'daily free quota used up' if kind == 'daily_quota' else 'rate limit'}; continues after {stamp(until)}")
            elif kind in ("network", "server"):
                # no connection / provider overloaded: not the experiment's fault and not counted as a failure; try again in 10 minutes
                J["outages"] = J.get("outages", 0) + 1
                J["status"], J["blocked_until"] = "waiting", now_s() + 600
                say(f"[{'no connection' if kind == 'network' else 'provider busy'}] {name}: trying again at {stamp(now_s() + 600)} ({res.get('message', '')[:120]})")
            elif kind in ("auth", "bad_request"):
                J["status"] = "failed"
                say(f"[FAILED] {name}: {res.get('message')}\n   This needs a fix (key or model id), not waiting. Send outputs/free_run/log.txt.")
            else:
                J["fails"] = J.get("fails", 0) + 1
                if "traceback" in res:
                    open(os.path.join(ctx.home, f"error_{name.replace(':', '_').replace(' ', '_')}.log"), "w", encoding="utf-8").write(res["traceback"])
                if J["fails"] >= 4:
                    J["status"] = "failed"
                    say(f"[FAILED] {name} after {J['fails']} attempts: {res.get('message')}. Send outputs/free_run/log.txt and the error_*.log file.")
                else:
                    J["status"], J["blocked_until"] = "waiting", now_s() + 600
                    say(f"[retry later] {name}: {res.get('message')} (attempt {J['fails']}/4, retrying in 10 min)")
            ctx.save()
        left = [(n, ctx.job(n)) for n, _ in jobs if ctx.job(n)["status"] not in ("done", "failed")]
        if not left:
            break
        until = min(j.get("blocked_until") or now_s() for _, j in left)
        if not wait or (until - t_begin) > ctx.cfg.get("max_wait_hours", 72) * 3600:
            say(f"Not finished: {', '.join(n for n, _ in left)}. Run the same command again after {stamp(until)}.")
            break
        paper_job(ctx)              # keep the paper numbers and the results zip current while waiting
        say(f"waiting until {stamp(until)} for the quota to reset (Ctrl-C is safe; run the same command later to continue)")
        try:
            while now_s() < until:
                time.sleep(min(600, max(1, until - now_s())))
                if now_s() < until:
                    say(f"  ... still waiting ({(until - now_s()) / 3600:.1f} h left)")
        except KeyboardInterrupt:
            ctx.save()
            say("Stopped while waiting. Run the same command again later to continue.")
            return 130
    ok = paper_job(ctx)
    status(ctx)
    failed = [n for n, _ in jobs if ctx.job(n)["status"] == "failed"]
    done_all = all(ctx.job(n)["status"] == "done" for n, _ in jobs)
    if done_all:
        say(f"ALL DONE. Send {os.path.join(ctx.home, 'RESULTS_TO_SEND.zip')}.")
    elif failed:
        say(f"Finished with failed stages: {', '.join(failed)}. Send {os.path.join(ctx.home, 'RESULTS_TO_SEND.zip')} and log.txt.")
    return 0 if (done_all and ok) else 1


if __name__ == "__main__":
    sys.exit(main())
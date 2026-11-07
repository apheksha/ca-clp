"""One-command pipeline: preflight -> static run -> dynamic run -> evaluation -> merged report. Safe to re-run: it RESUMES.
Each run lives in its own folder outputs/runs/<model>[_<dataset>]_<profile>_<timestamp>/ (state.json, raw/, processed/, figures/, FINAL_REPORT.md);
folders written by older versions under outputs/final/<provider>__<model>__<profile>/ are still found when resuming. A run is bound to one
code+config fingerprint: if the code changes between sessions the pipeline refuses to continue (mixing code versions would make the results
unreproducible)."""
from __future__ import annotations
import json
import os
import re
import time
import yaml
from src.benchmark.generator import load_scenarios
from src.experiments.runner import run_experiment, load_yaml, max_calls_per_scenario_seed
from src.experiments.dynamic_runner import run_dynamic_experiment, build_cases
from src.models.schemas import Mission
from src.repair.loop import LoopConfig
from src.planner.llm import make_client, LLMError
from src.planner.planner import LLMPlanner
from src.utils.io import code_fingerprint, env_manifest

D = "config/default.yaml"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def profile(name: str) -> dict:
    with open("config/profiles.yaml", encoding="utf-8") as f:
        p = yaml.safe_load(f)
    if name not in p:
        raise SystemExit(f"unknown profile {name!r}; available: {sorted(p)}")
    return p[name]


def state_dir(provider: str, model: str, prof: str) -> str:
    """Folder layout of earlier versions (outputs/final/<provider>__<model>__<profile>); new runs use make_run_dir()."""
    return os.path.join("outputs", "final", f"{_slug(provider)}__{_slug(model)}__{prof}")


def estimate(prof: dict, seeds=None) -> dict:
    """Worst-case number of LLM calls (no API used). Typical is lower: loops stop at the first success."""
    exp = load_yaml(prof["exp_config"])
    scen = load_scenarios(prof["scenario_dir"])
    conds = [LoopConfig(name=n, **c) for n, c in exp["conditions"].items()]
    sd = seeds if seeds is not None else exp["seeds"]
    llm = [c for c in conds if c.kind not in ("classical", "astar")]
    static = max_calls_per_scenario_seed(conds) * len(scen) * len(sd) if llm else 0
    dcfg = exp.get("dynamic", {})
    out = {"n_scenarios": len(scen), "static_seeds": list(sd), "static_worst_case_calls": static}
    if dcfg:
        cases = build_cases(scen, dcfg)
        dn = [LoopConfig(name=n, **exp["conditions"][n]) for n in (dcfg.get("conditions") or list(exp["conditions"]))]
        per = sum(c.max_iterations for c in dn if c.kind in ("closed_loop", "resample"))
        dsd = dcfg.get("seeds", [0])
        out.update(n_dynamic_cases=len(cases), dynamic_seeds=list(dsd), dynamic_worst_case_calls=per * len(cases) * len(dsd))
    out["total_worst_case_calls"] = out["static_worst_case_calls"] + out.get("dynamic_worst_case_calls", 0)
    return out


def preflight(provider: str, model: str, prof_name: str, smoke: bool = True, say=print) -> bool:
    prof = profile(prof_name)
    ok_all = True

    def gate(ok, text):
        nonlocal ok_all
        ok_all &= bool(ok)
        say(f"  [{'PASS' if ok else 'FAIL'}] {text}")

    say(f"PREFLIGHT provider={provider} model={model} profile={prof_name}")
    fp = code_fingerprint()
    env = env_manifest()
    gate(os.path.isdir(prof["scenario_dir"]) and len(load_scenarios(prof["scenario_dir"])) > 0, f"scenario folder {prof['scenario_dir']} present")
    gate(os.path.exists(prof["exp_config"]), f"experiment config {prof['exp_config']} present")
    say(f"  [info] code fingerprint {fp['code_sha256'][:16]} over {fp['n_files']} files; git_commit={env['git_commit']} dirty={env['git_dirty']}")
    say("  [info] git commit is optional: the code fingerprint identifies the code even without git")
    e = estimate(prof)
    say(f"  [info] worst-case LLM calls: static {e['static_worst_case_calls']} + dynamic {e.get('dynamic_worst_case_calls', 0)} = {e['total_worst_case_calls']} "
        f"(typical about half). Compare with YOUR provider's limits before starting; this tool cannot know them.")
    cfg = load_yaml(D)
    cfg["provider"], cfg["model"] = provider, model
    cfg.update(load_yaml(prof["exp_config"]).get("planner", {}))
    try:
        client = make_client(cfg)
        gate(True, f"client constructed ({client.name})")
    except LLMError as ex:
        gate(False, f"client: {ex}")
        return False
    if smoke and not client.is_mock and load_scenarios(prof["scenario_dir"]):
        s = load_scenarios(prof["scenario_dir"])[0]
        t0 = time.time()
        try:
            planner = LLMPlanner(client, cfg.get("temperature", 0.0), cfg.get("max_tokens", 2000), cfg.get("max_parse_retries", 2), True, cfg.get("normalize_format", True))
            o = planner.initial_plan(Mission.from_dict(s["mission"]), 0)
            raw = o.calls[0]["raw_response"] if o.calls else ""
            gate(bool(raw.strip()), f"smoke call returned text ({len(raw)} chars, {time.time()-t0:.1f}s, {len(o.calls)} call(s) incl. re-asks)")
            gate(o.plan is not None, f"smoke reply parsed into a plan (error: {o.error})")
            if not raw.strip():
                say("    empty text with a 'thinking' model usually means the token limit was consumed by reasoning: raise max_tokens in config/default.yaml")
        except LLMError as ex:
            gate(False, f"smoke call failed: {ex}")
    say("PREFLIGHT " + ("OK" if ok_all else "FAILED"))
    return ok_all


def make_run_dir(model: str, prof_name: str, dataset: str = "", root: str = "outputs/runs") -> str:
    """outputs/runs/<model>[_<dataset>]_<profile>_<timestamp>; never returns a folder that already exists (suffix -2, -3, ...)."""
    parts = [_slug(model)] + ([_slug(dataset)] if dataset and dataset != prof_name else []) + [_slug(prof_name), time.strftime("%Y%m%d-%H%M%S")]
    base = os.path.join(root, "_".join(parts))
    path, k = base, 1
    while os.path.exists(path):
        k += 1
        path = f"{base}-{k}"
    return path


def _read_state(run_dir: str) -> dict | None:
    sp = os.path.join(run_dir, "state.json")
    try:
        with open(sp, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_state(path: str, st: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)


def find_latest_run_dir(model: str, prof_name: str, roots: list[str] | None = None, provider: str | None = None) -> str | None:
    """Newest run folder (by state.json mtime) of this model + profile (+ provider when given and recorded in the state)."""
    candidates = []
    for r in roots if roots is not None else ["outputs/runs", "outputs/final"]:
        if not os.path.isdir(r):
            continue
        for name in os.listdir(r):
            full_path = os.path.join(r, name)
            st = _read_state(full_path) if os.path.isdir(full_path) else None
            if not st or st.get("model") != model or st.get("profile") != prof_name:
                continue
            if provider is not None and st.get("provider", provider) != provider:
                continue
            candidates.append((os.path.getmtime(os.path.join(full_path, "state.json")), full_path))
    return max(candidates)[1] if candidates else None


def run_final(provider: str, model: str, prof_name: str, max_calls: int | None = None, seeds=None, auto_retries: int = 3, retry_wait_min: float = 10.0,
              allow_code_change: bool = False, skip_smoke: bool = False, resume_raw: str | None = None, resume_dynamic_raw: str | None = None,
              fresh: bool = False, resume: bool | str = False, run_dir: str | None = None, say=print) -> str:
    prof = profile(prof_name)
    dataset = prof.get("dataset", prof_name)
    # Determine the run folder
    if run_dir:
        sdir = run_dir
        say(f"[run] Using specified run folder: {sdir}")
    elif isinstance(resume, str) and resume:
        if not os.path.isdir(resume):
            raise SystemExit(f"--resume: folder not found: {resume}")
        sdir = resume
        say(f"[resume] Resuming specified run folder: {sdir}")
    elif fresh:
        sdir = make_run_dir(model, prof_name, dataset)
        say(f"[new run] Created fresh run folder: {sdir}")
    else:
        latest = find_latest_run_dir(model, prof_name, provider=provider)
        st_latest = _read_state(latest) if latest else None
        if latest and (resume is True or not (st_latest or {}).get("static_done") or not (st_latest or {}).get("dynamic_done")):
            sdir = latest
            say(f"[resume] Resuming run folder: {sdir} (use --fresh/--new-run to start a new run)")
        else:
            sdir = make_run_dir(model, prof_name, dataset)
            say(f"[new run] {'Previous run was completed. ' if latest else ''}Created new run folder: {sdir}")

    os.makedirs(sdir, exist_ok=True)
    sp = os.path.join(sdir, "state.json")
    fp = code_fingerprint()["code_sha256"]
    st = _read_state(sdir) or {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "code_sha256": fp, "provider": provider, "model": model,
                               "profile": prof_name, "dataset": dataset,
                               "static_dir": None, "dynamic_dir": None, "static_done": False, "dynamic_done": False, "sessions": 0}
    raw_dir = os.path.join(sdir, "raw")
    proc_dir = os.path.join(sdir, "processed")
    fig_dir = os.path.join(sdir, "figures")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    if resume_raw:
        if not os.path.isdir(resume_raw):
            raise SystemExit(f"--resume-raw: folder not found: {resume_raw}")
        st["static_dir"], st["static_done"] = resume_raw, False
    if resume_dynamic_raw:
        if not os.path.isdir(resume_dynamic_raw):
            raise SystemExit(f"--resume-dynamic-raw: folder not found: {resume_dynamic_raw}")
        st["dynamic_dir"], st["dynamic_done"] = resume_dynamic_raw, False
    if st["code_sha256"] != fp and not allow_code_change:
        raise SystemExit(f"code/config changed since this run started (was {st['code_sha256'][:12]}, now {fp[:12]}). Mixing versions breaks reproducibility. "
                         f"Restore the code, or start a new run folder, or pass --allow-code-change and DISCLOSE it.")
    if st["sessions"] == 0 and not preflight(provider, model, prof_name, smoke=not skip_smoke, say=say):
        raise SystemExit("preflight failed: nothing was run")
    st["sessions"] += 1
    budget = max_calls
    for stage in ("static", "dynamic"):
        if st[f"{stage}_done"]:
            continue
        for attempt in range(auto_retries + 1):
            kw = dict(provider=provider, model=model, max_calls=budget, resume_dir=st[f"{stage}_dir"], raw_root=raw_dir)
            if stage == "static":
                d = run_experiment(D, prof["exp_config"], scenario_dir=prof["scenario_dir"], seeds=seeds, **kw)
            else:
                d = run_dynamic_experiment(D, prof["exp_config"], scenario_dir=prof["scenario_dir"], **kw)
            st[f"{stage}_dir"] = d
            _write_state(sp, st)
            with open(os.path.join(d, "manifest.json"), encoding="utf-8") as f:
                man = json.load(f)
            used = man.get("api_calls_this_session", 0)
            if budget is not None:
                budget = max(0, budget - used)
            reason = man.get("stopped_reason")
            if reason is None:
                st[f"{stage}_done"] = True
                break
            say(f"[{stage}] stopped: {reason}")
            if "budget" in reason or "daily quota" in reason or "interrupted" in reason or attempt == auto_retries:
                _write_state(sp, st)
                say(f"\nSTOPPED (not finished). Run the SAME command again later to continue: it resumes from {d}.")
                return sdir
            say(f"[{stage}] transient provider error: waiting {retry_wait_min:.0f} min, then resuming automatically ({attempt+1}/{auto_retries})")
            time.sleep(retry_wait_min * 60)
        _write_state(sp, st)
    from src.evaluation.final_report import build_final_report
    rp = build_final_report(sdir)
    say(f"\nDONE. Report: {rp}")
    return sdir

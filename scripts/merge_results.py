#!/usr/bin/env python
"""Bring the results of run_free.py from ANOTHER laptop into this one (e.g. the Gemma stage run on a second laptop with its own key).

On the other laptop (same code + raw-runs zips, its own .env key from a DIFFERENT Google account):
    python run_free.py --only second_models
    -> when it prints ALL DONE, copy outputs/free_run/RESULTS_TO_SEND.zip to this laptop.
On this laptop (the one that runs the other stages):
    python scripts/merge_results.py path/to/RESULTS_TO_SEND.zip
It copies the finished run folders into outputs/, records them in outputs/free_run/state.json (so results_index.json and the paper see them),
rebuilds the paper numbers and re-packs this laptop's RESULTS_TO_SEND.zip. Only stages that are COMPLETE on the other laptop are taken, and a stage
that is already complete here is never overwritten. Nothing is recomputed or edited: the files are copied as they are.
"""
import argparse
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip", help="RESULTS_TO_SEND.zip from the other laptop")
    ap.add_argument("--jobs", help="comma-separated stage names to take (default: every stage that is complete there and not complete here)")
    ap.add_argument("--no-paper", action="store_true", help="only merge; do not rebuild the paper numbers")
    ap.add_argument("--base", help=argparse.SUPPRESS)          # tests
    a = ap.parse_args()
    import run_free as rf
    zf = zipfile.ZipFile(a.zip)
    names = set(zf.namelist())
    if "free_run/state.json" not in names:
        sys.exit("this zip has no free_run/state.json: use outputs/free_run/RESULTS_TO_SEND.zip from the other laptop")
    other = json.loads(zf.read("free_run/state.json").decode("utf-8"))
    remote_base = "outputs_dryrun" if other.get("dry_run") else "outputs"
    ctx = rf.Ctx(bool(other.get("dry_run")), False, a.base)
    if bool(ctx.state.get("dry_run")) != bool(other.get("dry_run")) and ctx.state.get("jobs"):
        sys.exit("one side is a dry run and the other a real run: not merged")
    want = {j.strip() for j in a.jobs.split(",")} if a.jobs else None
    merged = []
    for name, J in other.get("jobs", {}).items():
        if want is not None and name not in want:
            continue
        if J.get("status") != "done" or not J.get("dir"):
            print(f"[skip] {name}: not complete on the other laptop ({J.get('status')})")
            continue
        here = ctx.state["jobs"].get(name)
        if here and here.get("status") == "done":
            print(f"[skip] {name}: already complete on this laptop (kept)")
            continue
        parts = os.path.normpath(J["dir"]).replace("\\", "/").split("/")   # .../outputs/raw/real_2026...  ->  raw/real_2026...
        cut = max((i for i, p_ in enumerate(parts) if p_ in ("raw", "nl_spec", "tempcheck")), default=None)
        rel = "/".join(parts[cut:]) if cut is not None else os.path.relpath(J["dir"], remote_base)
        files = [n for n in names if n == rel or n.startswith(rel.rstrip("/") + "/")]
        if not files:
            print(f"[skip] {name}: its files are not in the zip")
            continue
        for n in files:
            if n.endswith("/"):
                continue
            dst = os.path.join(ctx.base, n)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with zf.open(n) as src, open(dst, "wb") as out:
                out.write(src.read())
        new = dict(J, dir=os.path.join(ctx.base, rel))
        new.setdefault("history", []).append({"when": rf.stamp(), "event": f"merged from {os.path.basename(a.zip)} (other laptop)"})
        ctx.state["jobs"][name] = new
        mk = J.get("model_key") or name
        if mk in other.get("models", {}):
            ctx.state.setdefault("models", {})[mk] = other["models"][mk]
        merged.append(name)
        print(f"[merged] {name}: {len(files)} files -> {new['dir']}")
    ctx.save()
    if not merged:
        print("nothing merged")
        return 0
    if not a.no_paper:
        rf.paper_job(ctx)
    print(f"done: {', '.join(merged)}. Send {os.path.join(ctx.home, 'RESULTS_TO_SEND.zip')}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

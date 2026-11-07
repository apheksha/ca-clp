"""Disk cache for feasibility-oracle verdicts (v12). The oracle needs up to a few minutes per terrain mission with many buildings, and every run session
starts by assessing all its missions; with the cache a resumed session starts at once.

Key: SHA-256 of the mission (canonical JSON) and of the oracle/verifier/geometry source files, so any change to the mission or to that code misses the
cache. A FEASIBLE entry is accepted only after its witness passes the verifier again (cheap), so a stale or edited cache file can never make an
infeasible mission look feasible. Folder: outputs/oracle_cache/ (safe to delete)."""
from __future__ import annotations
import hashlib
import json
import os
from src.benchmark.feasibility import Feasibility, assess
from src.models.schemas import FlightPlan, Mission

_SRC = ("src/benchmark/feasibility.py", "src/verifier/verifier.py", "src/models/schemas.py", "src/utils/geometry.py", "src/models/terrain.py")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_code = None


def _code_hash() -> str:
    global _code
    if _code is None:
        h = hashlib.sha256()
        for p in _SRC:
            with open(os.path.join(_ROOT, p), "rb") as fh:
                h.update(fh.read().replace(b"\r\n", b"\n"))
        _code = h.hexdigest()[:16]
    return _code


def key(m: Mission) -> str:
    return hashlib.sha256(json.dumps(m.to_dict(), sort_keys=True).encode()).hexdigest()[:24] + "_" + _code_hash()


def assess_cached(m: Mission, cache_dir: str | None = None) -> Feasibility:
    cache_dir = cache_dir or os.environ.get("CA_CLP_ORACLE_CACHE") or os.path.join(_ROOT, "outputs", "oracle_cache")
    path = os.path.join(cache_dir, key(m) + ".json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh)
            plan = FlightPlan.from_dict(d["witness_plan"]) if d.get("witness_plan") else None
            f = Feasibility(d["status"], d["reason"], d.get("lower_bound"), d.get("witness_distance"), plan, d.get("guarantee", ""), d.get("proof_kind"),
                            tuple(d["route_bracket"]) if d.get("route_bracket") else None)
            if f.status != "FEASIBLE":
                return f
            from src.verifier.verifier import verify
            if plan is not None and verify(m, plan).success:
                return f
        except Exception:
            pass                       # unreadable or failed re-check: recompute
    f = assess(m)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({**f.to_dict(), "witness_plan": f.witness_plan.to_dict() if f.witness_plan else None, "mission_id": m.mission_id}, fh)
        os.replace(tmp, path)
    except OSError:
        pass
    return f

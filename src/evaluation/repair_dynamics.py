"""Repair-process analysis (survey Objective 3): constraint recurrence, cascading violations, oscillation ("chasing"), convergence.
Pure functions of the stored iteration history; constraint identity = Violation.cid() (stable when detour waypoints are renamed).

Definitions (state_k = set of constraint ids violated by the plan produced at round k; base_k = the round whose plan was repaired):
  recurrence : a constraint id that was violated, then absent in a later state, then violated again (a fixed violation came back)
  cascade    : in round k at least one constraint id of state[base] was fixed AND at least one id not in state[base] appeared
               (fixing one violation triggered another elsewhere); cross-family pairs (fixed family -> new family) are counted
  oscillation: final plan not successful AND (a state repeats non-adjacently OR the violation count reverses direction >= 2 times)
  stuck      : two consecutive rounds with an identical state (no effective progress)"""
from __future__ import annotations


def cid_of(v: dict) -> tuple:
    t = v["type"]
    if t == "NFZ_SEGMENT":
        return (t, v.get("zone_id"))
    if t == "NFZ_WAYPOINT":
        return (t, v.get("waypoint_id"), v.get("zone_id"))
    if t == "ORDER_VIOLATION":
        return (t, tuple(v.get("evidence", {}).get("expected", [])))
    if v["family"] in ("distance", "energy"):
        return (t,)
    return (t, v.get("waypoint_id"))


def _fam(cid, fam_by_cid):
    return fam_by_cid.get(cid, "unknown")


def repair_dynamics(rec: dict) -> dict:
    its = [it for it in rec["iterations"] if it.get("verification")]
    out = {"dyn_n_rounds": len(rec["iterations"]) - 1, "dyn_iters_to_success": None, "dyn_n_recurrences": 0, "dyn_recurrence": False,
           "dyn_n_cascades": 0, "dyn_cascade": False, "dyn_oscillation": False, "dyn_stuck": False, "dyn_recurred_types": [], "dyn_cascade_pairs": []}
    if len(its) < 2:
        if its and its[0]["verification"]["success"]:
            out["dyn_iters_to_success"] = 0
        return out
    states, fam, idx = [], {}, {}
    for pos, it in enumerate(its):
        vs = it["verification"]["violations"] + it["verification"]["preservation_violations"]
        for v in vs:
            fam[cid_of(v)] = v["family"]
        states.append(frozenset(cid_of(v) for v in vs))
        idx[it["iteration"]] = pos
    for it in its:
        if it["verification"]["success"]:
            out["dyn_iters_to_success"] = it["iteration"]
            break
    # recurrence
    seen_present, seen_gone = {}, {}
    types = []
    for st in states:
        for c in set(seen_present) | set(st):
            if c in st:
                if seen_gone.get(c):
                    out["dyn_n_recurrences"] += 1
                    types.append(c[0])
                    seen_gone[c] = False
                seen_present[c] = True
            elif seen_present.get(c):
                seen_gone[c] = True
    out["dyn_recurrence"] = out["dyn_n_recurrences"] > 0
    out["dyn_recurred_types"] = types
    # cascades
    pairs = []
    for it in its[1:]:
        b = it.get("base")
        prev = states[idx[b]] if (b is not None and b in idx) else states[idx[it["iteration"]] - 1]
        cur = states[idx[it["iteration"]]]
        fixed, new = prev - cur, cur - prev
        if fixed and new:
            out["dyn_n_cascades"] += 1
            pairs += [f"{_fam(f, fam)}->{_fam(n, fam)}" for f in fixed for n in new]
    out["dyn_cascade"] = out["dyn_n_cascades"] > 0
    out["dyn_cascade_pairs"] = pairs
    # oscillation / stuck
    success = rec["final_verification"] is not None and rec["final_verification"]["success"]
    seq = [it["score"] for it in its]
    diffs = [b - a for a, b in zip(seq, seq[1:]) if b != a]
    reversals = sum(1 for a, b in zip(diffs, diffs[1:]) if (a > 0) != (b > 0))
    nonadj = any(states[i] == states[j] and states[i] for j in range(len(states)) for i in range(j - 1))
    out["dyn_oscillation"] = (not success) and (nonadj or reversals >= 2)
    out["dyn_stuck"] = any(states[i] == states[i + 1] and states[i] for i in range(len(states) - 1))
    return out

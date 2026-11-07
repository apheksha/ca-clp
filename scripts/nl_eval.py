#!/usr/bin/env python
"""Evaluate natural-language mission understanding on the human-written collection (data/human_missions/missions.yaml or any file in that format).

  python scripts/nl_eval.py                                  # rule-based baseline, offline
  python scripts/nl_eval.py --provider gemini --model <id>   # LLM extractor (needs the API key of the provider)

Reports per category: extraction F1 / exact intent match, decisions (ready / needs_clarification / conflict), false-clarification rate on complete instructions,
clarification recall on incomplete+ambiguous ones (and whether the right fields were asked), conflict-detection recall (and right conflict type), and the
operational intent check (plan for the extracted mission verified against the true mission). Writes outputs/nl_eval/*.json|csv."""
import argparse, csv, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.nl.dataset import load_collection
from src.nl.extraction import RuleExtractor, LLMExtractor
from src.nl.analysis import analyze
from src.nl.intent import compare, cross_validate


def evaluate(extractor, mdef, missions):
    rows = []
    for hm in missions:
        e = extractor.extract(hm.text)
        a = analyze(e, mdef, hm.text)
        c = compare(e.constraints(), hm.truth)
        want_fields = set(hm.expect.get("clarify", []))
        asked = set(a.missing) | {x["field"] for x in a.ambiguities} | ({"unknown_place"} if a.unknown_places else set())
        want_conf = set(hm.expect.get("conflicts", []))
        got_conf = {x["type"] for x in a.conflicts}
        cv = cross_validate(e.constraints(), hm.truth, mdef) if hm.category == "complete" else {}
        rows.append({"id": hm.id, "category": hm.category, "decision": a.decision, "f1": round(c["f1"], 3), "intent_exact": c["intent_preserved"],
                     "clarified_fields": sorted(asked), "expected_fields": sorted(want_fields), "fields_recall": (len(want_fields & asked) / len(want_fields)) if want_fields else None,
                     "conflicts_found": sorted(got_conf), "conflicts_expected": sorted(want_conf), "conflict_type_ok": (want_conf <= got_conf) if want_conf else None,
                     "operational_intent_ok": cv.get("plan_for_extracted_valid_for_truth"), "extraction_error": e.error})
    return rows


def summarize(rows):
    out = {}
    cats = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r)
    comp = cats.get("complete", [])
    out["n"] = len(rows)
    out["extraction_mean_f1"] = {k: round(sum(r["f1"] for r in v) / len(v), 3) for k, v in cats.items()}
    out["complete_exact_intent_rate"] = round(sum(r["intent_exact"] for r in comp) / max(len(comp), 1), 3)
    out["false_clarification_rate_on_complete"] = round(sum(r["decision"] != "ready" for r in comp) / max(len(comp), 1), 3)
    need = cats.get("incomplete", []) + cats.get("ambiguous", [])
    out["clarification_recall_on_incomplete_or_ambiguous"] = round(sum(r["decision"] == "needs_clarification" for r in need) / max(len(need), 1), 3)
    with_f = [r for r in need if r["fields_recall"] is not None]
    out["asked_right_fields_recall"] = round(sum(r["fields_recall"] for r in with_f) / max(len(with_f), 1), 3)
    conf = cats.get("conflicting", [])
    out["conflict_detection_recall"] = round(sum(r["decision"] == "conflict" for r in conf) / max(len(conf), 1), 3)
    out["conflict_type_correct"] = round(sum(bool(r["conflict_type_ok"]) for r in conf) / max(len(conf), 1), 3)
    ok = [r for r in comp if r["operational_intent_ok"] is not None]
    out["operational_intent_preserved_on_complete"] = round(sum(bool(r["operational_intent_ok"]) for r in ok) / max(len(ok), 1), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--missions")
    ap.add_argument("--provider")
    ap.add_argument("--model")
    ap.add_argument("--out", default="outputs/nl_eval")
    a = ap.parse_args()
    mdef, ms = load_collection(a.missions)
    if a.provider:
        import yaml
        from src.planner.llm import make_client
        cfg = yaml.safe_load(open("config/default.yaml"))
        cfg.update(provider=a.provider, model=a.model)
        ex = LLMExtractor(make_client(cfg), mdef.site_names(), mdef.zone_names())
        name = f"llm_{a.provider}_{a.model}".replace("/", "_")
    else:
        ex, name = RuleExtractor(mdef.site_names(), mdef.zone_names()), "rule_baseline"
    rows = evaluate(ex, mdef, ms)
    s = summarize(rows)
    os.makedirs(a.out, exist_ok=True)
    json.dump({"extractor": name, "collection_provenance": "see header of the missions file; the bundled seed set is development-team written (NOT external operators)", "summary": s, "rows": rows},
              open(os.path.join(a.out, f"nl_eval_{name}.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"nl_eval_{name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()

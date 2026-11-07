# Natural-language understanding (v10)

* Collection: `data/human_missions/missions.yaml` — 22 instructions + a fictional map (sites, zones). **Provenance: written by the development team, not collected from operators.** The file format is documented in its header;
  add real operator instructions in the same format and run `python scripts/nl_eval.py --missions yourfile.yaml`.
* Extractors (`src/nl/extraction.py`): `RuleExtractor` (offline baseline) and `LLMExtractor(client)` (asks for JSON incl. `missing`, `ambiguous`, `conflicts`, never to guess).
* Analysis (`src/nl/analysis.py`): essential fields (start, destination, max altitude) → questions; hedge words/unknown places → clarification requests; conflicts (empty altitude band, cyclic order, waypoint inside an excluded zone,
  distance/energy proofs from the feasibility oracle) detected deterministically.
* Pipeline (`src/nl/pipeline.py`): `understand(text, extractor, map, ask=callback)` asks the operator instead of guessing, folds the answers in, re-analyses.
* Intent preservation (`src/nl/intent.py`): field-level precision/recall, plus an operational check — a verified plan for the *extracted* mission is verified against the *true* mission.
* Evaluation: `python scripts/nl_eval.py` (rule baseline, offline) or `--provider/--model` (LLM).

Rule baseline on the bundled set: exact intent on complete instructions 100%, false clarification 0%, clarification recall on incomplete/ambiguous 100% (right fields asked: 100%), conflict detection 100%.
**These numbers are NOT a result:** the rules were developed while looking at these 22 instructions (no held-out set), so they measure the pipeline, not language understanding. The LLM extractor has not been run (no API key here).

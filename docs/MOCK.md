# Mock mode (src/planner/mock_llm.py)
The mock is a seeded heuristic program, NOT a language model. Its error rates and its probability of correctly applying each kind of
feedback are ASSUMPTIONS in config/default.yaml (`mock:`). In particular it is *built* to fix evidence-level feedback more reliably than
localized/binary feedback, so mock results trivially "favour" richer feedback. They validate the pipeline (data flow, metrics, statistics, plots), 
**not any scientific hypothesis**. All mock records carry is_mock=true, run ids start with `mock_`, manifests carry a WARNING, figures are titled "MOCK RUN".
ScriptedLLM (src/planner/scripted.py) replays fixed replies for tests and the 10 demo cases and is likewise not LLM output.

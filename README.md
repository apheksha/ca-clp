# CA-CLP: Constraint-Aware Closed-Loop Planning for UAV missions with LLMs

Code, benchmarks and results for the study *"What Does the Language Model Contribute? A Controlled Study of Verification-Guided Repair for
Constrained UAV Mission Plans"* (Ramaiah Institute of Technology, Bengaluru).

A language model turns a mission brief into UAV waypoints; a deterministic verifier checks every plan against the altitude band, restricted
zones, visiting order and distance/energy budgets, and returns structured feedback for repair. Only the verifier can release a plan.
The repository contains the loop, the verifier, a feasibility oracle, classical planners, model-free controls, a flight simulator, three
mission sets and every script that produces the numbers in the paper.

```
mission text + spec ─► LLM planner ─► verifier ──valid──► release ─► flight simulator
                          ▲              │ invalid
                          └── repair ◄── feedback (binary / localized / evidence / prescriptive)
```

## Main findings
* The verifier gate removed every unsafe release (98% of unverified first plans crossed a zone).
* Evidence feedback raised success on the airport-anchored missions from 1.9% to 50.9%, verifier-computed detours to 100% — but applying
  the same detours **without** the model also reached 100%, for both models tested (Gemini 3.5 Flash-Lite, gpt-oss-120b).
* The model added success only where the verifier's fix was missing or insufficient (budget cuts, terrain clearance, some enlarged
  zones); a classical planner solved all of these cases.
* Given only the mission text, both models produced specifications that led to valid plans for all 165 missions: the model is most
  useful as the language interface to a verified planner.
* In simulation, verified plans still needed altitude margins, leg clearance and turn-radius limits to be flyable.

## Install
Python 3.10+.
```
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
For real model runs create a file `.env` with a free key, e.g. `GEMINI_API_KEY=...` (https://aistudio.google.com/apikey) or
`GROQ_API_KEY=...` (https://console.groq.com). Never commit `.env`.

## Quick start
```
python -m unittest discover -s tests -t .     # 220 tests, no key needed
streamlit run app.py                          # interactive demo (mock planner works without a key)
python run.py demo                            # 10 scripted demo cases with plots
```

## Reproduce the paper
The raw model runs (~480 MB) are archived on Zenodo (see *Data availability*); unzip them into `outputs/raw/`. Then
```
python scripts/make_paper_numbers.py          # every number and table of the paper -> paper/generated/
python scripts/check_paper_numbers.py         # recomputes 149 numbers independently from the raw files
```
New model runs (free tiers, resumable, waits for quota resets):
```
python run_free.py --dry-run --quick          # offline check with stand-in models
python run_free.py                            # temperature check, text-only specs, mid-flight re-runs, Bengaluru 3D, second model
python run.py experiment --provider gemini --model gemini-3.5-flash-lite --scenario-dir benchmark/real      # one static study
python run.py dynamic   --provider gemini --model gemini-3.5-flash-lite --exp-config config/experiments_dynamic_rerun.yaml
```
Model-free analyses (no key): `scripts/oracle_validation.py`, `scripts/geometry_test_report.py`, `scripts/energy_sensitivity.py`,
`scripts/vehicle_energy_sensitivity.py`, `scripts/execution_clearance_study.py`, `scripts/flight_dynamics_study.py`,
`scripts/delay_study.py`, `scripts/estimation_delay_check.py`; results are in `outputs/review_v12/`.

## Mission sets (`benchmark/`)
| Folder | Missions | What it is |
|---|---|---|
| `real/` | 53 | real airport and heliport coordinates from OurAirports in nine regions; zones, altitude bands and budgets are **modeled** (not authoritative airspace) |
| `generated/` | 118 | synthetic missions: easy/medium/hard, corridor, 14 constraint-combination sets, 6 infeasible |
| `bengaluru/` | 30 | 3D urban missions: OpenStreetMap buildings with recorded heights as obstacles, SRTM terrain (`data/geodata_bengaluru/`, built by `scripts/build_bengaluru_set.py`) |
| `pilot/`, `reduced/` | 10, 34 | small sets used by the tests |

## Conditions
A: LLM only (no verifier) · B: verify and release · D: compute-matched resampling · C_full: evidence feedback, targeted repair, regression
guard, feasibility precheck · C_prescriptive: verifier-computed detours · X_binary, X_binary_targ, X_localized, X_evid_regen, X_no_guard,
X_one_round, X_no_precheck: one factor changed each · G1–G3: model-free controls that apply the verifier's fix · E: visibility-graph planner
· F: grid A*. Definitions: `config/experiments.yaml`.

## Repository layout
```
src/verifier/     deterministic verifier (zones as closed sets, every leg tested)      src/benchmark/   generator, feasibility oracle (sound, incomplete)
src/planner/      prompts, LLM clients (Gemini, Groq, OpenAI-compatible, Ollama, mock)  src/feedback/    the four feedback levels
src/repair/       repair loop, regression guard, model-free direct patch (G1-G3)       src/dynamic/     mid-flight changes, re-planning
src/simulator/    point-mass flight simulator, wind, sensors, turn limits, PX4 export  src/geodata/     OSM, SRTM, airspace loaders
src/nl/           text -> specification extraction                                     src/evaluation/  metrics, statistics, reports
scripts/          analyses and paper builders     tests/  unit and regression tests     outputs/  analysis results (raw runs: Zenodo)
```
Further documentation: `docs/ORACLE_CORRECTNESS.md`, `docs/VERIFIER_RULES.md`, `docs/SIMULATION.md`, `docs/SAFETY_CLAIMS.md`,
`docs/LIMITATIONS.md`, `docs/GEODATA.md`, `docs/NL_UNDERSTANDING.md`, `docs/PX4_GAZEBO.md`.

## Scope and limitations
Claims stop at geometric feasibility under modeled constraints (verifier) and execution in a point-mass simulator; no PX4 software-in-the-loop
or real flight was run, and nothing here certifies flight safety or regulatory compliance. Zones in the airport set are modeled circles,
energy is a route-cost proxy (a physical model is used in the simulator), and mission texts are templated. See `docs/LIMITATIONS.md`.

## Data availability and licences
Raw model runs and an archived copy of this repository: Zenodo, DOI to be added. Code: MIT licence (`LICENSE`).
Data keep their own licences: OurAirports (public domain), SRTM (public domain), OpenStreetMap buildings © OpenStreetMap contributors,
ODbL 1.0 (`data/geodata_bengaluru/README.md`).

## Citation
If you use this code or the mission sets, please cite the paper (reference will be added after publication).
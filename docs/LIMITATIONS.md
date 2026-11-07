# Limitations audit (what was fixed, what remains)

## Fixed in this version
| Problem | Fix | Evidence |
|---|---|---|
| HTTP 429 aborted or poisoned runs; long runs printed nothing | Retry-After / "retry in Ns" honoured, exponential backoff with jitter, per-provider requests-per-minute pacing, daily-quota detection (fails fast), visible wait messages, progress line per scenario with ETA, `--estimate`, `--max-calls`, `--resume`, Ctrl-C saves | tests/test_robustness.py (local HTTP server returning 429 and quota errors) |
| Provider errors could be recorded as "model FAILURE" | Provider/transport errors now propagate and stop the run; they are never stored as outcomes | `test_quota_stop_saves_progress_without_fake_failures_then_resume` |
| Only synthetic data | Airport-anchored pipeline from the OurAirports open dataset (zones are MODELLED radii, not regulatory airspace) (real airport/heliport coordinates and names, 12 metro regions) + extra synthetic families (corridor with narrow gap, concave zones) + 3 NL phrasing styles | tests/test_real_airspace.py (offline fixture), `python run.py real` for the true download |
| Geometry written by me, unverified | 4,500 randomized differential tests against dense sampling (circles, convex and concave polygons) | tests/test_robustness.py |
| Streamlit app never executed | app.py executed end-to-end under a stub of streamlit (found and fixed a real crash) | tests/test_app_stub.py |
| `default_transport` never exercised | exercised against a live local OpenAI-compatible server | tests/test_robustness.py |
| Seeds of one scenario treated as independent pairs | primary inference is scenario-level (exact sign-flip permutation test, scenario bootstrap CI) with Holm correction; run-level McNemar kept as secondary | report.py |
| No non-LLM reference | `E_classical` baseline (visibility-graph planner, 0 LLM calls) | loop.py |
| Hard-coded 0-100 map | scale-aware clearance, plot bounds, mock margins (meters work) | real scenarios |

## Added to match the literature survey (this version)
Battery/energy constraint + flight-time model; obstacles; severity levels and suggested fixes in feedback; single-constraint vs combined-constraint family with interaction analysis; grid A* baseline; constraint recurrence / cascade / oscillation analysis; Objective 4 dynamic changes with state preservation, failure-cause classification, runner, report and figures; UI section for mid-flight change. See OBJECTIVES_TRACEABILITY.md.

## Remaining (cannot be removed from inside this build environment)
1. **No real-LLM results exist yet.** The sandbox had no network. Every number in `outputs/` produced here is from the labelled MOCK planner. A paper needs a real run (free options: `ollama`, `gemini`, `groq`, `openrouter`).
2. **Real dataset not bundled.** It must be downloaded by `python run.py real` (needs internet). Offline tests use an invented fixture in the same CSV schema; the real file was never parsed in this build, so expect to adjust REGIONS or radii if a region yields too few sites.
3. **What is "real" in the real scenarios:** airport/heliport identities and coordinates. **Modelled:** protection radii per airport type, 120 m ceiling, distance budget, mission sites, templated NL text. Not legal advice; not real operational flight plans. No real human-written UAV mission corpus with ground-truth constraints was found that fits (UAVBench is itself LLM-generated; UAV-VLPA-nano-30 has images, not constraints).
4. **Model fidelity:** the energy model is a simple linear proxy (distance + k_v*climb + k_wp*waypoints), not a battery discharge model; flight time is a constant-speed estimate. Zones are static between changes; no wind, no flight dynamics (as stated in the survey, route/mission constraints only). Geometry: circles + simple polygons. Geometry is self-implemented; Shapely cross-check is recommended (see tests) but Shapely was not installable here.
5. **Prompt sensitivity:** one prompt version (`plan-v1`); the structured spec is always in the prompt, so this tests repair, not constraint *extraction* from free text.
6. **Feasibility oracle** can answer UNKNOWN; its witness is not the optimal path.
7. **Statistical power** depends on the number of scenarios (default 76 synthetic + real). See docs/PREREGISTRATION.md for the minimum-detectable-effect reasoning.
8. **Novelty is conditional.** The plain verify-feedback-repair loop is known (docs/NOVELTY.md). Publishability depends on real results, a full literature review, and venue fit. Nothing here guarantees acceptance.
9. Free-tier providers may log prompts; prompts contain only synthetic or public-airport data.
10. **Dynamic experiments start from a controlled "comfortable" valid plan** (conservative clearance, mid-band altitude), not from an LLM-produced plan, so every model starts from the identical state. An LLM-produced starting plan would couple Objective 4 to Objective 3 outcomes.
11. **Terrain:** no real terrain/building data was bundled; obstacles are synthetic. Real elevation (e.g. SRTM) or building footprints (OpenStreetMap) would be the next data source.


## Scope of the evaluation (added v9)
- The verifier checks **geometric/resource feasibility under modelled constraints**. It does not establish flight safety, regulatory compliance, or executability by a vehicle. `src/verifier/dynamics.py` adds necessary dynamics screens; PX4 SITL/Gazebo execution is NOT performed here (`src/simulator/export_px4.py` exports a `.plan`).
- Energy is a linear cost proxy (units of route length), not a battery model; see `scripts/energy_sensitivity.py`.
- Oracle: FEASIBLE is sound by construction; INFEASIBLE only from five necessary conditions; otherwise UNKNOWN (`scripts/oracle_validation.py` reports soundness and the measured incompleteness).
- Zero invalid releases is not mission success: report completed / unsafe-release / withheld separately (`completion_rate`, `unsafe_release_rate`, `withheld_rate`).
- `C_prescriptive` success partly reflects verifier computation; `G_direct_patch` (no LLM after the initial plan) quantifies how much.
- Mid-flight changes are injected, not sensed; `detection_delays` models late knowledge only.


## Status after v10 (what is still NOT done)
- **PX4 SITL / Gazebo have not been run**: only the bridge, mission export and world generator exist. The built-in simulator is a point-mass model with representative, uncalibrated vehicle parameters.
- **Events are sensed in simulation** by statistical sensor models; there is no real sensor pipeline.
- **No authoritative airspace / DEM / OSM data is bundled**; loaders are tested on synthetic fixtures only.
- **NL evaluation data are team-written**; the LLM extractor and the clarification dialogue have not been evaluated with real operators or a real model.
- **Physical energy** is uncalibrated and ignores acceleration phases; **oracle** is sound on all tested missions but incomplete (UNKNOWN 11.7 % on the stress set) and its physical-energy bound is numerical.
- Model registry prices/capabilities are placeholders until verified.

# Safety-claim levels (use these words in the paper and in talks)

| Level | Meaning | How it is produced here | Allowed wording |
|---|---|---|---|
| **L1 Geometric route feasibility** | the plan satisfies the modeled constraints | deterministic verifier (+ oracle verdicts); `python run.py evaluate` | "verified plans are geometrically feasible under the modeled constraints" |
| **L2 Simulated execution feasibility** | a simulated vehicle flying the verified plan arrives, completes the mission, stays out of zones, keeps battery | `python run.py simulate <raw_dir>` (built-in simulator; `docs/SIMULATION.md`) | "in simulation (uncalibrated <profile>, wind …, gust …, N seeds) X % of verified plans were executed without intrusion" |
| **L3 Actual UAV flight safety** | safe in real airspace and conditions | not assessed | **never claimed** |

Every flight record carries `L1_geometric_verified`, `L2_executed_feasible` and `L3_flight_safety = "not assessed"`. `claim_gap.json` counts plans that are L1-verified but not L2-executable and why. `claim_levels.md` is written next to every evaluation.

## Measurable outcomes of the simulation study (`src/simulator/study.py`)
planned vs actual (mean/RMS/p95/max 3D, horizontal and vertical deviation from the planned polyline; time vs planned time) · cross-track error (mean/RMS/p95/max) · mission completion (arrived ∧ all required waypoints ∧ order ∧ no intrusion)
· energy (used, predicted by the physical model, error %, remaining battery, depletion) · obstacle intrusions (zones entered, first time, time inside, max penetration depth, collision) · trajectories as CSV, planned-vs-actual plots as PNG
· recovery (`python run.py online`: detection delay, plan-activation latency, recovery rate on affected episodes with Wilson CI, extra distance/duration/energy vs the event-free flight, intrusions).

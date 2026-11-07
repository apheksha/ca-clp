# Flight simulation, vehicle physics and online events (v10)

| Piece | File | Status |
|---|---|---|
| Vehicle profiles (mass, rotor area / L/D, efficiency, battery, speeds, limits, GNSS noise) | `config/vehicles.yaml`, `src/models/vehicle.py` | 3 REPRESENTATIVE classes, **not calibrated** to any airframe |
| Physical energy model (Wh): momentum-theory induced power, parasitic drag, climb, payload, steady wind, hover per waypoint | `src/models/energy.py`, verifier (`energy_model: physical`) | implemented, model-level validation only |
| Battery state-of-charge during flight, remaining-battery estimation | `src/simulator/sim.py`, `energy.remaining_battery_wh` | implemented |
| Wind (mean + OU gusts + shear), GNSS localization (bias + noise + dropouts) + alpha-beta estimator | `src/simulator/wind.py`, `sensors.py` | implemented (statistical models) |
| Autopilot-like tracking (lookahead path following, velocity lag, accel limit, corner speed scheduling) | `src/simulator/sim.py` | simplified point-mass; **not** PX4's controllers |
| Trajectory execution + tracking-error + continuous 3D zone/terrain/altitude checks on the FLOWN path | `src/simulator/evaluation.py` | implemented |
| Batch: fly all released plans of a run | `python run.py simulate <raw_dir>` | implemented |
| Obstacle-detection range/latency, broadcast (NOTAM-like) events | `src/simulator/sensors.py::ZoneEventSensor`, `src/dynamic/detection.py` | implemented (simulated sensors) |
| Online loop: detect → update map → re-plan from the ESTIMATED state → activate after latency → continue | `src/dynamic/online.py`, `python run.py online` | implemented |
| PX4 SITL + Gazebo | `src/simulator/px4_sitl.py`, `gazebo_world.py`, `export_px4.py`, `python run.py px4`, `docs/PX4_GAZEBO.md` | **bridge code written and unit-tested for its pure parts; NOT run against a live PX4/Gazebo** (not available here) |

## What the simulator already shows (representative, uncalibrated; synthetic benchmark scaled to 20 m per map unit, `generic_quad_large`, gusts 0.5 m/s)
* Plans that the verifier accepts are not necessarily flown without incident: in the event-free baseline of `python run.py online --n 10`, 70 % of the verified reference plans were flown without
  zone contact / missed waypoint (the rest grazed zones: shortest-path witnesses hug zone boundaries with ~10 m clearance and the vehicle cuts corners). Execution-robust planning needs extra clearance
  (`classical_replanner(clearance_m=...)`).
* The verifier's physical energy model ignores acceleration/stop-and-go phases; on short legs the simulated consumption exceeds the prediction by a large factor (`energy_model_error_pct` in `evaluate_execution`),
  on long cruise legs the gap shrinks. Use it as a planning proxy, not as a flight-time guarantee.
* Event handling (`python run.py online`): re-planning raises mission success over keeping the stale plan (0 % → 26–47 % in the 10-scenario demo), but latency, detection range and thin clearances limit it. These
  numbers are a **demonstration of the machinery on a small sample**, not a result to cite.

## Not modelled
Attitude dynamics, motor/ESC dynamics, real EKF, controller saturation, vertical wind, battery voltage sag / temperature, sensor noise models for cameras/lidar (detection is a range+delay abstraction), 3D collision
geometry of the vehicle (a sphere of `collision_margin_m` is used), moving obstacles, airspace deconfliction.

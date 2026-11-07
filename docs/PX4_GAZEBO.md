# PX4 SITL / Gazebo integration (bridge; not run in this repository's environment)

1. Start PX4 SITL with Gazebo, e.g. `docker run --rm -it -p 14540:14540/udp jonasvautherin/px4-gazebo-headless` (or build PX4: `make px4_sitl gz_x500`).
2. `pip install pymavlink`.
3. Export the world and mission, then (optionally) fly it:
   ```
   python run.py px4 --scenario benchmark/generated/easy_000.json --plan my_plan.json --origin 47.397742 8.545594 --unit-m 20 --dry-run   # writes outputs/px4/mission_world.sdf + mission.plan
   python run.py px4 --scenario ... --plan ... --origin 47.397742 8.545594 --unit-m 20                                                  # uploads, arms, AUTO.MISSION, logs, evaluates
   ```
4. The flight log is converted by `log_to_trajectory` and evaluated by the same `evaluate_execution` as the built-in simulator (zone intrusions on the flown path, tracking, budgets).
`mission_world.sdf` contains one translucent prism per zone (vertical extent respected). Terrain is not exported (needs a heightmap image). Load the world in Gazebo and spawn the vehicle at START.

Known gaps: the MAVLink handshake/mode switch is written from the protocol specification and has not been exercised here; PX4 parameters (speed, acceptance radius), geofence upload and failsafes are left at defaults.

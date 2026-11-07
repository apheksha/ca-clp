# Constraint rules (deterministic verifier, src/verifier/verifier.py)
- Altitude: every waypoint and the destination must satisfy min <= altitude <= max (limits inclusive, tol 1e-9). START is ground-level and not checked.
- No-fly zones are CLOSED sets (circle or polygon). Boundary contact is a violation:
  - NFZ_WAYPOINT: waypoint/destination inside or on the boundary.
  - NFZ_SEGMENT: a route segment (START->wp1 ... wpN->DEST) intersects or touches a zone. Reported only when neither plan endpoint of that segment is already inside that zone (avoids double-counting), so "two legal endpoints, illegal route" is its own violation.
- Distance: horizontal Euclidean length START->waypoints (in `order`)->DEST must be <= max_distance (inclusive).
- Ordering: for each pair [before, after], `before` must precede `after` among plan waypoints (only when both are present; absence is a preservation violation).
- Mission preservation (reported separately from validity): required waypoints present at their coordinates (tolerance `position_tolerance`), destination unchanged, mission id matches.
- `valid` = no constraint violations. `success` = valid AND mission_preserved. Only the verifier sets these.
Geometry: own implementation (src/utils/geometry.py); Shapely was unavailable offline. Open-boundary semantics are not implemented.


## Formal definitions (v9)
Plan P = (p_0=START, p_1..p_n, p_{n+1}=DEST), p_i = (x_i, y_i, h_i); h_0 = start_altitude. Legs s_i = [p_i, p_{i+1}], horizontal length ℓ_i = ‖xy_{i+1} − xy_i‖.
Zone Z_j is a closed prism D_j × [z_min_j, z_max_j] (D_j a closed disc or polygon; absent bounds = ±∞, i.e. the original full-height 2D zone).
Altitude along a leg is linear in the horizontal parameter t∈[0,1]: h(t)=h_i+t(h_{i+1}−h_i).
- Altitude: h_min ≤ h_i ≤ h_max for i=1..n+1 (if family active; TOL = 1e-9 absolute).
- Zone (waypoints): ∄ i∈1..n+1: xy_i ∈ D_j ∧ h_i ∈ [z_min_j, z_max_j]. Zone (legs): ∄ i, t: xy(t) ∈ D_j ∧ h(t) ∈ [z_min_j, z_max_j] (also when both end points are legal). Boundary counts as inside.
- Distance: L(P) = Σ ℓ_i ≤ L_max.   Energy: E(P) = L(P) + k_v·A(P) + k_wp·n ≤ E_max, A(P) = Σ max(0, h_{i+1} − h_i); units of E = route-length units; k_v = length-units per metre climb; k_wp = length-units per waypoint (modelling choice).
- Ordering: for each pair (a,b): index(a) < index(b) in P. Preservation: every required waypoint present within position_tolerance of its coordinates, destination within tolerance, mission_id equal.
- Release: released(P) ⇔ all active families satisfied ∧ preservation. Run outcome ∈ {completed (released ∧ success), unsafe release (released ∧ ¬success), withheld (¬released)}.
Feasible set F = {P : all of the above}. The oracle (`benchmark/feasibility.py`) decides: FEASIBLE ⇒ ∃P∈F (verified witness); INFEASIBLE ⇒ F = ∅ by one of five necessary conditions (listed in the module docstring, tolerance-aware); UNKNOWN otherwise.


## v10 additions
- **Physical energy** (`energy_model: physical`): E in Wh = Σ_legs P(v_air, climb rate, payload) · t_leg + hover; ground speed from steady wind (a leg with |w⊥| ≥ v_cruise is not flyable → energy violation); budget `max_energy` in Wh (usable battery = capacity × usable_fraction). See `src/models/energy.py`.
- **Terrain clearance** (when `terrain` is present): for every flight segment sampled at ≤ ½ cell, `min_agl ≤ h(t) − ground(x(t), y(t)) ≤ max_agl`; the region within `terrain_exempt_m` of START and DEST is exempt from `min_agl` (take-off/landing). Violations: `TERRAIN_CLEARANCE`, `ABOVE_MAX_AGL`.
- Mission fields: `unit_m`, `energy_model`, `vehicle`, `payload_kg`, `wind`, `terrain`, `min_agl`, `max_agl`, `terrain_exempt_m` (all optional; absent ⇒ v9 behaviour).

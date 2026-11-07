# Feasibility oracle: supported problem class and correctness argument (v10)

Code: `src/benchmark/feasibility.py`. Empirical validation: `python scripts/oracle_validation.py` (418 missions in the last run: 118 stored + 300 random stress missions with random
altitude/ordering/distance/energy/zone/terrain combinations, families randomly switched off, proxy and physical energy). Result: 0 soundness violations (S1, S2, S3), UNKNOWN rate 11.7 %.

## 1. Problem class
A mission M = (start s, destination d, required points R, altitude band [h_lo, h_hi], limits L_max / E_max, precedence pairs, zones Z_j, position tolerance τ, optional terrain,
vehicle/wind/payload for the physical energy model). Zones are closed prisms D_j × [z_min_j, z_max_j] (D_j a disc or simple polygon). A plan P is *feasible* iff the verifier
(`verify`) accepts it (`docs/VERIFIER_RULES.md`, "Formal definitions"). F(M) denotes the set of feasible plans.

## 2. Guarantees
**Theorem 1 (FEASIBLE is sound).** If `assess(M)` returns FEASIBLE then F(M) ≠ ∅.
*Proof.* The returned witness plan is passed through `verify` before the verdict is issued; F(M) is defined by `verify`. ∎ (No optimality is claimed.)

**Theorem 2 (INFEASIBLE is sound).** If `assess(M)` returns INFEASIBLE then F(M) = ∅, assuming the lemmas below. Each INFEASIBLE verdict names the failed condition (`proof_kind`):
1. `empty_altitude_band`: altitude family active and h_lo > h_hi. Every plan has at least one waypoint (the destination) whose altitude must lie in the band.
2. `cyclic_ordering`: ordering family active and the precedence relation contains a cycle; no linear order exists.
3. `point_in_zone`: the start (fixed) or a required point / the destination lies in a zone that covers the whole altitude band, and a displacement of at most τ cannot leave the zone
   (penetration depth > τ; start: no tolerance). The verifier requires the point within τ of its coordinates, so every candidate position is inside the zone.
4. `distance_lower_bound`: the shortest precedence-feasible polyline through the required points (minimised over all admissible orders) minus τ·(2|R|+1) exceeds L_max. Each of the
   2|R|+1 leg endpoints that may move shortens the polyline by at most τ; the straight chain is a lower bound of every route through the points.
5. `energy_lower_bound` (proxy): the bound of 4 plus k_v·(h_lo − h_start)⁺ + k_wp·|R| exceeds E_max (ascent ≥ the climb to the band; ≥ |R| waypoints).
   (physical): see lemma P.
6. `routed_distance_lower_bound` / `routed_energy_lower_bound`: as 4/5 but with the length replaced by the shortest path *around the obstacles* (lemma R); only tried when no witness exists.

**Lemma R (routed lower bound).** Let Q_j be a polygon contained in a zone D_j that covers the whole band (inscribed 24-gon of a disc; the polygon itself if convex). Any path that avoids the
closed zones avoids the open interior of every Q_j, so its length is ≥ the shortest path among the open polygons Q_j (shortest paths among polygonal obstacles bend only at obstacle vertices;
the visibility graph over {key points} ∪ {vertices of Q_j} contains such a path). Q_j is shrunk by 10⁻⁹ about its centroid so that edges between its own vertices are admissible: this admits *more* paths
than the exact problem, hence can only lower the bound (sound direction). Non-convex polygons and zones that can be overflown are omitted (fewer obstacles ⇒ lower bound). The minimum over all admissible visiting orders is
taken by enumeration (|R| ≤ 6).

**Lemma P (physical energy bound).** For any airspeed policy the energy of a leg of ground length d is ≥ d·min_v P_level(v)/(v+|w|) (ground speed ≤ airspeed + |wind|, time ≥ d/(v+|w|), level power ≥ minimum
over v) plus m g Δh⁺/η for climb (independent of speed in the model), avionics included in P_level; hover energy is dropped (≥ 0). The minimum over v is taken on a 0.1 m/s grid with a 0.2 % safety factor.
**This minimisation is numerical, not interval-rigorous**: the claim is "sound up to grid resolution and the 0.2 % margin".

## 3. What is *not* guaranteed (and how it is reported)
* **Completeness.** UNKNOWN means "no witness and no proof"; it is not "infeasible". Measured: 11.7 % UNKNOWN on the stress set; for a sample of those an independent solver finds solutions, i.e. the
  oracle is incomplete there. Reasons: constant-altitude or terrain-following witness only; fixed visiting order for the witness; polygon inflation by vertex push-out (heuristic for concave zones).
* **Optimality / approximation error of circles.** Witness: circumscribed 12-gon, wrapped length overestimates by ≤ tan(π/12)/(π/12) − 1 = 2.35 % (v12 correction: earlier versions said 1.15 %, the value for a 17-gon). Lower bound: inscribed 24-gon, underestimates by ≤ 1 − sin(π/24)/(π/24) = 0.29 %.
  `assess(M, certify=True)` returns the certified bracket `route_bracket = (LB, UB)` of the true shortest chain length. Measured on the stress set: median relative gap 0.74 %; the maximum (86 %) occurs when zones that can be
  overflown, or non-convex zones, are excluded from the lower bound (the bracket is then valid but loose).
* **Numerics.** float64 + absolute tolerance 1e-9 (conservative: a grazing contact counts as intersecting). `src/utils/exact_geometry.py` decides the same predicates in exact rational arithmetic; the float verifier never
  missed an exact intersection in 3 000 + 1 000 randomised and tolerance-adjacent cases (`geometry_selftest.exact_crosscheck`); it reports 89 intersections that exist only within the tolerance (conservative).
* **Interaction coverage.** Altitude, ordering, distance, energy (proxy/physical), zones (2D/3D) and terrain are exercised jointly by the random stress missions in `scripts/oracle_validation.py` (table `by_active_family_set`).
  This is evidence, not a proof of interaction soundness beyond the lemmas above.

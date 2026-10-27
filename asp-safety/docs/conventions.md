# Conventions — ASP Safety Layer (DC-2)

Single reference for the units and contracts every rule, builder,
and test assumes. AGENTS.md Subdivision 3's original float-based
schema is superseded by this + `asp_traffic_rules.md`.

## Units and frames

- Positions are meters in code, **integer centimeters (x100)** in ASP.
  `checker.SCALE_FACTOR = 100`. All ASP arithmetic is integer-only:
  no floats, no `sqrt()`; squared distances replace Euclidean ones.
- Timesteps `T` are `0..5` (6 total). `checker.STEP_DT = 1.0 s` is the
  canonical step every speed/gap threshold is derived for (pinned by
  `test_step_dt_contract`). Model-native 0.5 s outputs MUST be
  resampled before `check()`; calibration and `pipeline.run()` both
  anchor T=0 at the ego pose and stamp `horizon_s` (3 s pred, blind
  beyond).
- Trajectories are ego-frame `(T, 2)`, x = forward, y = left.
  Map layers are global in nuScenes data and converted to ego frame
  by the loader (`_map_to_ego_frame`) before checking.
- Clingo integers are 32-bit: keep every product below ~2e9
  (decimeter units, sign-split abs forms, banded lookups).

## Speed semantics (GAP-3/GAP-9)

- `speed(V,T,S)` / `speed2(V,T,S)` / `velocity_sq(V,T,S)` are SQUARED
  speeds in (cm/step)^2. Limit/cap facts (`speed_limit`,
  `zone_speed_limit`, `vru_speed_cap`, `min_speed_limit`,
  `confidence_speed_cap`) stay LINEAR (cm/step); rule bodies square
  them at comparison (`S2 > Limit*Limit`).
- 50 km/h default cap = 1389 cm/step, squared = 1929321. It applies
  ONLY to bare trajectories (no mapped lanes); mapped-but-unposted
  roads stay silent (N-1).

## Fact naming

- `pos(V,T,X,Y)` + `vehicle(V)` are the only required inputs.
  `point/4` aliases `pos/4`.
- `gap_dist` = Euclidean total separation (cm). True along-track is
  `longitudinal_gap` (signed, + = ahead). `following_gap` mirrors
  `gap_dist` from the follower perspective. `closing_rate` =
  instantaneous velocity-projection closing speed (cm/step, >= 0).
- `about_to_enter` = approach band + motion toward center
  (`approaching_zone`) or holding at the stop line. `approaching/3`
  with a vehicle second arg is vehicle-vehicle convergence (R6.4);
  with a numeric second arg, a merge endpoint (R15.3).
- `stopped_before` = stopped NEAR the feature (not anywhere);
  `yielding_maneuver` = stopped anywhere (intentional, R6.4).
- `stopped(V,0)` holds via zero heading (velocity needs T-1).
- Agent vehicles are `vehicle_{i}` in array order. `emergency_vehicle`
  facts must reuse those ids. Planner boxes are `[x,y,z,w,l,h,yaw]`
  with planar `[vx,vy,vz]` kept separately as `velocities`.
- Internal arbitration atoms (`ttc_band`, `threat_level`,
  `comfort_score`, `future_violation`, `any_violation`) surface only
  via `check(include_internal=True)` and never count toward unsafe.

## Correction contract

- `TrajectoryCorrector` owns all strategies (single dispatch table in
  `correction_strategies.py`). `checker.get_safe_trajectory`
  delegates; safe input returns the IDENTICAL object.
- Fixpoint: per-timestep batching with dirty-set guards, per-group
  caps (`group_cap`), stagnation break (3 identical sets), then
  `RuntimeError` ('raise') or flagged `emergency_hold_uncorrectable`
  ('brake').

## Constraint atoms (P1, 2026-10-03)

- Emitted alongside violation/hazard heads, collected only via
  `check(include_constraints=True)`; `#show` block lives in
  `helpers.lp`. Units are cm / cm-step integers like all ASP facts.
- `constraint_v_max(V,T,Limit)` / `constraint_v_min(V,T,Limit)`:
  linear cm/step caps/floors (rules square at comparison).
- `constraint_x_bounds(V,T,Min,Max)` /
  `constraint_y_bounds(V,T,Min,Max)`: allowed-region boxes in cm.
- `constraint_a_max(V,T,Limit)`: cm/step^2 accel bound (comfort).
- `constraint_lane_center(V,T,Y)`: preferred lateral position, cm.
- Python shape: `{'v_max': [(t, limit), ...], 'v_min': [...],
  'x_bounds': [(t, min, max), ...], 'y_bounds': [...],
  'a_max': [...], 'lane_center': [(t, y), ...]}` (see
  `TrajectoryChecker._group_constraints`). Rules without a natural
  bound document their exemption in-file ('documented exemption').

## QP solver contract (P3/P4, 2026-10-03)

- `ConstraintOptimizer.solve(traj_m, constraints_dict)` -> (corrected
  meters, info {status, iterations, solve_time_ms, pinned_t0}).
  Pure NumPy; defaults lr 0.2 / 50 steps / 1 cm tol / 0.05 lane pull.
- T=0 (ego pose) is never moved. Overlapping bounds: tightest box
  and minimum cap win. v_min skips directionless (stopped) segments.
- `CorrectorQP.correct()` returns the heuristic 5-key applied-entry
  schema with `rule='qp_optimization'` (plus fallback entries when
  the heuristic finishes the fix). Safe input returns a copy and [].

## Corrector selection (2026-10-03)

- `SafetyLayerPipeline(corrector_mode='heuristic'|'qp')`, default
  heuristic. `run()` stamps `corrector_mode`; error dicts carry it
  too (schema parity). `run_calibration.py --corrector` mirrors the
  flag and stamps `corrector_mode` in the JSON summary.
- P6-lite reading: on frozen cold preds QP matches the heuristic fix
  rate with lower mean latency, but both fail the same re-route
  cases -- constrained nudging is not re-routing. Closed-loop and
  warm-data numbers stay open.

## Bound families (P6 ablation, 2026-10-03)

- `CorrectorQP(bounds='full'|'caps'|'boxes')`. caps = limit-type
  bounds (v_max/v_min/a_max); boxes = region-type bounds
  (x_bounds/y_bounds plus the lane_center preference). Filtering is
  applied to the initial constraints AND every rounds-loop
  re-check. Subset runs disable heuristic fallback so fix rates
  measure the family, not the fallback.

## Hold-back boxes (S17 fix, 2026-10-03)

- Following/longitudinal/platoon bounds are ONE-SIDED: an ahead
  threat caps ego at (threat - gap) with the far side open at
  +-1000000 cm; a behind threat mirrors. Centered boxes are
  forbidden for gap rules (they attract closest-point optimizers).
  Side comes from the signed gap (R4.2) or X2>=X in ego-forward
  frame (R3.2/R15.2).
- Known limit: static boxes are per-timestep; against moving
  threats with tail coupling, rounds do not converge (scene 8).
  Horizon-joint solving is future work; do not raise max_rounds
  to compensate.

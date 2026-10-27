# Test Coverage — ASP Safety Layer

Generated 2026-09-26 from `experiments/coverage_probe.py` (73/73),
refreshed 2026-10-21 (test-count refresh, S13-S17 suites).
The P-9 registry test (`test_rule_registry`) fails the suite if any
catalog `violation/`/`hazard`/`warning` head lacks a quoted firing
assertion under `tests/`.

## Headline

- 120 tests green (collected 2026-10-21): 104 fast (Clingo, no GPU)
  + 8 sparse_drive + 7 pipeline + 1 `test_all_components()` (RQ-6
  pytest entry). S13-S17 contributed 26: `test_constraint_emission.py`
  (12), `test_qp_feasibility.py` (13, incl. bound-family + one-sided
  pins), `test_pipeline_corrector_modes` (1, in pipeline file).
- 73/73 catalog findings fire on demand; every family has firing +
  quiet pairs except where noted.
- Solver-internal atoms (`ttc_band`, `threat_level`,
  `comfort_score`, `future_violation`, `any_violation`) are covered
  via `check(include_internal=True)` in `test_t2_coverage.py`.
- New S13-S17 coverage: constraint emission (head-coverage
  self-check, firing+emission per family, exemption pin), QP exact
  projections (v_max/a_max/box/spring), safe passthrough, speed +
  red end-to-end re-validation through real Clingo, bound-family
  caps/boxes routing, one-sided hold-back exact tuples.

## Per-family status

| Family | Firing tests | Notes |
|---|---|---|
| R1 speed | urban/hwy/zone/min/VRU/default | min-band feasibility assumed |
| R2 lanes | keeping/transition, illegal, gap, wrong-way, HOV, departure warning | transition exception tested |
| R3 following | time floor, tailgating, queue oscillation | R3.4 overflow fixed |
| R4 collision | lateral/longitudinal, bubble, cut-in, rear-end, sideswipe | R4.2 now along-track |
| R5 signals | red, yellow zone, green+obstacle, flashing red/yellow, arrow | RTOR 3 gates |
| R6 priority | uncontrolled (+tie/clear), roundabout, turn yield, emergency, crosswalk | R6.2 needed conflict-zone fix |
| R7 turns | RTOR x3, left-across, U-turn (sign + sightline), signal, radius | radius needs `min_turning_radius` extra |
| R8 overtake | solid, oncoming, zone, sightline, right-side, clearance | `overtaken_type` via extras |
| R9 VRU | unmarked corner, school guard, bike lane, blind spot, sidewalk | blind-spot stale clause via extras |
| R10 intersection | queue cut, box blocking, protected lane, right-turn ped | queue facts via extras |
| R11 emergency | brake feasibility, evasion bounds, guardrail, construction x2 | evasive/hazard via extras (no generator yet) |
| R12 comfort | accel/decel, lat accel/jerk, reversal, score | advisory-only score via internal atoms |
| R13 predictive | collisionecast, 3 TTC bands, arbitration cascade, horizon agg | cascade pinned single-label |
| R14 environment | wet, headlights, visibility (+weather variants) | caps via weather dict |
| R15 complex | platoon x2, merge, wave, multiway | merge order is a labeled heuristic |
| Checker | shape/cull/warnings/times/determinism/aliasing/duplicates/conflict-cull | edges file |
| Corrector | converge/no-op/raise/advisory/idempotent/min-boost/brake/cap/tail/dual-id/stagnation | 10 tests |
| Constraints | contract, head-coverage self-check, R1-R5/R6.5/R12/R13 firing+emission, exemption pin | emission file (S13) |
| QP corrector | exact v_max/a_max/box/spring projections, fail-loud, safe passthrough, speed+red re-validation, bounds families (full/caps/boxes), one-sided hold-back | test_qp_feasibility.py (S14-S17) |
| Loader | placeholder schema/determinism/hostile-9/reachable-red | real path via calibration |
| Metrics | formulas + yaw/validation/FDR/keys + real outputs | T-10 |
| Planner | shapes/golden/fail-loud/STEP_DT/determinism/temporal/status | GPU paths via harnesses |
| Pipeline | run/batch/red-E2E/detections/errors/raise/timing/modes | real inference (slow); `corrector_mode` pin S15 |
| Perf | bare <50 ms (best-of-5 min, two-tier breach retry; runs FIRST via tests/conftest.py cold-machine ordering) | 3 hardenings; full-suite flake fixed 2026-10-21 |

## Known gaps (not coverage debt)

- `evasive_path` has no generator (corrector holds instead) — P-6
  convergence guards shipped; a full generator remains future work
  (`evasive_path`/`hazard_distance` stay manual-extras, tested).
- Parked-vs-live distinction — answered without box classes (S10:
  tracked-series variance explains ~60% of collision hits on both
  GT and pred sides).
- Real-scene perf headroom — tile caps shipped (S10, real-scene
  checks ~100 ms); corrector latency under heavy agent load stays
  open (multi-shot deferred with reasons, S7/S8).
- Horizon-joint QP solving — open as P-23 (static per-timestep
  boxes whack-a-mole moving threats at high rounds, scene-8 class).

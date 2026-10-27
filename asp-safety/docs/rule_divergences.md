# Rule Divergences — spec (`asp_traffic_rules.md`) vs implementation (DC-4)

Status key: KEEP (deliberate, tested) / WIRE (implemented to spec) /
DROP (spec item with no implementation) / NEW (extension beyond spec).
All KEEP items have firing tests (see `test_rule_registry`).

## Resolved this month (were open P-19 items)

| # | Item | Disposition |
|---|---|---|
| 1 | R2.1 lane_transition exception (mid-change midpoint excused) | NEW, tested (`test_r2_transition`) |
| 2 | R3.4 sign-split abs form + referenced tunables (old square overflowed: severe swings never fired) | NEW (fix), tested |
| 3 | R4.2 true `longitudinal_gap` (was Euclidean `dist2`) | WIRE, tested (`test_longitudinal_clearance_adjacency`) |
| 4 | Default 50 km/h cap gated on no-mapped-lanes (was universal; FP on arterials) | NEW (fix), tested |
| 5 | `conflict_zone` includes roundabouts (R6.2 was unfireable) | NEW (fix), tested |
| 6 | `obstacle_in_box` includes pedestrians (R5.3/R7.1 clear-path) | NEW (fix), tested |
| 7 | R15.4 sign-split abs form + `SV != Avg` guard (old square overflowed; steady-lead FP) | NEW (fix), tested |
| 8 | `stopped(V,0)` via zero heading (T=0 was never "stopped") | NEW (fix), tested |
| 9 | `about_to_enter` motion-gated (`approaching_zone`) + stopline fallback for holds | NEW (fix), tested |
| 10 | `stopped_before` stop-line-relative (was stopped-anywhere) | NEW (fix), tested |
| 11 | `closing_rate` velocity-projection (was noisy first-difference) | NEW (fix), tested |
| 12 | `gap_dist` stays Euclidean; true projection is `longitudinal_gap` (matches `data_contracts.md`, which already claimed longitudinal) | KEEP, documented |
| 13 | `following_gap` = Euclidean from follower perspective (spec "relabel" complaint: it IS total separation, kept + documented) | KEEP |
| 14 | `yielding_maneuver` stopped-anywhere is INTENTIONAL (R6.4 has no stop line) | KEEP, documented in rules |
| 15 | `approaching_zone/3` (vehicle-zone) vs `approaching/3` (vehicle-vehicle, R6.4; endpoint form, R15.3): split by design, documented at both sites | KEEP |
| 16 | `occupancy`/`current_time` now built from `ego_intent` (were manual-only) | WIRE, tested (R2.4) |
| 17 | `conflict_points` + `oncoming_lanes` map keys (were manual-only) | WIRE, tested (R15.1/R7.2) |
| 18 | `merge_gap_sequence`/`mainline_vehicle` geometric heuristic (ego-in-merge-lane + along-track order) | NEW, tested, heuristic labeled |
| 19 | `rolling_avg_speed` emitted (speed2-space window avg) | WIRE, tested (R15.4) |
| 20 | `approaching(V,ego,T)` emitted (closing + <50 m); `emergency_vehicle` stays manual with the array-id contract (`vehicle_{i}`) | WIRE (half), tested (R6.4) |
| 21 | `evasive_path`/`hazard_distance` stay manual (no corrector search yet; P-6): R11.1->R11.2 handoff tested with pinned `insufficient` link | KEEP, tested |
| 22 | `crosses_oncoming_path`: no rule references it | DROP (no impl) |
| 23 | `threat_level`/`ttc_band`/`comfort_score`/`future`/`any` surfaced via `check(include_internal=True)`; corrector does not consume them yet (P-6) | WIRE (half) |

## Pre-existing, kept (with tests)

R1.5 close-only bucket; R2.2 fixed 5 m gap; R2.3 motion guard;
R2.4 `/4` restriction facts; R3.1 squared headway form (equivalent);
R3.2 band-indexed gaps (`min_gap_table/2` removed: never defined);
R3.3/R3.4 arity-3 (R13.4-compatible); R4.3 fixed 8 m; R4.4
accel-vector form; R4.5 fixed 150 cm; R5.2 banded stop needs; R5.3
`clear_path/3`; R5.5 arrow-red-only; R6.1 conservative tiebreak; R6.2
any-circulation gap model; R6.3 turn_duration+window timing; R7.4
universal signal (stronger than spec); R8.4/R4.1 decimeter internals;
R9.1 15 m radius without corner facts; R10.1 `behind/4` + fixed 600;
R10.3 maneuver vocabulary (`left_turn`/`stopped`); R11.1/R14.2 banded
stop needs (no braking table); R11.2 `evasion_in_lane` + 150 cm;
R12.1 accel-vector-squared; R12.2 kink ratios; R12.3 >90-degree only;
R12.4 counts `excess_accel` only; R14.1 squared reduced form
(equivalent); R15.1 shared-rank ties; R15.3 no endpoint-approach
guard; `speed/3` squared alias (no rename); `in_lane` axis-aligned +
`in_lane_fact` fallback; `override_active` 8-rule hand list (policy:
new Critical rules MUST extend it); `turn_crossing`/`clear_path`
masking reviewed acceptable (narrow turn-window exemptions).

## Policy for new rules

Every new `violation/`/`hazard/` head needs a firing test or
`test_rule_registry` fails the suite.

## P1 constraint emission (2026-10-03, session S13)

| # | Item | Disposition |
|---|---|---|
| 24 | Constraint atoms mirror violation bodies minus the comparison (holds = v_max 0, corridors = x/y_bounds, caps exact where derivable, comfort = a_max minus override) | NEW, tested (`test_constraint_emission`) |
| 25 | 14 rules exempt with documented reason (no natural box browsable in .lp via 'documented exemption') | KEEP, pinned (`test_exemptions_documented`) |
| 26 | checker `include_constraints` 3-tuple path; constraints never unsafe, never in violations list | NEW, tested (contract test) |
| 27 | Env clingo reads 5.8.2 vs documented 5.7.1 | KEEP (drift logged, no action) |

## P3/P4 constrained solve (2026-10-03, session S14)

| # | Item | Disposition |
|---|---|---|
| 28 | Projected-gradient MVP: joint solve over all bounds (T0 pinned, tightest-box/min-cap on overlap), re-validated symbolically | NEW, tested (`test_qp_feasibility`) |
| 29 | CorrectorQP falls back to the heuristic fixpoint (flagged head entry) instead of inventing a new failure contract | KEEP (same on_failure raise/brake semantics) |
| 30 | Overlapping same-timestep bounds intersect (may go empty on conflicting corridors, e.g. lane-change union) -- QP takes the intersection; empty intersection clips to a point and re-validation catches it | KEEP, documented limitation for P6 ablation |

## Pipeline QP mode + P6-lite (2026-10-03, session S15)

| # | Item | Disposition |
|---|---|---|
| 31 | `corrector_mode` heuristic default; QP opt-in only (no silent behavior change for existing callers) | KEEP, tested (mode test) |
| 32 | P6-lite verdict: fix parity 0.38/0.38 on frozen cold preds; QP faster on average, 10x on scene 7; identical give-up set | KEEP (evidence in `experiments/qp_ablation.json`; not a superiority claim) |
| 33 | `--corrector qp` GPU calibration not run here (no torch on this machine) | Open laptop task, flagged in whatodo |

## Bound-family ablation (2026-10-03, session S16)

| # | Item | Disposition |
|---|---|---|
| 34 | caps = limit-type (v_max/v_min/a_max), boxes = region-type (x/y_bounds + lane_center preference). Grouping is a Python filter, not an ASP distinction | KEEP, documented in conventions |
| 35 | Rounds-loop re-check initially bypassed the filter (round 2+ solved full constraints) -- fixed, covered by multi-round lane test | NEW (bug found by new test path), fixed + tested |
| 36 | Ablation verdict full 0.38 > boxes 0.12 > caps 0.00 reads as joint-solve evidence, not family ranking (n=8, cold preds) | KEEP (claim fenced in whatodo/session) |

## QP sweep + hold-back fix (2026-10-03, session S17)

| # | Item | Disposition |
|---|---|---|
| 37 | Centered threat boxes attract closest-point optimizers; one-sided hold-back gated on signed side (R4.2 gap sign, R3.2/R15.2 X2>=X) | WIRE (fix), tested (exact tuples + absence) |
| 38 | Static per-timestep boxes cannot coordinate a tail against a moving threat (scene-8 whack-a-mole at 3/5/10 rounds) -- needs horizon-joint solving, not more rounds | KEEP (limitation; max_rounds stays 3) |
| 39 | Optimizer knobs flat across 12 combos on frozen data; defaults kept per sweep precedent | KEEP (evidence `experiments/qp_sweep.json`) |

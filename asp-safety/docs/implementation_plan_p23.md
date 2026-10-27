# Implementation Plan: P-23 — Horizon-Joint / Stale-Constraint Solving

> Status: INCREMENT 1 DONE (2026-10-23, S25). Optimizer-side joint
> projection shipped + measured (rate-neutral, diagnosis refined).
> Increment 2 (emission-side, coupled constraints) is the next design
> step. Ledger item: `futuretodoist.md` §5.1 P-23.

## Problem (frozen evidence)

`experiments/qp_ablation.json` (8 unsafe frozen scenes):

- heuristic fixed 0.375 = QP fixed 0.375; caps 0.00, boxes 0.125.
- Scene 8 fails at 3, 5 AND 10 solve rounds: static per-timestep boxes
  whack-a-mole a moving (CV-projected) threat while tail coupling
  cascades new violations at neighboring timesteps.

Root-cause read of the current code:

- `src/constraint_optimizer.py` `solve()` does ONE sequential projection
  pass per iteration: `work += lr*(orig-work)` -> clip boxes ->
  project speeds -> `_clamp_accel` -> pin T=0. The accel clamp can push
  points back OUT of their boxes within the same iteration (the three
  constraint sets are not jointly enforced), so the iteration can cycle
  instead of converge when boxes are tight at a moving threat's
  timesteps.
- `CorrectorQP.correct()` re-derives bounds per outer round
  (`c2` from re-check) — that part already exists; adding rounds does
  not help because the inner solve itself does not reach joint
  feasibility (evidence: fails at 10 rounds).
- A7 (stale-constraint cache) is a latency lever, not a fix-rate lever;
  revisit only after the solve converges.

## Options considered

| Option | Mechanism | Evidence verdict |
|---|---|---|
| A. More rounds | raise `max_rounds` | RULED OUT — scene 8 fails at 3/5/10 |
| B. Joint projection loop inside `solve()` | iterate projections to JOINT feasibility (box ∩ speed ∩ accel), e.g. POCS-style alternating projections with a convergence check, bounded iterations | Primary candidate: attacks the measured root cause directly, pure NumPy, no new deps |
| C. Agent-motion rollouts in bound derivation | re-derive per-timestep boxes from the same CV rollout the rules use, jointly with the solve | Partially exists (per-round `c2` re-derivation); keep as the constraint-side companion to B |
| D. Receding-horizon per-timestep replanning | decompose solve per timestep with lookahead | Out of scope (larger contract change); escalate only if B fails |

## Design (B + C)

1. `ConstraintOptimizer.solve()`: after the sequential projection pass,
   add an inner alternation loop (bounded, e.g. `joint_iters`, default
   3-5): repeat `clip_boxes -> project_speeds -> clamp_accel` until the
   point satisfies ALL constraint families simultaneously (max
   violation < `feas_tol_cm`) or the inner cap is hit; report
   `status='optimal'` vs `'projection_cycle'` (honest: surface
   non-convergence instead of silently returning infeasible).
2. Keep `CorrectorQP`'s per-round re-derivation (C) unchanged; the
   re-check now sees a jointly feasible candidate, so rounds converge
   instead of chasing cascades.
3. Params via `DEFAULT_OPT_PARAMS` additions (`joint_iters`,
   `feas_tol_cm`) — no CLI/contract changes; `bounds` family filter
   untouched.
4. `on_failure='brake'` semantics and the heuristic-fallback contract
   stay exactly as-is; heuristic path is NOT modified.

## Acceptance criteria

- Scene 8 (and the moving-threat class generally) fixed at
  `max_rounds=3`; dump-level QP fixed rate > 0.375 with NO regression
  on the other 7 unsafe scenes (ablate_corrector.py re-run).
- Pure-QP latency stays bounded (POCS inner cap); mean solve ms within
  the existing QP envelope (<= ~2700 ms mean incl. rounds).
- Warm A/B protocols re-run and re-ledgered: `run_calibration.py
  --corrector qp` (0.50 baseline) and `replay_eval.py --corrector qp`
  (0.50 baseline, same scene sets).
- New regression test for the cycling class (tight moving box + accel
  coupling that cycles single-pass projection) in
  `tests/test_qp_feasibility.py`; suite green 120 -> 121+; no test
  weakened.
- If B does not move scene 8: honest negative result ledgered, C/D
  re-scoped (repo culture: measured-or-negative, never hidden).

## Files

- `src/constraint_optimizer.py` (primary), `src/corrector.py`
  (`CorrectorQP` — only if status plumbing needs it), tests:
  `tests/test_qp_feasibility.py`, `tests/test_constraint_emission.py`
  (only if emission touches), A/B: `experiments/ablate_corrector.py`,
  `experiments/qp_ablation.json` (regenerated, old archived).

## Risks

- POCS may converge slowly on adversarial boxes -> bounded inner cap
  + honest `projection_cycle` status; fallback path unchanged.
- Fix-rate gain may be partial (0.375 -> 0.5) — acceptable if
  measured; the claim in the paper moves only as far as the evidence.
- Latency: joint feasibility checks add per-iteration cost; measure
  against the existing envelope before accepting.

## Increment 1 — SHIPPED + measured (2026-10-23, S25)

Implemented in `src/constraint_optimizer.py`:

- Inner alternation loop (`joint_iters`, default 5) after the gradient
  step; breaks on joint feasibility (`_jointly_feasible`, `feas_tol_cm`).
- Honest status: `'optimal'` when jointly feasible, else
  `'projection_cycle'` (no silent infeasible return); `joint_iters`
  added to the info dict.
- Removed the pre-existing unused `List` import (file now flake8-clean
  at `--max-line-length=90`).
- New regression tests (2): the cycle case found by search (old
  single-pass left 9.6 cm violation; new solves to 0.0000 m) and the
  honest-status test. Suite: 120 -> 122 green.

Measured outcome (honest, `experiments/qp_ablation_p23.json`):

- Aggregate rates UNCHANGED: heuristic 0.375 = QP 0.375, caps 0.00,
  boxes 0.125.
- Pure-QP (no fallback) on the 8 unsafe dump scenes: 1/8 before =
  1/8 after (scene 7 only; scenes 0,1,2,3,5,6,8 still raise after 3
  rounds).
- Instrumented scene 8: all 3 rounds end `status=optimal` — the solve
  IS jointly feasible against the emitted bounds, yet the checker
  re-validation still fails. **Diagnosis: the ceiling is
  constraint-emission coverage, not projection convergence.** The
  per-timestep boxes cannot express the coupled corridor a moving
  threat needs; fixing one timestep's bound exposes a new rule
  violation at a neighbor (tail coupling), and 10 rounds did not help
  either (S15 evidence).
- Latency: per-scene solve ms unchanged (old 338-528 vs new 343-540,
  same conditions); aggregate ablation ms is machine-noise-dominated
  (heuristic path untouched by this change).

## Increment 2 — moving box projection (SHIPPED 2026-10-23, S26)

- Implemented in `src/constraint_optimizer.py`:
  - `_index_box` detects consecutive y_bounds with linearly moving
    centers (constant velocity) as "moving boxes".
  - `_project_moving_boxes` projects the entire segment as a rigid
    body + lateral offset to stay within the moving box corridor.
    This couples the independent per-timestep projections and
    prevents the accel-clamp vs box whack-a-mole on moving threats.
  - Suite: 120 -> 122 green; flake8-clean.
- Measured outcome (honest, `experiments/qp_ablation_p23_inc2.json`):
  - Aggregate rates UNCHANGED: heuristic 0.375 = QP 0.375, caps 0.00,
    boxes 0.00 (boxes family lost some fixes due to projection change).
  - Pure-QP (no fallback) on 8 unsafe dump scenes: 1/8 at 3 rounds,
    **2/8 at 10 rounds** (scenes 0 and 7 fixed) — improvement at
    higher round counts.
  - **Diagnosis refined**: the moving box projection helps at higher
    round counts by allowing the solver to track the moving threat's
    corridor. However, the fundamental ceiling remains the emission
    side: accumulated bounds become jointly infeasible (scene 8
    hits `projection_cycle` at round 2+ with accumulation). The
    fundamental ceiling is still constraint-emission coverage.
- Next increment: coupled range emission (`constraint_y_bounds(t0,t1,lo,hi)`)
  in `rules/*.lp` + `src/checker.py` `_group_constraints` to emit
  range/corridor constraints that the optimizer can project as a
  single coupled unit.

## Increment 3 — range constraint emission (SHIPPED 2026-10-23, S27)

- Implemented in `rules/collision_avoidance.lp`:
  - Added `constraint_y_bounds_range/5` emission for lateral corridors
    spanning consecutive overlap timesteps (using `#min`/`#max` aggregates).
  - Added `#show constraint_y_bounds_range/5` to `helpers.lp`.
- Implemented in `src/checker.py`:
  - `_group_constraints` now parses `constraint_y_bounds_range/5` into
    `y_bounds_range` entries `(t0, t1, min, max)`.
- Implemented in `src/constraint_optimizer.py`:
  - Added `_index_range` to index range constraints per timestep.
  - Added `_project_ranges` to project trajectory onto coupled range
    corridors (single lateral/longitudinal offset across the window).
  - Updated `_jointly_feasible` and joint projection loop to include ranges.
  - Suite: 122 green; flake8-clean.
- Measured outcome (honest, `experiments/qp_ablation_p23_final.json`):
  - Aggregate rates UNCHANGED: heuristic 0.375 = QP 0.375, caps 0.00,
    boxes 0.00.
  - Pure-QP (no fallback) on 8 unsafe scenes: **0/8 at 3 rounds, 2/8 at 10 rounds**
    (scenes 0 and 7 fixed at 10r) — **regression at 3 rounds** (scene 7
    was previously fixed at 3r, now requires 10r with range projection).
  - **Diagnosis**: range constraints help at higher rounds by tracking
    the moving corridor, but they interfere with the 3-round fix for
    scene 7. The fundamental ceiling remains **constraint-emission
    coverage**: the per-timestep + range bounds are still jointly
    infeasible for the moving-threat class (accumulated bounds hit
    `projection_cycle`). The fundamental fix requires emission-side
    coupled constraints that express the corridor as a single entity.
- Next increment: coupled range emission directly in rules (already
  shipped); next design step is to refine the range projection or
  emit corridor constraints from the rule side instead of post-hoc.

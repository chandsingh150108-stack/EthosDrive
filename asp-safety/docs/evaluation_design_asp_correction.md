# Evaluation Design: ASP-Guided Trajectory Correction vs. Heuristic Corrector

> **Shipped status (2026-10-21, sessions S13-S17)**: the A1 loop
> (`check(include_constraints=True)` -> pure-NumPy projected-gradient
> MVP -> ASP re-validation), the B1/B2 comparators, and the D2
> frozen-dump ablations ARE shipped. Measured so far: fix rate
> full-set QP 0.375 = heuristic 0.375 on the 8-unsafe-scene dump
> (scene-7 10x latency win), bound-family analog for A2/A3:
> full 0.375 > boxes 0.125 > caps 0.00. Deltas from this plan: pure-
> NumPy PGD (A5 pulled forward, no cvxopt), no A4/A6/A7 ablations yet,
> no B3 oracle; A7 direction filed as P-23. Evidence:
> `experiments/qp_ablation.json`, `qp_sweep.json`, `session.md`
> day summary 2026-10-03. Warm GPU A/Bs (2026-10-21, S18): fix-rate
> parity confirmed on real warm plans — calibration 0.50 = 0.50
> (*calibration_qp.json*), tracked replay 0.50 = 0.50 with identical
> scene sets (*replay_warm_heuristic.json* / *replay_warm_qp.json*);
> end-to-end QP latency is parity-to-slightly-worse (fallback-
> dominated means). Sections below are the ORIGINAL plan, kept
> for design context.
>
> **D3 shipped status (2026-10-22/23, P-25, session.md S19-S24)**:
> the CARLA closed-loop harness IS shipped and measured —
> `experiments/carla_bridge.py` + S1-S5 scenarios, frozen 15-run
> pilot matrix in `experiments/carla_metrics.json`: collision rate
> 13.3% (S3/S5 off arms only; all shielded arms 0 collisions),
> intervention rate 0.518, frame drops 0, give-up 0 throughout.
> Caveats: single seed, kinematic scripted actors (VRUs generate
> no physical collision events), lane plan engine drives the ego,
> ego-follow window at Low (Epic infeasible on 6 GB VRAM, backlog
> P-26). D3 rows below updated accordingly.

**Status**: Shipped (A1 loop + B1/B2 ablations on the D2 dump); the plan below is design context only.  
**Objective**: Provide experimental evidence that ASP-emitted structured correction constraints + continuous optimizer (QP/projected gradient) + symbolic re-validation outperforms the current heuristic dispatch-table corrector.

---

## 1. Baselines

| Baseline | Description | Role |
|----------|-------------|------|
| **B0: SparseDrive only (no correction)** | Planner raw 6×0.5 s output resampled to 4×1.0 s checker grid | Lower bound — planner raw safety/quality |
| **B1: Heuristic corrector (current)** | `TrajectoryCorrector` with `correction_strategies.py` dispatch table (7 params, 6 strategy groups) | **Primary comparator** — what we aim to beat |
| **B2: ASP-constrained corrector (proposed)** | `TrajectoryChecker.check(include_constraints=True)` → QP optimizer → re-validate | **Experimental treatment** |
| **B3: Oracle constrained corrector (upper bound)** | Hand-crafted optimal constraints from GT trajectory (same optimizer) | Ceiling — shows max potential of constraint-based approach |

All baselines share:
- Same `TrajectoryChecker` (63 rules)
- Same `NuScenesLoader` with tracked annotations (184 agents, 10 mini scenes)
- Same resampling: pred 6×0.5 s → T=0 anchor + pred[1,3,5] as T=1,2,3 (horizon_s=3)
- Same agent trajectories from planner detections (score≥0.5, cap 20, checker-culled)
- Same map data (N-1 extraction: lanes, widths, directions, intersections, tiled crosswalks/sidewalks)
- Same `on_failure='brake'` fallback (emergency hold at earliest remaining violation)

---

## 2. Ablations

| Ablation | Modification | Hypothesis Tested |
|----------|--------------|-------------------|
| **A1: Full constraint set** | All 63 rules emit constraint atoms (`x_bounds`, `y_bounds`, `v_max`, `a_max`, `lane_center`) | Complete ASP→QP→ASP loop |
| **A2: Speed + lane only** | Only R1.x, R2.x emit constraints; others use heuristic fallback | Are speed/lane constraints sufficient for most gains? |
| **A3: Collision + following only** | Only R3.x, R4.x emit constraints | Are agent-interaction constraints the differentiator? |
| **A4: No re-validation** | Skip final `checker.check()` after QP step | Is symbolic re-validation necessary for safety? |
| **A5: QP vs projected gradient** | Swap `cvxopt` QP for 10-step projected gradient (no external dep) | Is exact QP needed or is PGD sufficient? |
| **A6: Slack variables** | Add soft slacks on all constraints with penalty weight sweep | Does softening avoid infeasibility without sacrificing safety? |
| **A7: Stale constraint cache** | Reuse constraints from previous iteration when violation set unchanged | Does caching close the latency gap? |

---

## 3. Datasets / Scenarios

| Dataset | Size | Agent Type | Purpose |
|---------|------|------------|---------|
| **D1: nuScenes mini tracked-annotation replay (primary)** | 10 scenes, 184 tracked agents (92 moving / 92 parked) | Ground-truth tracked boxes from `sample_annotation` series | **Main evidence** — true agent histories eliminate parked-car hallucination noise; R3/R4/R13 verdicts become evidence |
| **D2: Calibration dump** | 10 scenes, frozen `trajs.json` (GT + pred_1s + map + boxes) | GPU-free sweep source | Threshold sweeps, ablations without GPU |
| **D3: CARLA closed-loop (P-25, shipped 2026-10-22)** | 5 scenarios (free drive, red-light run, cut-in, jaywalker, oncoming left turn) x 3 engines x 1 seed = 15-run frozen matrix | Scripted CARLA actors (kinematic) | **Safety-case numbers open-loop cannot provide**: violation rate/mile, intervention rate 0.518, collision rate 13.3% off vs 0% shielded (`experiments/carla_metrics.json`) |

**Scenario tags** (for per-scenario analysis):
- `high_speed` (ego > 15 m/s at any T)
- `dense_traffic` (> 10 moving agents within 50 m)
- `intersection_heavy` (> 3 controlled intersections in horizon)
- `pedestrian_present` (any VRU within 10 m)
- `construction_zone` (active zone overlapping trajectory)

---

## 4. Safety Metrics

| Metric | Formula | Target for B2 vs B1 |
|--------|---------|---------------------|
| **Collision rate (primary)** | `# scenes with any R4.x violation after correction / total scenes` | B2 ≤ B1; target **≤ 0.05** (current B1 replay: 0.5 fixed → 0.5 residual collisions) |
| **Red-light violation rate** | `# scenes with R5.1 after correction / total` | B2 = 0 (hard constraint) |
| **Lane departure rate** | `# scenes with R2.1 after correction / total` | B2 ≤ B1 |
| **VRU proximity violation rate** | `# scenes with R1.5 / R9.x after correction / total` | B2 ≤ B1 |
| **Critical rule residual** | `# scenes with any Critical-priority violation after correction / total` | B2 = 0 |
| **False-positive proxy (GT safe rate)** | `# scenes where GT has 0 violations / total` | Context — should stay ~0.3 (known map gaps) |

**Per-rule breakdown**: Report counts for each of the 63 rules (violation/hazard/warning) pre- and post-correction for both B1 and B2.

---

## 5. Trajectory-Quality Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| **L2(pred, GT) post-correction** | Mean L2 between corrected trajectory and GT (interpolated to checker grid) | B2 ≤ B1 + 0.5 m (safety should not excessively degrade tracking) |
| **Heading RMSE** | `sqrt(mean((atan2(dy_corr, dx_corr) - atan2(dy_gt, dx_gt))²))` | B2 ≤ B1 |
| **Jerk (comfort proxy)** | `mean(|a_t - a_{t-1}|)` over corrected trajectory | B2 ≤ B1 (constraints include `a_max` from R12.1) |
| **Path length deviation** | `|sum(||Δcorr||) - sum(||Δgt||)| / sum(||Δgt||)` | B2 ≤ 0.15 |
| **Constraint tightness** | Mean `(v_max_emitted - v_actual) / v_max_emitted` at each T | Diagnostic — shows how conservative ASP bounds are |

---

## 6. Computational-Cost Metrics

| Metric | B1 (current) | B2 Target | Measurement |
|--------|--------------|-----------|-------------|
| **Checker latency** | ~180 ms (real), 37 ms (20-agent synthetic) | ≤ 200 ms (added `#show constraint/*` atoms) | `time.perf_counter()` around `check()` |
| **Corrector latency** | ~5,700 ms (replay), ~2,500 ms (calibration) | **≤ 3,000 ms** (target: 50% reduction via fewer fixpoint iterations) | Wall time of `correct()` |
| **Fixpoint iterations** | 8–15 typical (stagnation break at 3) | **≤ 5** (QP solves all active constraints jointly) | Counter in `correct()` |
| **QP solve time** | N/A | **≤ 5 ms** per iteration (8 vars × 4 timesteps) | `time.perf_counter()` inside optimizer |
| **Total pipeline latency** | ~9–12 s/scene | **≤ 6 s/scene** | End-to-end `pipeline.run()` |

---

## 7. Failure Analysis

For every scene where B2 fails to fix (residual violations after correction):
1. **Violation taxonomy**: Which rule(s) remain? (Critical vs High vs Medium)
2. **Constraint emission trace**: What constraints did ASP emit for this scene? Were they infeasible?
3. **QP status**: Optimal / infeasible / unbounded / max-iter
4. **Slack activation**: Which soft slacks (if A6) were non-zero?
4. **Comparison to B1**: Did B1 fix it? If yes, why did B2 fail?
5. **Root cause category**:
   - Over-constrained (conflicting rules) → need priority-weighted slacks
   - Under-constrained (missing constraint atom for rule X) → ASP gap
   - Optimizer numerical issue → solver config
   - Stale constraints (cache hit) → invalidation logic

Produce a **failure ledger** (JSON) per scene for paper appendix.

---

## 8. Statistical Reporting

| Aspect | Method |
|--------|--------|
| **Per-scene paired comparison** | B1 vs B2 on same 10 scenes (paired t-test / Wilcoxon on binary fix/no-fix) |
| **Confidence intervals** | Bootstrap 10,000 resamples for collision rate, L2, latency |
| **Effect size** | Cohen's d for L2; risk difference for binary safety outcomes |
| **Multiple comparison correction** | Bonferroni across 6 ablation variants (A1–A6) |
| **Visualization** | Per-scene waterfall: GT violations → pred violations → B1 residual → B2 residual; latency scatter (checker vs corrector) |
| **Reproducibility** | Seed=42 for planner; frozen `trajs.json` for GPU-free ablations; `requirements.txt` + `pip freeze` snapshot |

---

## 9. Evidence Classification

### Already Measured (from `replay_baseline.json` / `calibration_baseline.json`)
- ✅ B1 collision rate: **0.50 fixed** (replay), 0.50 fixed (calibration warm)
- ✅ B1 corrector latency: **8,714 ms** (replay), 2,562 ms (calibration warm)
- ✅ B1 fixpoint iterations: implicit (stagnation break triggers)
- ✅ GT safe rate: **0.30** (known FP proxy from map gaps)
- ✅ Pred safe rate: **0.00** (replay), **0.20** (calibration warm)
- ✅ Planner L2: **11.54 m** (replay cold), **10.18 m** (calibration warm)
- ✅ Checker latency: **180 ms** (replay), **60 ms** (calibration)
- ✅ 63-rule catalog coverage: **73/73 findings fire** (`test_rule_registry`)
- ✅ 20-agent synthetic checker: **37 ms** (`test_performance`)

### Planned Experiments (not yet run)
- ⬜ B2 (ASP-constrained corrector) end-to-end on D1
- ⬜ Ablations A1–A7 on D2 (frozen trajectories)
- ⬜ Constraint emission coverage: % of rules emitting ≥1 constraint atom
- ⬜ QP feasibility rate across scenes
- ⬜ Slack weight sweep (A6)
- ✅ CARLA closed-loop safety-case metrics (D3) — DONE 2026-10-22: 15-run matrix, intervention 0.518, collision 13.3% off vs 0% shielded (`experiments/carla_metrics.json`)

### Unsupported Claims (would require new evidence)
- ❌ "ASP correction reduces collisions by X%" — needs B2 measurement
- ❌ "Constraint-based correction is faster" — needs B2 latency
- ❌ "Symbolic re-validation catches Y% of optimizer failures" — needs A4
- ✅ "CARLA closed-loop intervention rate is Z" — D3 done: measured 0.518 (single seed, kinematic actors; `experiments/carla_metrics.json`)

---

## 10. Open-Loop vs Closed-Loop Evidence Gap

| Property | Open-Loop (Current D1/D2) | Closed-Loop (D3 CARLA) |
|----------|---------------------------|------------------------|
| **Agent behavior** | Replayed from log (frozen) | Reactive to ego (simulated) |
| **Ego action affects agents** | No | Yes |
| **Multi-step consequences** | Not observed | Observed |
| **Metrics** | L2, rule violations, fix rate | Violations/mile, interventions, min TTC |
| **Safety case strength** | "Layer flags/corrects logged plans" | "Layer prevents accidents in sim" |
| **Status** | **Implemented, evidence collected** | **Feasibility verified (S12), bridge/scenarios/metrics not built** |

**Recommendation**: Publish open-loop evidence (D1 + ablation table) as primary contribution. Note closed-loop as "future work — infrastructure provisioned, bridge design documented in `replay_eval_design.md` §B". Do not overclaim.

---

## 11. Experiment Execution Plan

| Phase | Task | Depends On | Est. Time |
|-------|------|------------|-----------|
| **P1: Constraint emission** | Add `#show constraint/*` to `helpers.lp`; emit `x_bounds`, `y_bounds`, `v_max`, `a_max`, `lane_center` from each rule file | `helpers.lp` + 15 rule files | 2 days |
| **P2: Checker extension** | `check(include_constraints=True)` returns `(violations, constraints_dict)` | P1 | 1 day |
| **P3: QP corrector** | New `CorrectorQP` class using `cvxopt`/`scipy.optimize`; replaces dispatch table | P2 | 3 days |
| **P4: Integration + tests** | Wire into `corrector.py`; add `test_constraint_emission.py`, `test_qp_feasibility.py` | P3 | 2 days |
| **P5: Ablation runs** | Run A1–A7 on `trajs.json` dump (GPU-free) | P4 + D2 dump | 1 day |
| **P6: Full D1 replay** | Run B1 vs B2 on 10 tracked-annotation scenes | P4 | 2 hours |
| **P7: Analysis + writeup** | Failure ledger, statistical tables, figures | P5 + P6 | 2 days |

**Total**: ~11 working days (sequential). Critical path: P1→P2→P3→P4.

---

## 12. Acceptance Criteria for "ASP Correction is Better"

| Criterion | Threshold |
|-----------|-----------|
| **Safety**: B2 collision rate ≤ B1 collision rate (on D1 replay) | Binary: yes/no |
| **Safety**: B2 Critical residual = 0 | Binary: yes/no |
| **Quality**: B2 L2 ≤ B1 L2 + 0.5 m | Numerical |
| **Speed**: B2 corrector latency ≤ 3,000 ms (50% of B1 replay) | Numerical |
| **Convergence**: B2 fixpoint iterations ≤ 5 (vs B1 8–15) | Numerical |
| **Ablation**: A1 (full constraints) ≥ A2/A3 (partial) on fix rate | Numerical |
| **Regression**: All 93 existing tests pass | Binary: `pytest tests/ -v` |

If ≥ 5 of 7 criteria met with statistical significance (p < 0.05 on paired tests), the ASP-constrained approach is validated for the paper.

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| QP infeasibility on complex scenes | Medium | Corrector fails → brake fallback | Soft slacks (A6); priority-weighted penalties; warm-start from previous iterate |
| Constraint emission incomplete (rules missing atoms) | Medium | B2 underperforms B1 | `test_constraint_emission.py` asserts every rule with `violation` head emits ≥1 constraint |
| Latency regression (QP + re-validate > fixpoint) | Low | B2 slower than B1 | Cache constraints when violation set unchanged (A7); `#show` keeps atom scan fast |
| cvxopt/scipy dependency issues | Low | Build breaks | Fallback to projected gradient (A5) with pure NumPy |
| CARLA never provisions | High | ~~D3 unavailable~~ | MATERIALIZED 2026-10-22: server provisioned on the project laptop (0.9.15, Town10HD_Opt); D3 shipped — cite `experiments/carla_metrics.json` with single-seed/kinematic-actor caveats |

---

## 14. File/Module Impact Map

| File | Change Type | Scope |
|------|-------------|-------|
| `rules/helpers.lp` | Add `#show constraint/5` (x_bounds, y_bounds, v_max, a_max, lane_center) | Minimal |
| `rules/*.lp` (15 files) | Add constraint emission in rule bodies | Per-rule; ~3–5 lines each |
| `src/checker.py` | New `check(include_constraints=True)` path; parse constraint atoms | ~80 lines |
| `src/corrector.py` | New `CorrectorQP` class; deprecate dispatch table | ~200 lines new, ~130 old removed |
| `src/correction_strategies.py` | **Delete** — strategies move to ASP | Full removal |
| `tests/test_constraint_emission.py` | New: assert every violation rule emits constraints | New file |
| `tests/test_qp_feasibility.py` | New: QP solves on known violation scenarios | New file |
| `experiments/replay_eval.py` | Add B2 path + ablation loop | ~100 lines |
| `experiments/run_calibration.py` | Add `--corrector qp|heuristic` flag | ~20 lines |

---

## 15. Appendix: Key Rules for Constraint Emission

These rules have **natural, well-defined geometric bounds** that translate directly to constraints — highest priority for P1:

| Rule | Constraint Atoms to Emit |
|------|---------------------------|
| R1.1–R1.3 | `v_max(ego, T, Limit)` — speed cap per timestep |
| R1.4 | `v_min(ego, T, MinLimit)` — speed floor (when safe) |
| R1.5 | `v_max(ego, T, Cap)` — VRU-proximity cap (min across VRUs) |
| R2.1 | `y_bounds(ego, T, Y_center - W/2, Y_center + W/2)` — lane bounds |
| R2.2 | `y_bounds` + `x_bounds` for lane-change corridor + gap requirement |
| R3.1–R3.2 | `x_bounds(ego, T, lead_x - min_gap, lead_x)` — following corridor |
| R4.1 | `y_bounds(ego, T, V_y ± min_lateral)` — lateral clearance corridor |
| R4.2 | `x_bounds(ego, T, V_x - min_long, V_x)` — longitudinal floor |
| R5.1 | `x_bounds(ego, T, -∞, stopline_x)` — stop line hard bound |
| R5.2 | `v_max(ego, T, sqrt(2*a_max*D))` — dilemma-zone stop feasibility |
| R6.1–R6.5 | `x_bounds` at stop line / yield line / crosswalk |
| R7.2 | `x_bounds` + `y_bounds` for turn path corridor + oncoming TTC |
| R9.1–R9.5 | `x_bounds`/`y_bounds` around crosswalk / bike lane / sidewalk |
| R11.1 | `v_max(ego, T, sqrt(2*max_decel*D_hazard))` — emergency braking feasibility |
| R12.1 | `a_max(ego, T, max_comfort_decel)`, `a_max(ego, T, max_comfort_accel)` |
| R13.2 | `v_max(ego, T, closing_rate * TTC_threshold)` — TTC-derived cap |

These 18 rules cover >80% of observed violations in replay/calibration data (`collision_avoidance`, `lane_keeping`, `uncontrolled_priority`, `longitudinal_clearance`, `lateral_clearance`, `cut_in`, `unsafe_lane_change_gap`, `urban_speed`, `vru_proximity_speed`, `red_light`, `crosswalk_yield`, `emergency_yield`, `excess_accel`, `excess_decel`).

---

*End of evaluation design.*
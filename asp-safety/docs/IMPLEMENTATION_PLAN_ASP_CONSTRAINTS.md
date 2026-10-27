# Implementation Plan: ASP-Guided Correction Constraint Generation

> **Shipped status (2026-10-21, sessions S13-S17)**: P1-P5 shipped with
> deltas — constraint atoms emitted from all 15 rule files (grouped
> bound heads `v_max/v_min/x_bounds/y_bounds/a_max/lane_center`, not a
> `constraint/5` tuple), `check(include_constraints=True)`, pure-NumPy
> `src/constraint_optimizer.py` (projected gradient, T0 pinned; no
> cvxopt), `CorrectorQP` (3 solve/re-validate rounds + flagged heuristic
> fallback, `bounds='full'|'caps'|'boxes'`), pipeline `corrector_mode`,
> `--corrector` calibration flag. Deltas: `src/correction_strategies.py`
> was NOT deleted (heuristic stays the default engine); no slacks (A6);
> `asp_constraints.py` folded into checker/optimizer. Evidence:
> `experiments/{qp_ablation,qp_sweep}.json`; ledger in
> `docs/rule_divergences.md` (P1/P3-P4/P6/QP sections). GPU runs
> (2026-10-21, S18): warm calibration + warm tracked-replay A/Bs
> both fix-rate parity (0.50/0.50) — evidence
> `experiments/{calibration_qp,replay_warm_heuristic,replay_warm_qp}.json`.
> The plan below
> is the original design; on conflict, this banner plus the divergence
> sections are authoritative.

## Overview

Replace the heuristic dispatch-table corrector with an **ASP-emitted constraint interface** where the solver returns structured correction bounds that a continuous optimizer (QP/projected gradient) solves, followed by symbolic re-validation.

**Current**: SparseDrive → ASP Check (violations) → Heuristic Dispatch Table → Fixpoint Loop → Re-check  
**Proposed**: SparseDrive → ASP Check (violations + **constraints**) → **QP Optimizer** → Re-check

---

## 1. Proposed Modules

### New Modules

| Module | Purpose |
|--------|---------|
| `src/asp_constraints.py` | Constraint data structures, emission parsing, QP formulation |
| `src/constraint_optimizer.py` | QP/projected-gradient solver over constraint polytope |
| `tests/test_constraint_emission.py` | Assert every violation rule emits ≥1 constraint atom |
| `tests/test_qp_feasibility.py` | QP solves on known violation scenarios |

### Modified Modules

| Module | Changes |
|--------|---------|
| `rules/helpers.lp` | Add `#show constraint/5` for 5 constraint atom types |
| `rules/*.lp` (15 files) | Emit constraint atoms in rule bodies |
| `src/checker.py` | New `check(include_constraints=True)` path; parse constraint atoms |
| `src/corrector.py` | New `CorrectorQP` class; deprecate dispatch-table methods |
| `src/correction_strategies.py` | **Delete** — strategies move to ASP constraint emission |
| `experiments/replay_eval.py` | Add B2 path + ablation loop |
| `experiments/run_calibration.py` | Add `--corrector qp|heuristic` flag |

---

## 2. Interfaces

### 2.1 Constraint Atoms (ASP → Python)

Add to `#show` block in `helpers.lp`:

```asp
#show constraint/5.
```

**Constraint atom schema** (all values in cm, cm/timestep, integer):

| Atom | Arguments | Meaning |
|------|-----------|---------|
| `constraint_x_bounds(V, T, Min, Max)` | vehicle, timestep, min_x_cm, max_x_cm | Longitudinal corridor |
| `constraint_y_bounds(V, T, Min, Max)` | vehicle, timestep, min_y_cm, max_y_cm | Lateral corridor |
| `constraint_v_max(V, T, Limit)` | vehicle, timestep, max_speed_cm_per_step | Speed cap |
| `constraint_a_max(V, T, Limit)` | vehicle, timestep, max_accel_cm_per_step2 | Accel cap (comfort) |
| `constraint_lane_center(V, T, Y)` | vehicle, timestep, center_y_cm | Preferred lateral position |

### 2.2 Checker Extension

```python
def check(
    self,
    trajectory: np.ndarray,
    ...,
    include_constraints: bool = False,
) -> Tuple[bool, List[Dict], Optional[Dict]]:
    """
    Returns:
        (is_safe, violations, constraints_dict)
        constraints_dict = {
            'x_bounds': [(t, min_cm, max_cm), ...],
            'y_bounds': [(t, min_cm, max_cm), ...],
            'v_max': [(t, limit_cm_per_step), ...],
            'a_max': [(t, limit_cm_per_step2), ...],
            'lane_center': [(t, y_cm), ...],
        } if include_constraints=True else None
    """
```

### 2.3 Corrector Interface (Unchanged)

```python
def correct(
    self,
    trajectory: np.ndarray,
    **kwargs
) -> Tuple[np.ndarray, List[Dict]]:
    """Returns (corrected_trajectory, applied_corrections)"""
```

### 2.4 QP Optimizer Interface

```python
class ConstraintOptimizer:
    def solve(
        self,
        trajectory: np.ndarray,      # (T, 2) in meters
        constraints: Dict,           # from checker with include_constraints=True
        params: Dict,                # solver params (slack weights, etc.)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Returns:
            corrected_trajectory: (T, 2) in meters
            info: {
                'status': 'optimal'|'infeasible'|'unbounded',
                'slack_values': {...},
                'iterations': int,
                'solve_time_ms': float,
            }
        """
```

---

## 3. Data Structures

### 3.1 ConstraintDict (Python-side)

```python
@dataclass
class ConstraintDict:
    x_bounds: List[Tuple[int, int, int]]      # (t, min_cm, max_cm)
    y_bounds: List[Tuple[int, int, int]]      # (t, min_cm, max_cm)
    v_max: List[Tuple[int, int]]              # (t, limit_cm_per_step)
    a_max: List[Tuple[int, int]]              # (t, limit_cm_per_step2)
    lane_center: List[Tuple[int, int]]        # (t, y_cm)
```

### 3.2 QP Formulation

**Variables**: `x_t, y_t` for `t = 0..T-1` (meters → scaled to cm in QP)

**Objective** (minimize deviation from original trajectory + slack penalties):
```
minimize Σ_t ||(x_t, y_t) - (x_t^orig, y_t^orig)||² 
       + W_slack * Σ_slack slack²
```

**Constraints** (all linear in x_t, y_t):
- `x_bounds[t][0] <= x_t <= x_bounds[t][1]`
- `y_bounds[t][0] <= y_t <= y_bounds[t][1]`
- `(x_t - x_{t-1})² + (y_t - y_{t-1})² <= (v_max[t])²` → linearized via convex relaxation or kept as QCQP
- `|a_t| <= a_max[t]` where `a_t = (x_t - 2*x_{t-1} + x_{t-2})` (discrete accel)
- Optional: `y_t == lane_center[t]` (soft, with slack)

**Simplification for MVP**: Use projected gradient on box constraints (x_bounds, y_bounds, v_max as speed limit per step) + re-validation. Full QP with acceleration constraints in Phase 2.

---

## 4. Checker/Corrector Interaction Changes

### 4.1 Checker Changes (`src/checker.py`)

1. Add `include_constraints: bool = False` parameter to `check()`
2. When `True`:
   - Add `#show constraint/5` to the control (already in helpers.lp)
   - Parse constraint atoms in `on_model` alongside violations
   - Return `(is_safe, violations, constraints_dict)`
3. When `False` (default): unchanged behavior for backward compatibility

### 4.2 Corrector Changes (`src/corrector.py`)

1. Add `CorrectorQP` class (or extend `TrajectoryCorrector` with `mode='qp'|'heuristic'`)
2. `correct()` flow:
   ```python
   # 1. Check with constraints
   is_safe, violations, constraints = self.checker.check(traj, include_constraints=True, **kwargs)
   
   # 2. If safe, return early
   if is_safe: return traj.copy(), []
   
   # 3. Solve QP with constraints
   corrected, qp_info = self.optimizer.solve(traj, constraints)
   
   # 4. Re-validate
   is_safe2, violations2 = self.checker.check(corrected, **kwargs)
   
   # 5. If still unsafe: fall back to heuristic or brake
   if not is_safe2:
       return self._fallback_correction(traj, violations, **kwargs)
   
   return corrected, applied_corrections
   ```

### 4.3 Backward Compatibility

- Default `TrajectoryCorrector` constructor keeps `mode='heuristic'` (current behavior)
- New `TrajectoryCorrector(checker, mode='qp')` uses ASP constraints
- `correction_strategies.py` remains for heuristic mode; deleted only after QP validated
- All existing tests pass with `mode='heuristic'`

---

## 5. ASP Output Representation

### 5.1 Constraint Emission Pattern (per rule file)

Each rule with a `violation` head adds a corresponding `constraint` atom in its body:

```asp
% Example: R1.1 urban_speed
violation(urban_speed, ego, T) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, urban), speed_limit(L, Limit),
    not speed_adjusted(L, T),
    S2 > Limit*Limit.

% Constraint emission (same condition, emits bound instead of violation)
constraint_v_max(ego, T, Limit) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, urban), speed_limit(L, Limit),
    not speed_adjusted(L, T).
```

### 5.2 Priority Handling

- Critical/High rules emit **hard** constraints (no slack)
- Medium/Low rules emit **soft** constraints (with slack variables)
- `threat_level` / `ttc_band` can modulate slack weights

### 5.3 Rules Emitting Constraints (Priority 1)

| Rule | Constraint Atoms |
|------|------------------|
| R1.1–R1.3 | `v_max(ego, T, Limit)` |
| R1.4 | `v_min(ego, T, MinLimit)` (when safe) |
| R1.5 | `v_max(ego, T, Cap)` (min across VRUs) |
| R2.1 | `y_bounds(ego, T, Y_c - W/2, Y_c + W/2)` |
| R2.2 | `x_bounds` + `y_bounds` for lane-change corridor |
| R3.1–R3.2 | `x_bounds(ego, T, lead_x - min_gap, lead_x)` |
| R4.1 | `y_bounds(ego, T, V_y ± min_lateral)` |
| R4.2 | `x_bounds(ego, T, V_x - min_long, V_x)` |
| R5.1 | `x_bounds(ego, T, -∞, stopline_x)` |
| R5.2 | `v_max(ego, T, sqrt(2*a_max*D))` |
| R6.x | `x_bounds` at stop/yield lines |
| R7.2 | `x_bounds` + `y_bounds` for turn corridor |
| R9.x | `x_bounds`/`y_bounds` around crosswalks/bike lanes |
| R11.1 | `v_max(ego, T, sqrt(2*max_decel*D_hazard))` |
| R12.1 | `a_max(ego, T, max_comfort_decel/accel)` |
| R13.2 | `v_max(ego, T, closing_rate * TTC_threshold)` |

---

## 6. Candidate Trajectory Representation

- **Input to optimizer**: `(T, 2)` numpy array in **meters** (ego-frame)
- **Optimizer internal**: Scale to cm (×100) for integer-friendly QP, or keep float
- **Output**: `(T, 2)` numpy array in **meters**
- **Timestep convention**: `T=0` is current ego pose (0,0); `T=1,2,3` are +1,+2,+3 s (STEP_DT=1.0)

---

## 7. Optimizer Integration Point

### 7.1 MVP: Projected Gradient Descent (no external deps)

```python
def projected_gradient_step(traj, constraints, lr=0.1, steps=50):
    """Simple box-projection + speed-limit projection."""
    for _ in range(steps):
        grad = 2 * (traj - traj_orig)  # L2 toward original
        traj -= lr * grad
        
        # Project onto constraints
        for t, (x_min, x_max) in constraints.get('x_bounds', []):
            traj[t, 0] = np.clip(traj[t, 0], x_min/100, x_max/100)
        for t, (y_min, y_max) in constraints.get('y_bounds', []):
            traj[t, 1] = np.clip(traj[t, 1], y_min/100, y_max/100)
        for t, v_max in constraints.get('v_max', []):
            if t > 0:
                v = np.linalg.norm(traj[t] - traj[t-1])
                v_lim = v_max / 100  # cm/step -> m/step
                if v > v_lim:
                    traj[t] = traj[t-1] + (traj[t] - traj[t-1]) * (v_lim / v)
    return traj
```

### 7.2 Phase 2: QP with `cvxopt` / `scipy.optimize`

```python
# cvxopt QP: minimize 0.5 x'P x + q'x  s.t.  Gx <= h, Ax = b
# Variables: [x_0, y_0, x_1, y_1, ...] (2T vars)
```

---

## 8. Re-Validation Loop

```python
def correct(self, trajectory, **kwargs):
    # 1. Check with constraints
    is_safe, violations, constraints = self.checker.check(
        trajectory, include_constraints=True, **kwargs)
    
    if is_safe:
        return trajectory.copy(), []
    
    # 2. Optimize within constraints
    corrected, opt_info = self.optimizer.solve(trajectory, constraints)
    
    # 3. Re-validate (symbolic check)
    is_safe2, violations2 = self.checker.check(corrected, **kwargs)
    
    # 4. If failed: iterative refinement or fallback
    if not is_safe2:
        # Option A: Add violated constraints, re-solve (multi-shot)
        # Option B: Fall back to heuristic corrector
        # Option C: Emergency brake (on_failure='brake')
        return self._fallback(trajectory, violations, **kwargs)
    
    return corrected, [{'rule': 'qp_optimization', ...}]
```

---

## 9. Backward Compatibility

| Aspect | Strategy |
|--------|----------|
| `TrajectoryCorrector` API | Unchanged; add `mode` parameter |
| `checker.check()` API | Add optional `include_constraints` param |
| `correction_strategies.py` | Keep for heuristic mode; deprecate later |
| Config YAML | Add `corrector_mode` and QP params |
| Tests | All existing tests run with `mode='heuristic'` |

---

## 10. Tests

### 10.1 New Tests

| Test File | Purpose |
|-----------|---------|
| `tests/test_constraint_emission.py` | For every rule in `all_known_rules()` that has a `violation` head, assert ≥1 `constraint` atom emitted on a violating trajectory |
| `tests/test_qp_feasibility.py` | QP solver produces feasible trajectory on synthetic violations (speed, lane, collision, stop) |
| `tests/test_qp_revalidation.py` | Optimized trajectory passes `checker.check()` |
| `tests/test_new_rule_constraint.py` | Add dummy rule + constraint; verify corrector respects it without Python changes |
| `tests/test_ablation_constraints.py` | Ablation: full constraints vs speed-only vs collision-only vs none |

### 10.2 Existing Tests (Must Pass)

- All 93 current tests (`pytest tests/ -v`)
- `test_all.py` 7/7 categories
- `test_rules.py` 13/13
- `test_performance.py` budgets (checker <50ms bare, <100ms 20-agent)

---

## 11. Benchmarks

| Benchmark | Target |
|-----------|--------|
| **Constraint emission coverage** | 100% of violation rules emit ≥1 constraint |
| **QP solve time** | ≤ 5 ms per iteration (4 timesteps × 2 vars) |
| **Corrector latency (real scenes)** | ≤ 3,000 ms (vs current 5,700–8,700 ms) |
| **Fixpoint iterations** | ≤ 5 (vs current 8–15) |
| **Replay baseline (10 scenes)** | pred_fixed_rate ≥ 0.5 (current heuristic) |
| **Collision rate (replay)** | ≤ 0.05 (current 0.5 residual) |
| **L2 post-correction** | ≤ heuristic L2 + 0.5 m |

---

## 12. Documentation Changes

| File | Change |
|------|--------|
| `docs/api_reference.md` | Document `check(include_constraints=True)` and `CorrectorQP` |
| `docs/usage.md` | Add ASP-constrained corrector example |
| `docs/conventions.md` | Document constraint atom schema |
| `paper/report.tex` | Rewrite Method section around constraint-based correction |

---

## 13. Implementation Phases

| Phase | Tasks | Est. Days |
|-------|-------|-----------|
| **P1: Constraint Emission** | Add `#show constraint/5` to helpers.lp; emit constraints in 18 priority rule files | 2 |
| **P2: Checker Extension** | `include_constraints` path; parse constraint atoms; return ConstraintDict | 1 |
| **P3: QP Optimizer (MVP)** | Projected gradient on box+speed constraints; `ConstraintOptimizer` class | 2 |
| **P4: Corrector Integration** | `CorrectorQP` class; re-validation loop; fallback to heuristic/brake | 2 |
| **P5: Tests** | 5 new test files; ensure all 93 existing pass | 2 |
| **P6: Ablation & Validation** | Run on `trajs.json` dump (GPU-free); full replay baseline | 1 |
| **P7: Docs & Cleanup** | Delete `correction_strategies.py`; update docs | 1 |

**Total**: ~11 working days

---

## 14. Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| QP infeasibility on complex scenes | Medium | Soft slacks with priority weights; warm-start from previous iterate; fallback to heuristic |
| Constraint emission incomplete | Medium | `test_constraint_emission.py` tripwire; `test_rule_registry` already asserts 73/73 findings |
| Latency regression (QP + re-validate > fixpoint) | Low | Cache constraints when violation set unchanged; `#show` keeps atom scan fast |
| cvxopt/scipy dependency issues | Low | MVP uses pure NumPy projected gradient; QP as optional Phase 2 |
| CARLA never provisions | High | Explicitly exclude from paper; document as future work |

---

## 15. Acceptance Criteria (Definition of Done)

1. **All 93 existing tests pass** with `mode='heuristic'`
2. **New tests pass**: constraint emission (100%), QP feasibility, re-validation
3. **Replay baseline**: B2 (QP) pred_fixed_rate ≥ 0.5, collision rate ≤ 0.05, corrector_ms ≤ 3000
4. **Ablation**: Full constraints (A1) ≥ partial (A2/A3) on fix rate
5. **No regression**: Heuristic mode unchanged; config flag switches cleanly
6. **Documentation**: API reference, conventions, usage updated

---

## 16. Clarifying Questions

Before implementation begins, please confirm:

1. **Optimizer choice**: MVP with pure-NumPy projected gradient (no new deps), or Phase 2 with `cvxopt`/`scipy.optimize` from start?
2. **Constraint granularity**: Emit constraints per-timestep (T=0..3) or only at violation timesteps?
3. **Slack variables**: Include from MVP or add in Phase 2?
4. **Fallback strategy**: On QP failure → heuristic corrector, or → brake hold, or → multi-shot re-solve?
5. **Config management**: New `configs/correction_qp.yaml` or extend existing `correction_strategies.yaml`?

---

*End of implementation plan.*
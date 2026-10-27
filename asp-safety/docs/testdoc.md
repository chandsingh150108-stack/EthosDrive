# Test Documentation — ASP Safety Layer

> All commands run from `asp-safety/` with the environment active.
> CPU-only; no GPU or dataset needed. All tests use the placeholder
> planner/loader path unless stated otherwise.

```powershell
conda activate asp-safety
cd <path-to>\asp-safety
```

---

## 1. tests/test_all.py — 7-block smoke suite

> Pytest entry: `def test_all_components():` runs the same 7 checks
> under `pytest tests/` (RQ-6, 2026-10-21); the file is
> CWD-independent (paths derive from the tests-dir parent).

**Command:**

```powershell
python tests/test_all.py
```

**What we are doing:** One pass over every component: planner predict shape,
checker slow=safe, corrector fast→safe, loader sample shape, `l2 == 1/3`,
pipeline `run` keys, full 63-rule catalog grounding via `basic.lp`.

**What it demonstrates:** The pipeline is plumbed end to end
(predict → check → correct → batch → ground). A pass here does not prove
rule semantics — only that nothing is unwired.

---

## 2. tests/test_checker.py — safety discrimination + vacuous guard

**Command:**

```powershell
python tests/test_checker.py
```

**What we are doing:**
- Slow straight trajectory `[0..5]` must return safe with zero violations.
- Fast trajectory `[0..100]` must return unsafe with a `speed_limit` finding.
- Vacuous guard: fast must yield ≥1 finding; overlapping pair must yield
  `collision_avoidance`; red-light geometry must yield `red_light`.
- Correction smoke via `get_safe_trajectory`.

**What it demonstrates:** The checker discriminates safe from unsafe on three
canonical hazards (speeding, overlap, red run) through real Clingo, and cannot
pass while silently reporting everything safe.

---

## 3. tests/test_corrector.py — repair convergence

**Command:**

```powershell
python tests/test_corrector.py
```

**What we are doing:**
- Fast trajectory through `corrector.correct()` must re-`check` safe with
  `corrections[0] == speed_limit`.
- Safe trajectory must return zero corrections and an equal array.
- Multi-violation input (speed + nearby agent) must produce corrections.
- Uncorrectable input raises `RuntimeError` (or flagged hold with
  `on_failure='brake'`); advisory findings pass through untouched;
  correction is idempotent; min-boost converges without oscillation;
  group caps + stagnation break bound churn.

**What it demonstrates:** The fixpoint loop converges on the canonical speed
case and is a no-op when already safe, with honest failure modes and
P-6 convergence guards pinned.

---

## 4. tests/test_data_loader.py — placeholder contracts

**Command:**

```powershell
python tests/test_data_loader.py
```

**What we are doing:** Init on a nonexistent path (placeholder mode), then
assert sample keys and shapes (`images (6,3,256,704)`, `intrinsics (6,3,3)`,
`extrinsics (6,4,4)`, `ego2global (4,4)`, float timestamp), trajectory shape
and dtype, map keys (`lanes`, `intersections`), lights key (`int_0.color`),
and `IndexError` on out-of-range scene index.

**What it demonstrates:** Downstream code receives the exact shapes the
planner and checker validate against. Scene 9 is hostile by design
(obstacle + reachable red) with fix-verified tests; the real nuScenes
branch is covered by calibration, not here.

---

## 5. tests/test_evaluation.py — metric formulas

**Command:**

```powershell
python tests/test_evaluation.py
```

**What we are doing:** Hand-made arrays through `MetricsCalculator`:
collision 100% (box over path) vs 0% (distant box), `l2 == 1/3`, detection
rate 1.0, false-positive rate 1.0 on empty truth, average time 20.0 ms.

**What it demonstrates:** Metric arithmetic is correct, including
yaw-aware footprints, batch-length validation, NaN guards, FDR naming,
and key-aware matching. Real checker/planner outputs feed the T-10
test; pure formula cases never touch Clingo.

---

## 6. tests/test_sparse_drive.py — planner interface shape

**Command:**

```powershell
python tests/test_sparse_drive.py
```

**What we are doing:** Init on the checkpoint path (falls back to placeholder),
`predict` must return `trajectory (6,2)`, `boxes (N,7)`, `scores (N,)`,
`velocities (N,3)`; a `(3,3,256,704)` input must raise `ValueError`.
Golden `_postprocess` values (incl. sliced velocities), fail-loud
matrix, RQ-2 projection math, STEP_DT pin, placeholder + temporal-API
determinism, and `estimate_ego_status` layout are pinned.

**What it demonstrates:** The neural↔symbolic interface contract (shapes and
keys). Real-model inference runs on the GPU build (planner/temporal
harnesses); the golden path keeps the schema honest without one.

---

## 7. tests/test_pipeline.py — end-to-end (most valuable)

**Command:**

```powershell
python tests/test_pipeline.py
```

**What we are doing:**
- Init on CPU, `run` on synthetic images; assert result keys
  (`original_trajectory` (6,2) native, `safe_trajectory` (4,2)
  checked grid, `horizon_s`, `violations`, `is_safe`,
  `inference_time_ms`, `timing_ms` split, `boxes`, `scores`),
  boolean verdict, positive latency.
- Detections-to-trajectories unit test (score filter, cap, CV
  projection + `dt`; no model). Batch error-schema + corrector-raise
  propagation (T-11, placeholder-fast).
- `run_batch` on 3 placeholder scenes; assert length and `scene_idx` keys.
- Planted red-light run: lane built from the planner's own CHECKED
  waypoints (T=0..3 grid), red on T=3; assert `is_safe is False`
  with `red_light` fired, then re-`check` the returned
  `safe_trajectory` and assert it is clean.

**What it demonstrates:** The full loop — neural proposal, ASP veto, heuristic
repair, verified fix — through real Clingo on placeholder data, with
detections feeding agent context (P-18) and stable failure contracts.

---

## 8. tests/test_rules.py — per-category rule harness (14 functions)

**Command:**

```powershell
python tests/test_rules.py
```

**What we are doing:** Safe/unsafe pairs per rule family through `check`,
plus co-firing priority and correction convergence:
helpers (bare slow safe), R1 speed, R2 lane, R3 following, R4 collision,
R5 red vs green, R6.5/R9 pedestrian, R11 obstacle + evasive handoff
(`insufficient` link pinned), R14 wet/night, R15 platoon + IF-2 wirings
(wave/multiway/merge), T-2 spot probes (6 extra firings), red+pedestrian
priority with corrected-safe check, correction shape+safety.

**What it demonstrates:** Core families fire on demand; the full
battery (turning, overtaking, predictive, intersection, comfort,
flags) lives in `test_t2_coverage.py` (§11) with the P-9 registry
(73/73 findings asserted) as the tripwire.

---

## 9. experiments/run_evaluation.py — offline batch demo

**Command:**

```powershell
python experiments/run_evaluation.py --num-scenes 10
```

**What we are doing:** Builds the CPU pipeline with placeholder planner and
loader, runs `run_batch` over 10 scenes, aggregates safe-rate, violation
breakdown, and average/p95 latency.

**What it demonstrates:** Batch orchestration and reporting on synthetic
scenes (hostile-by-design placeholder geometry fires every scene). Only
`average_inference_time` is wired; collision-rate and L2 improvement on real
data are measured by `run_calibration.py` (§12), not here.

---

## 12. experiments/run_calibration.py — real-data baseline (GPU)

**Command:**

```powershell
python experiments/run_calibration.py --out experiments/calibration_baseline.json --dump-trajs experiments/trajs.json
```

**What we are doing:** Per mini scene (CUDA): warm multi-frame walk
(default; `--cold` reproduces the old single-frame pass), predict,
L2 vs GT on the native grid, GT check (FP proxy) + anchored T=0 pred
check (T=1..3 from indices 1,3,5; `horizon_s` stamped), corrector
fix-rate, planner/checker/corrector timing split. Agent trajectories
come from planner detections (P-18). `--corrector heuristic|qp`
selects the corrector engine (S15; stamps `corrector_mode` in the
JSON summary). `--extrapolate` extends to T=4,5 by constant velocity
(N-3(b): rejected — artifact hits, fix-rate drop). `--dump-trajs`
saves maps+trajectories+boxes for GPU-free `sweep_thresholds.py` runs.

**What it demonstrates:** The honest warm numbers: L2 10.18 m, GT 0.3 /
pred 0.2 / fixed 0.5 (`calibration_baseline.json`, warm by default;
cold was L2 11.54 / GT 0.5 / pred 0.1 / fixed 0.7) with ledgered
residuals (no invented map data). `run_temporal.py` compares
warm-vs-cold (10.18 m warm); `cmd_sweep.py` probes command-head
semantics (weak — unverified mapping).

---

## 10. Quick Reference

| File | Real Clingo? | Real weights? | Proves what |
|------|--------------|---------------|-------------|
| test_all.py | yes (partial) | no | plumbing, not semantics |
| test_checker.py | yes | n/a | safe/unsafe discrimination |
| test_checker_edges.py | yes | n/a | edges: cull, warnings, times, P-4 geometry, aliasing |
| test_corrector.py | yes | n/a | fixpoint converges (+T-8 no-oscillation) |
| test_data_loader.py | no | no | placeholder schema |
| test_evaluation.py | no | n/a | metric formulas |
| test_sparse_drive.py | no (golden only) | no | interface shapes + STEP_DT + N-2 API/status |
| test_pipeline.py | yes | no | proposal→veto→fix loop |
| test_rules.py | yes | n/a | per-category firing |
| test_t2_coverage.py | yes | n/a | full battery firing+quiet + registry 73/73 |
| test_constraint_emission.py | yes | no | rules emit ASP bound atoms |
| test_qp_feasibility.py | yes (re-validate) | no | QP projections + bounds families + hold-back |
| run_evaluation.py | yes (via pipeline) | no | batch reporting |

## 11. tests/test_t2_coverage.py — full battery + registry (16 functions)

**Command:**

```powershell
python tests/test_t2_coverage.py
```

**What we are doing:** Firing + quiet pairs for every family missing
from §8: R1.4/R1.5, R2.2/R2.4/transition, R3.4, R4.1/R4.3-R4.5,
R5.2-R5.5, R6.1-R6.3, R7.1/R7.3-sightline/R7.5, R8.1-R8.4,
R9.1/R9.3-R9.5, R10.1/R10.3/R10.4, R11.4, R12.1-R12.3, R13.1-R13.4
(+R12.4 score via `include_internal`), R14.2/R14.3, plus
`test_rule_registry` (every catalog head asserted somewhere).

**What it demonstrates:** 73/73 catalog findings fire on demand.
Rule-design fixes proven here: R3.4/R15.4 int32 overflows,
R6.2 roundabout conflict zone, pedestrian obstacles, R2.1
transitions, targeted min-boost (corrector).

---

## 13. tests/test_constraint_emission.py — ASP bounds for every rule (12 functions)

**Command:**

```powershell
python tests/test_constraint_emission.py
```

**What we are doing:** With `include_constraints=True`, every rule
family on a violating trajectory emits ≥1 constraint atom (`v_max`/`
v_min`/`x_bounds`/`y_bounds`/`a_max`/`lane_center`): stop/hold boxes
(R5/R6/R7.1/R7.2/R9.1/R9.2/R10/R11.1/R15.1), corridors (R2.1 lane +
pull-back, gap bands R3.2/R15.2), exact caps (R3.1/R3.3, R14.1,
R5.4/R14.3), acceleration caps (R12.1/R3.4/R15.4 minus
`override_active`). Head-coverage self-check across the catalog;
documented exemptions (wrong-way, U-turn, signal timing, turn
radius, bike lane, sidewalk, evasion candidates, fixed obstacles,
TTC bands, threat arbitration, aggregation, parameter layer,
comfort shapes).

**What it demonstrates:** The solver hand-off for the QP corrector:
the ASP layer can express a violation as a continuous bound, not
just a label; constraints never affect the safe/unsafe verdict and
never pollute the violations list.

---

## 14. tests/test_qp_feasibility.py — CorrectorQP + optimizer (13 functions)

**Command:**

```powershell
python tests/test_qp_feasibility.py
```

**What we are doing:** Exact projected-gradient math on emitted
bounds (per-segment `v_max`/`v_min` rescale, `a_max` forward clamp,
tightest-box overlap, T=0 pinned, weak lane-center spring),
fail-loud on bad shape/non-finite input, safe-trajectory
passthrough; end-to-end re-validation: QP speed fix and QP red hold
both re-check clean through real Clingo; `bounds='full'|'caps'|
'boxes'` routing (caps fix pure speed where boxes honestly raise;
boxes-only converges a drift-out back in-lane across rounds);
one-sided hold-back vs centered boxes (exact `(t, -offset, cap)`
tuples, centered boxes absent — the optimizer cannot pull ego toward
the threat).

**What it demonstrates:** The QP path is honest: it never moves
T=0, it re-validates through ASP after updating, and it falls back
to the heuristic with a flagged head entry; the fallback never sees
`include_constraints`/`include_internal` (legacy 2-tuple contract
preserved by a pinned test).

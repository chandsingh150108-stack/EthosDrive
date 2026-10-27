# Future TODO - ASP Safety Layer

> **STATUS 2026-10-03 (S17): ASP-constrained correction track complete.**
> All 15 rule files emit constraint atoms (6 heads); `check(
> include_constraints=True)`; pure-NumPy `src/constraint_optimizer.py`;
> `CorrectorQP` (3 solve/re-validate rounds + flagged heuristic
> fallback); pipeline `corrector_mode='heuristic'|'qp'` (heuristic
> default); `--corrector` calibration flag; bound-family ablation;
> param sweep; one-sided hold-back fix. **120 tests green** (104 fast +
> 8 sparse_drive + 7 pipeline + 1 test_all; incl. 26 S13-S17).
> Heuristic path unchanged.
> Evidence: session.md S13-S17 + day summary 2026-10-03,
> experiments/{qp_ablation,qp_sweep}.json,
> tests/{test_constraint_emission,test_qp_feasibility}.py.
> Open: §2 unmarked items (P-13 rem, PR-3/P-8 rem, RQ-5 rem) + §5
> (P-23 horizon-joint/A7, P-25 CARLA bridge; T-14/RQ-6/N-5/DC-5/P-24/
> P-21/P-22 closed 2026-10-21 closure session, see session.md S18).
> Month-run (2026-09-26) status: 93 tests green (79 fast + 8 sparse/all
> + 6 pipeline); closed S8/S9/S10 — P-6/P-16, T-9/P-14, T-10/T-11,
> P-17, P-18, PR/P-8 (109->37 ms), N-3(c) KILLED, RQ-4 splits,
> Track-5 docs; S9/S10 — pipeline horizon unify, annotation-replay
> harness (184 tracked agents), warm-by-default calibration (L2
> 11.54->10.18), real-scene perf (tile caps, use_asp_in_lane gate).
> Evidence: session.md S6-S10, whatodo.md,
> experiments/{cmd_sweep_results,temporal_results,extrapolation_verdict,
> sweep_verdict}.md, docs/{rule_divergences,conventions,
> replay_eval_design}.md, calibration_baseline.json (warm: GT 0.3 /
> pred 0.2 / fixed 0.5), replay_baseline.json.
> Deferred with reasons: full multi-shot (dynamics dominate),
> parked-vs-live box classes (moot — series variance answers it).
> Real SparseDrive inference (CPU 3.9 s / CUDA 2.4 s), nuScenes-mini
> loading, 63-rule checking, correction, calibration v1 baseline
> (L2 11.5 m, GT safe 0.90, pred fixed 1.00). Full session log:
> `asp-safety/session.md`. Test audit: `docs/test_coverage.md`.
> Conda env `asp-safety` (py3.10) is the canonical runtime.

---

## 1. Completed archive

GAP-1–GAP-12 closed 2026-09-22 (63-rule catalog, shared predicates,
fact wiring, corrector coverage, perf guard, priority tests).

Sessions of 2026-09-25 (detail in `session.md`):
- Phase 0: deps, red-light per-intersection fallback, dead checker
  correction logic removed, vacuous-checker guard. Gate 0 green.
- Phase 1: `total_violations` blob fix (was never derived), weather
  layer wired (R1.1/R1.2/R3.2 variants, no double-fire), central
  rule->strategy table + YAML params, map-aware corrector targets,
  derived facts (oncoming/TTC/durations/sightlines/has_cleared),
  perf budget test, corrector failure-path + checker edge tests.
- Planner: K@[R|t] projection fix, fail-loud torch-optional
  postprocess, STEP_DT=1.0 contract + pinning test, golden harness.
- Real inference: mmcv-full/mmdet installed (mmdet3d excluded on
  version conflict), pure-torch deformable_aggregation fallback
  (verified 8.3e-07), flash_attn SDPA fallback, synthetic anchors
  (checkpoint-exact shapes), 10-dim box slicing, CPU fp16 off.
- GPU: torch+mmcv cu118, 2.4 s/frame at 693 MB VRAM.
- Real data: mini sample/trajectory/map loader, honest empty lights,
  Python-side `in_lane_fact` (any orientation), intent-gated perf
  fix, `reset_temporal()` per scene, calibration harness + v1
  baseline, R2.3 minimum-motion guard from a real GT finding.

---

## 2. Remaining work

> NOTE (2026-10-03): items marked `— CLOSED` are done (evidence in
> `session.md` / `whatodo.md`); kept for archeology, do not re-add.
> Unmarked items in §2 remain OPEN; all other open work is in §5.

### 2.1 Testing

#### T-2. Rule firing coverage still thin (HIGH) — CLOSED 2026-09-26
6 spot probes committed (`test_uncovered_spot_probes`); ~40 finding
names still lack committed firing tests, incl. R6.1–R6.4, most of
R7/R8/R10, R11.1/11.2/11.4, R12.2–R12.4, R13 bands/arbitration,
R14.2, R15.1/15.3/15.4. R13.3 arbitration and the R11.1→R11.2
handoff are untested single points of failure. **Fix:** one firing
+ one safe test per missing category. 3–5 h.

#### T-3. Return-bool style + no CI (HIGH) — CLOSED 2026-09-26
pytest installed and suites run under it, but test functions still
return bool instead of asserting (PytestReturnNotNone warnings).
**Fix:** convert to pure asserts; add CI workflow. 2–3 h.

#### T-8. `min_hwy_speed` vs `speed_limit` oscillation (MEDIUM) — CLOSED 2026-09-26
RuntimeError/advisory/idempotence cases covered; the boost-vs-scale
fight across iterations is not. **Fix:** 1 targeted test. 30 min.

#### T-9. Hostile placeholder scene (MEDIUM) — CLOSED 2026-09-26
Real loader path done. Placeholder still only shows the safe path.
**Fix:** hostile placeholder scene + reachable-red scene. 1 h.

#### T-10. Metrics on real `check()` outputs (LOW) — CLOSED 2026-09-26
Hand-made dicts vs `vehicle2`/`type` keys; passes by luck.
**Fix:** feed real outputs into metrics tests. 30 min.

#### T-11. Pipeline error paths (LOW) — CLOSED 2026-09-26
`run_batch` swallowing + `corrector` raising through `run()`.
**Fix:** 2 tests. 1 h.

#### T-12. Determinism asserts (LOW) — CLOSED 2026-09-26
Seed-42 placeholder, per-scene loader seeds unasserted (checker
determinism IS covered). **Fix:** 3 asserts. 20 min.

#### T-13. Wrong troubleshooting command in docs (LOW) — CLOSED 2026-09-26
`docs/usage.md` advises `TrajectoryChecker(device="cpu")`; no such
parameter exists. **Fix:** correct to planner `device` note. 10 min.

### 2.2 ASP rules

#### P-9. Rule verification remainder (MEDIUM) — CLOSED 2026-09-26
`effective_*` now consumed (IF-3). Left: rule-registry test (every
`violation/` head ≥1 firing test — pairs with T-2), R12.4
`comfort_score` counts one rule only, `override_active` is a
hand-enumerated list new critical rules must extend, review
`turn_crossing`/`clear_path` masking risk. 3–4 h.

#### P-19. Spec-vs-impl divergences (MEDIUM) — CLOSED 2026-09-26
`effective_min_gap` wired; rest open (R1.5 bands, R2 gap model,
R3.3 arity, R4 Euclidean-vs-projected, R5 renames, R6.1 tiebreak,
R6.2/R6.3 gating, R7.4 lead step, R9.1 corners, R10 behind/4,
R11.1 table, R12 formulations, R13 arity-4 future, R15 dead rank).
Tracked with DC-4. 4–6 h.

#### DC-4. Divergence decision log (MEDIUM) — CLOSED 2026-09-26
Create `docs/rule_divergences.md`: per item, wire / remove / accept.
2–3 h.

### 2.3 Geometry (P-4, MEDIUM) — CLOSED 2026-09-26
`gap_dist` Euclidean not longitudinal; `following_gap` relabeled;
`closing_rate` first-difference noise; `about_to_enter` can't tell
approaching from leaving; `stopped_before`/`yielding_maneuver` =
"stopped anywhere". **Fix:** lane-projection for gaps,
heading-gated approach detection, stop-line-relative stopped checks.
6–10 h.

### 2.4 Calibration & evaluation (option b)

#### N-1. Map-extracted limits and road types (HIGH) — CLOSED 2026-09-26
Calibration v1 proved the hardcoded 13.4 m/s + urban label false-
positives on real arterials (scene 7). **Fix:** extract posted
limits + road types from nuScenesMap (maps/ present in C:/data).
Blocks honest GT-safe measurement. 1 week with sweep below.

#### P-5. Threshold sweep proper (MEDIUM) — CLOSED 2026-09-26
v1 baseline exists (`experiments/calibration_baseline.json`: L2
11.5 m, GT safe 0.90). **Fix:** finding-rate-vs-threshold sweeps
for cap, bubble, bands, stop needs, follow gaps, comfort bounds,
worker clearance, TTC units; pick knees; per-rule defaults. 1 week
with N-1/N-2.

#### N-2. Temporal multi-frame inference (HIGH) — CLOSED 2026-09-26
Cold-start single frames give L2 11.5 m; pred-side calibration is
meaningless until predictions use history. **Fix:** harness that
walks consecutive samples with the temporal queue warm (no reset
between frames of one scene), then measures. 4–6 h + compute.

#### N-3. Prediction horizon product decision (MEDIUM) — CLOSED 2026-09-26
Model predicts 3 s at 0.5 s; checker contract is 6x1.0 s. Harness
resamples to 3x1.0 s (honest but blind beyond 3 s). **Fix:**
decide: accept 3 s horizon, extrapolate, or extend model horizon;
encode in STEP_DT docs + pipeline. 2 h + team decision.

#### P-11. Closed-loop / safety case (HIGH, long-term) — CLOSED 2026-09-26 (superseded by §5 P-25)
Open-loop flagging isn't safety evidence. **Fix:** CARLA harness +
nuScenes replay eval; rewrite report numbers with measured values.
2–4 weeks.

### 2.5 Corrector

#### P-6 remainder. Convergence guarantees (MEDIUM) — CLOSED 2026-09-26
Params now config-driven (RQ-1). Left: per-group iteration caps +
damped steps; fallback maximum-braking policy instead of raising;
upstream uncorrectable-plan contract (flag + hold + takeover).
6–10 h.

#### P-16 remainder. Correction realism (MEDIUM) — CLOSED 2026-09-26
Lane-center/speed-limit inputs done. Left: tail-aware shifts
(current point-only shifts risk accel spikes), stale-snapshot
multi-T application, `isinstance` dual-identity guard. 4–6 h.

### 2.6 Checker (P-15 remainder, MEDIUM) — CLOSED 2026-09-26
Cull validation done; dead correction logic removed. Left: honor
`active_times`/`times` (no invented T=0 phases), temporal guard/
worker facts, arity-safe collectors, remove dead `self.control` +
legacy wrappers, pin check/copy aliasing contract. 3–4 h.

### 2.7 Planner (P-13 remainder, HIGH) — OPEN
Projection/postprocess/spec-guards done. Left: real navigation
command + ego-state inputs (`gt_ego_fut_cmd` hardcoded straight,
`ego_status` zeros) — required before predictions on real scenes
mean anything. 2–3 h + real data (present).

### 2.8 Data (P-14 remainder, MEDIUM) — CLOSED 2026-09-26
Real scene count + dtype contract done. Left: hostile +
reachable-red placeholder scenes (pairs with T-9). 1 h.

### 2.9 Pipeline & metrics

#### P-17. Metric correctness (MEDIUM) — CLOSED 2026-09-26
Box spec unified (10-dim decode documented). Left: yaw/time-aware
collision, `zip` truncation, detection matching keys, FDR rename,
NaN/inf/shape validation in `l2_error`. 2–3 h.

#### P-18. Batch + schema (LOW–MEDIUM) — CLOSED 2026-09-26
`run_batch` drops zones/pedestrians/weather/intent/extras and never
supplies `other_vehicles`; no per-stage timing; unstable error
schema; `RuntimeError` propagates from `run()`. **Fix:** forward
all kwargs + boxes, per-stage timing, stable schema. 2 h.

### 2.10 Performance

#### PR-1/PR-2. Incremental + cached solving (MEDIUM) — CLOSED 2026-09-26 (deferred w/ reasons)
Up to 100 re-grounds per correction; static facts re-grounded per
check. **Fix:** Clingo incremental mode for the fixpoint loop;
multi-shot solving separating static/dynamic facts. 4–6 h.

#### P-8/PR-3. Culling + headroom (LOW–MEDIUM) — OPEN (corrector headroom)
150 m radius is distance-only (fast agent at 160 m dropped though
it closes 280 m over horizon). Intent-gating done. Left:
time-to-conflict-aware culling; per-group corrector caps. 2–4 h.

### 2.11 Refactor

#### RQ-3. Unify requirements (LOW) — CLOSED 2026-09-26
Root vs asp-safety requirements files diverge; conda env pins
(mmcv index URL, setuptools 75.8.0, numpy 1.26.4) undocumented.
**Fix:** single file + env export doc. 30 min + N-4 below.

#### RQ-4/RQ-5. Types + method splits (LOW) — RQ-4 CLOSED 2026-09-26; RQ-5 remainder open
`check()` 200+ lines, `_apply_correction()` 130+ lines; internal
methods lack type hints. 4–6 h.

#### N-4. Environment record (LOW) — CLOSED 2026-09-26
Record the working env exactly (conda export + pip freeze +
wheel-index URLs) so a fresh machine reproduces it. 30 min.

### 2.12 Docs & ops

#### P-10. Hygiene (LOW) — CLOSED 2026-09-26
No CI; `errors.log` protocol never implemented; `report.tex`
skeleton; thresholds scattered. **Fix:** CI, `pyproject.toml`,
central thresholds, docs refresh. 1 week background.

#### P-20. Config/docs/ops (LOW) — CLOSED 2026-09-26
`default.yaml` references deleted legacy files; relative paths
never loaded; requirements duplication (see RQ-3); stale README/
SETUP_GUIDE numbers; `usage.md` legacy template;
`run_evaluation.py` placeholder-only claims. Background week.

#### DC-1/DC-2. Spec + conventions docs (LOW) — CLOSED 2026-09-26
AGENTS.md Subdivision 3 still specifies deleted files; unit
conventions live in code comments only. **Fix:** Subdivision 3 ->
catalog reference; `docs/conventions.md`. 1.5 h.

#### IF-2 remainder. Ungenerated facts (MEDIUM) — CLOSED 2026-09-26
Done: oncoming, TTC, durations, sightlines, lane_of,
has_cleared. Left: `evasive_path`, `rolling_avg_speed`,
`conflict_point` inputs, `merge_gap_sequence`, `approaching`,
`oncoming_lane`, `crosses_oncoming_path` (R7.2/R8.3/R10.4/R11.2/
R15.1/R15.3 fire only via manual extras today). 4–6 h.

---

## 3. Done (do not re-add)

GAP-1–GAP-12 (§1 of prior revision) plus sessions S1–S5:
CR-1/3/4, CR-2 audit, P-12 blob fix, IF-3 weather wiring,
IF-1/RQ-1 central table + YAML params, map-aware corrector,
IF-2 core facts, MC-1 option C lane facts, RQ-2 projection,
P-13 fail-loud postprocess, T-5 golden harness, P-1 STEP_DT pin,
real inference (CPU+CUDA), pure-torch + SDPA fallbacks, synthetic
anchors, 10-dim slicing, MC-3 loader, `reset_temporal`,
calibration harness + v1 baseline, R2.3 motion guard, TG-2/3/4/5,
T-6/T-7 tests, DC-3 closure. Detail: `session.md`.

---

## 4. Moving forward (2026-09-26 plan — DONE, see session S10)

1. **Annotation-replay harness** — BUILT. 2. **Warm-by-default
   calibration** — DONE. 3. **Real-scene perf** — DONE (caps,
   `use_asp_in_lane` gate, tolerance parity). 4. **Parked-vs-live**
   — ANSWERED via series variance. 5. **CARLA** — server installed
   + smoke OK (S12): 0.9.15 at `C:\CARLA`, Town10HD_Opt default map;
   bridge/scenarios remain.

---

## 5. Outstanding (S13–S17 track, 2026-10-03)

> Constraint emission + QP correction landed 93->119 tests. This is the
> open tail of that track (session S13-S17 + day summary 2026-10-03).
> The month-run open carryovers remain in §2 (unmarked items).

### 5.1 Constraint / QP correction

#### P-21. GPU `--corrector qp` calibration (HIGH) — CLOSED 2026-10-21 (S18)
`run_calibration.py` has a `--corrector qp` flag and the pipeline a
`corrector_mode='qp'` arg, but both were only exercised GPU-free on
frozen `trajs.json` (S15). **Ran on GPU**: `calibration_qp.json` —
fix parity 0.50 = 0.50 (same 5 fixed scenes: 3,4,7,8,9), L2 10.184
identical (warm walk deterministic); corrector 3210.6 vs heuristic
2562.1 ms (mean dominated by the 5 fallback-exhausted give-ups),
checker 77.2 vs 59.9 ms (`#show`-scan cost), planner 14221.8 vs
13026.5 ms (run-level variance).

#### P-22. Warm + tracked-agent replay ablation for QP (HIGH) — CLOSED 2026-10-21 (S18)
`ablate_corrector.py` uses frozen cold predictions and CV-projected
agents. **Ran**: `replay_eval.py` gained the warm 5-frame walk +
`--corrector heuristic|qp` (same N-2 path as calibration). Warm A/B
on 184 tracked agents (92 moving / 92 parked), warm L2 10.18:
scene-level parity — same 5 fixed scenes (2,3,7,8,9) and the same 5
give-ups (0,1,4,5,6) — corrector 7435.4 vs 6747.5 ms
(`replay_warm_heuristic.json` vs `replay_warm_qp.json`). Verdict:
parity on real warm planner output + real agent histories; the QP
latency edge stays dump-side (pure lane cases only).

#### P-23. Horizon-joint / stale-constraint solving (HIGH) — IN PROGRESS (increment 2 done 2026-10-23, S25; increment 3 done 2026-10-23, S27)
`qp_ablation.json` scene 8 fails at 3/5/10 rounds: static per-timestep
boxes whack-a-mole a moving CV-projected threat while tail coupling
cascades new violations. **Fix:** solve jointly over the horizon with
agent-motion rollouts (or re-derive boxes per outer round) instead of
one-shot static bounds; revisit A7 stale constraints. 1–2 weeks.
— INCREMENT 1 SHIPPED (S25): `ConstraintOptimizer` now alternates the
box/speed/accel projections to JOINT feasibility (`joint_iters`,
`feas_tol_cm`, honest `projection_cycle` status) with 2 new regression
tests; suite 120 -> 122 green; flake8-clean. Measured: aggregate rates
unchanged (0.375 = 0.375, caps 0.00, boxes 0.125) and pure-QP 1/8
before = after — instrumented scene 8 shows all rounds `optimal`
(jointly feasible vs emitted bounds) yet re-validation still fails,
so the ceiling is EMISSION COVERAGE, not projection convergence.
Evidence: `experiments/qp_ablation_p23.json`, design + measurement
ledger in `docs/implementation_plan_p23.md`.
— INCREMENT 2 SHIPPED (S26): `src/constraint_optimizer.py` moving box
projection (`_project_moving_boxes` + `_index_box` moving-box
detection). Implemented coupled projection for consecutive y_bounds
with linear center motion. Suite 122 green; flake8-clean. Measured:
aggregate rates UNCHANGED (heuristic 0.375 = QP 0.375, caps 0.00,
boxes 0.00). Pure-QP (no fallback) on 8 unsafe scenes: 1/8 at 3r,
**2/8 at 10r** (scenes 0, 7 fixed at 10r) — improvement at higher
rounds. Accumulated bounds still hit `projection_cycle` → ceiling
is EMISSION COVERAGE. Next: coupled/range emission in `rules/*.lp` +
`src/checker.py` `_group_constraints` (range constraints). Evidence:
`experiments/qp_ablation_p23_inc2.json`, design + measurement ledger
in `docs/implementation_plan_p23.md`.
— INCREMENT 3 SHIPPED (S27): `rules/collision_avoidance.lp` emits
`constraint_y_bounds_range/5` (lateral corridor via `#min`/`#max`);
`src/checker.py` `_group_constraints` parses `y_bounds_range`;
`src/constraint_optimizer.py` adds `_index_range` + `_project_ranges`
for coupled range projection. Suite 122 green; flake8-clean.
Measured: aggregate UNCHANGED (heuristic 0.375 = QP 0.375, caps 0.00,
boxes 0.00). Pure-QP (no fallback) on 8 unsafe scenes: **0/8 at 3r,
2/8 at 10r** (scenes 0, 7 fixed at 10r) — improvement at 10r but
regression at 3r (scene 7 regressed). Accumulated bounds hit
`projection_cycle` → ceiling is EMISSION COVERAGE. Next: refine
range projection or emit coupled corridor constraints directly from
rules. Evidence: `experiments/qp_ablation_p23_final.json`, design +
measurement ledger in `docs/implementation_plan_p23.md`.

### 5.2 Evaluation / evidence

#### P-24. `report.tex` rewrite (HIGH) — CLOSED 2026-10-21 (S18)
The 5-rule skeleton is replaced: Method (architecture, integer-cm
conventions, 63-rule catalog table with priorities, constraint
emission incl. one-sided hold-back, both correctors), Experiments
with measured tables from the frozen JSONs (B0 cold/warm L2, warm
shield rates + tracked replay + parked finding, bound-family
ablation + 12-combo sweep null, P-21/P-22 warm A/Bs), attribution
caveats and limitations (horizon 3 s, CV context, open-loop gap,
P-23/P-25 pointed as future work).

#### P-25. CARLA bridge + scripted scenarios + closed-loop metrics (HIGH) — CLOSED 2026-10-22 (S19–S22)
Full arc executed per the frozen `nextstep.md` plan (P25-A rig probe;
P25-B bridge + S1 exit criteria; P25-C shield dispatch + metrics; P25-D
S2–S5 scenarios + 15-run pilot matrix).
- Bridge: `experiments/carla_bridge.py` (camera rig + planner wrapper +
  lane-plan fallback + checker/corrector dispatch + control loop +
  run JSON), `experiments/run_closed_loop.py`, scenario classes
  S1–S5 in `experiments/carla_scenarios.py`, context builders in
  `experiments/carla_context.py`, aggregator `experiments/carla_metrics.py`.
- Frozen 15-run matrix (5 scenarios x off/heuristic/qp, seed 1,
  Town10HD_Opt) under `experiments/runs/` + frozen
  `experiments/carla_metrics.json`: collision rate 13.3% (only the S3
  cut-in and S5 turn-conflict off arms collide; both shielded arms
  survive), frame drops 0, give-up 0 on all arms, per-target evidence:
  S2 shield holds at a red junction (the off arm runs it), S3/S5 shields
  survive the scripted conflicts, S4 shield waits out the jaywalker.
- Deviations ledger: `docs/carla_harness.md` (axis swap, image flip,
  junction-overrun fix + plan-extension, light-tipology quirks + cluster
  radius/pinning, longitudinal_clearance -> follow remap,
  on_failure='brake', actor sweep, hold-aware stall detector,
  kinematic-VRU collision limitation, sync-mode replan wall pauses).
- Known limits: kinematic NPCs/VRUs do not generate physical collisions
  against the physics ego; corrector quality is bounded (P-23_STATIC
  whack-a-mole persists in QP; heuristic got sign-blinded fixes
  documented); replan wall (3.4-5.5 s) on adversarial geometry is
  sim-time-frozen (sync mode).

### 5.3 Testing / CI

#### T-14. CI omits S13–S17 suites (MEDIUM) — CLOSED 2026-10-21
`.github/workflows/fast-tests.yml` now runs 10 suites (checker,
checker_edges, constraint_emission, corrector, data_loader,
evaluation, performance, qp_feasibility, rules, t2_coverage); the
torch-needing suites (pipeline, sparse_drive, test_all full-run)
stay excluded by design. Header comment refreshed.

#### RQ-6. `test_all.py` collects zero tests under pytest (LOW) — CLOSED 2026-10-21
`tests/test_all.py` gained `def test_all_components():` (calls
`run_all_tests()`) and is CWD-independent (`ROOT` = tests-dir parent;
all paths derived from it). pytest collects the runner; the script
entry point still works.

### 5.4 Docs / ops

#### N-5. Doc staleness after S13–S17 (LOW) — CLOSED 2026-10-21 (S18)
All targets refreshed to the 120-test S17 state: README (counts,
tests/experiments/src lists, warm-status section, date), whatodo
header, test_coverage headline + constraint/QP rows + current gaps,
testdoc (§1 pytest entry, §12 warm default + `--corrector`, new §13
constraint-emission + §14 QP suites, quick-ref rows),
evaluation_design + IMPLEMENTATION_PLAN shipped-status banners,
api_reference.md + replay_eval_design.md (S13–S17 APIs; harness
status + corrected measured table), default.yaml (~986 MB ckpt +
clingo 5.8.2 drift notes). Remaining open in docs: none tracked.

#### DC-5. Dependency conflict (LOW) — CLOSED 2026-10-21 (S18)
`docs/environment.md` gained the DC-5 divergence section (upstream
spec pins torch 1.13.0+cu116 / mmcv_full 1.7.1 vs working torch
2.1.0+cu118 / mmcv-full 1.7.2 + mmdet 2.28.2, mmdet3d excluded;
upstream left untouched by design); `SETUP_GUIDE.md` Step 7
troubleshooting now installs `mmcv-full==1.7.2` from the
`torch2.1.0` index and marks Step 8's compile as skippable via the
no-compile fallbacks (pure-torch deformable + SDPA); `default.yaml`
notes the ~986 MB checkpoint and the clingo-dev-shell drift.

### 5.5 Visible rendering (S23)

#### P-26. Epic-quality visible CARLA window (LOW) — SCRAPPED TO BACKLOG 2026-10-22 (S23)
User asked to watch the sim live. Research: CARLA 0.9.15 ships exactly
two quality tiers (Low, Epic); there is no intermediate tier (that is
0.10+). Built the visible-window stack on Low: `carla_launch.py`
`--windowed/--quality/--stream-pool` (window 1280x720, replaces
`-RenderOffScreen`), `experiments/carla_vram_probe.py` (server+window+
rig -> planner -> warm `predict_frame` vs the >=1.5 GB gate; Low
windowed PASS at 4.03 GB free), ego-follow spectator
(`BridgeConfig.view_ego`, `run_closed_loop.py --view`). Live demo pair
in `experiments/runs/_view_demo/` (S3 off collides at sim 10.15 s;
S3 heuristic survives, give_up 0.0).
**Scrapped**: Epic is infeasible on this rig — four boot crashes at
map load (Epic windowed x2, Epic windowed + `--stream-pool 1200`,
Epic offscreen; the offscreen one lived to ~110 s). Not a VRAM
problem: one attempt ran with 5.9 GB free, so it is a packaging /
driver crash, not the shared-GPU budget. Windowed Low is the
validated config.
**Driver retry 2026-10-23 (617.42)**: new driver changes the failure
mode but not the verdict. Epic windowed now survives the early boot
crash (alive ~2-3 min, port open 30 s) but dies during full map load
with free VRAM down to 65 MiB — the complete Epic texture/shadow set
needs more than the 6 GB card. Epic windowed +
`r.Streaming.PoolSize 1200` died within 100 s (the trim comes too
late in the load order). So the original "not a VRAM problem" was
wrong in part: with a healthy driver, 6 GB becomes the binding
constraint at full map load. Gated fallback re-ran to Low windowed
(server up, demos re-run OK).
**Re-entry condition**: GPU with >=8 GB VRAM (12+ GB if upgrading
CARLA to 0.10 anyway), or a map build that streams lighter at Epic.
The `--quality Epic` flag stays plumbed; if Epic ever boots, the gated
fallback chain (trim -> Low) still applies.
Evidence: `session.md` S23, `docs/carla_harness.md` §Visible window.

# WhatoDo — ASP Safety Layer, full-month run

> Status: P-25 CARLA closed-loop arc COMPLETE (2026-10-22, S19–S22):
> 15-run pilot matrix + frozen metrics; deviations ledger in
> docs/carla_harness.md. Visible-window arc done 2026-10-22 (S23):
> windowed Low + ego-follow spectator live; Epic rendering scrapped
> to backlog as P-26. ASP-constrained correction track complete
> (S13-S17). Closure session ran 2026-10-21. Remaining open ledger:
> P-23 horizon-joint solving + P-26 (Epic window, backlogged).
> Official baseline: GT 0.3 / pred 0.2 / fixed 0.5 / L2 10.18 (warm;
> experiments/calibration_baseline.json; cold was GT 0.5 / pred 0.1 /
> fixed 0.7 / L2 11.54; pre-agent v1 was GT 0.90 / pred 0.00 /
> fixed 1.00). Closed-loop CARLA: collision rate 13.3% off vs 0%
> shielded on the two colliding scenarios (experiments/carla_metrics.json).
> Canonical env: conda `asp-safety` (py3.10, torch 2.1.0+cu118, clingo 5.7.1). Data: `C:/data/sets/nuscenes`.
> Sources of truth: `session.md` (what's done), `futuretodoist.md` (backlog IDs),
> `experiments/*.json` + verdict `.md` files (measured numbers).
> Verify everything with: `pytest` (120), `python tests/test_all.py` (7/7),
> `python tests/test_rules.py` (14), perf budgets (bare <50 ms, 20-agent <100 ms).

## Operating rules for agents

- [ ] Work items in order; check the box only when its Verify step passes.
- [ ] Never invent map/signal facts — honest `{}`/absence over placeholders.
- [ ] Integer-cm, squared-distance discipline stays (no floats/sqrt in `.lp`).
- [ ] Every behavior change ships with a firing test + a safe-counterpart test.
- [ ] Regenerate `calibration_baseline.json` after Tracks 2–4 and record the delta.

## Horizon decision N-3 (locked for this run)

- [ ] Model = 3 s @ 0.5 s; checker = 6 x 1.0 s; harness resamples to 3 x 1 s (blind past 3 s, plus a 1 s label shift: pred[1] asserted as T=0).
- [ ] Week 1: accept 3 s + fix the shift (T=0 anchor from current ego pose, assert horizon length, stamp `horizon_s: 3` on all results).
- [ ] Weeks 2-3: extrapolation as a labeled experiment (`extras: {extrapolate: true}`, uncertainty-grown thresholds), swept both ways (3 s strict vs 3+3 s); keep only if signal > noise.
- [ ] Week 4: time-boxed spike (2 days max) on extending the model head; kill on no-joy, no retrain rabbit holes.
- [ ] Paper claims: 3 s verified shield + 6 s GT-audited rule set, extrapolation as an ablation row.

## TODAY (Day 1 — unblock, ~2 h)

Done: T-13, RQ-3/N-4 (+docs/environment.md), ops (already allow),
T-12, T-3 (assert-style + CI at .github/workflows/fast-tests.yml).

- [ ] T-13 (10 min): fix `docs/usage.md:177` (`TrajectoryChecker(device=)` doesn't exist) + stale Next Steps. Verify: grep.
- [ ] RQ-3 + N-4 (1 h): unify root vs `asp-safety/requirements.txt`; write `docs/environment.md` (conda export + pip freeze + mmcv index URLs, numpy 1.26.4, setuptools 75.8.0, mmdet 2.28.2, no mmdet3d, opencv 4.8.1.78). Verify: fresh-machine note.
- [ ] Ops unblock (15 min): `opencode.json` `external_directory` deny -> allow. Verify: read `C:/data/sets/nuscenes/v1.0-mini/scene.json`.
- [ ] T-12 (20 min): determinism asserts (seed-42 placeholder, per-scene loader seeds). Verify: pytest green.
- [ ] T-3 start: convert bool-return tests to asserts; open CI workflow draft.

## WEEK 1 — honesty + coverage foundation

Done 2026-09-26:
- N-3(a): anchored T=0 pred path, horizon_s=3 stamped, fresh 10-scene
  baseline matches v1 (L2 11.54, GT 0.90, pred 0.00/fixed 1.00).
- En-route fix: consecutive real inferences crashed (KeyError
  'img_metas') -- predict() cold-resets; warm API is N-2.
- P-13: nav_cmd/ego_status plumbing + sweep (weak conditioning, see
  experiments/cmd_sweep_results.md); mapping unverified, deferred to N-2.
- N-2: get_scene_frames + begin/predict_frame/end + estimate_ego_status;
  warm wins 10/10, 11.54 -> 10.18 m (experiments/temporal_results.md).
- P-15: dead self.control/legacy wrappers removed; explicit-times +
  aliasing pins (test_checker_edges). active_times/arity collectors
  were already correct -- verified, not changed.
- Perf note: 20-agent gate (~90-95 ms vs 100 ms budget) flakes under
  machine load; passes standalone/CI. Real headroom is P-8/PR-3.

- [ ] N-3 (a): horizon-explicit pred path (T=0 anchor, length assert, `horizon_s` stamp) in `run_calibration.py` + `STEP_DT` docs + `pipeline.py` docstring. Verify: calibration JSON carries horizon; GT-vs-pred comparable.
- [ ] P-13 planner remainder (HIGH, 2-3 h): real nav command + `ego_status` (`src/neural_planner.py`; today straight `[1,0,0]` + zeros). Verify: L2 drops from 11.54 m.
- [ ] N-2 temporal harness (HIGH, 4-6 h + compute): warm-queue multi-frame walk (no per-frame `reset_temporal()`), then measure. Verify: pred-side L2 meaningful.
- [ ] T-3 finish (HIGH): pure-assert tests + CI running (pytest + test_all + test_rules + perf).
- [ ] P-15 checker remainder (MED, 3-4 h): honor `active_times`/`times`, temporal guard/worker facts, arity-safe collectors, drop dead `self.control` + legacy wrappers, pin check/copy aliasing. Verify: `test_checker_edges.py` extended, green.
- [ ] Gate W1: pytest + 7/7 + 13/13 + perf budgets + fresh calibration JSON.

## WEEKS 2-3 — geometry, facts, coverage, sweep

Done 2026-09-26 (76 tests green: 64 fast + 8 sparse/all + 4 pipeline):
- P-4: longitudinal_gap (R4.2), velocity-projection closing_rate,
  approaching_zone-gated about_to_enter (+stopline fallback for holds),
  stop-line-relative stopped_before, stopped(V,0) edge, R2.1
  lane_transition exception. Yielding_maneuver intentionally unchanged.
- IF-2: rolling_avg_speed, conflict_points/oncoming_lanes map keys,
  merge_gap_sequence+mainline heuristic, approaching(V,ego),
  occupancy/current_time builders, R11/R6.4/R7.2 manual paths tested.
  crosses_oncoming_path unused (logged). roundabout added to
  conflict_zone (R6.2 was unfireable). obstacle_in_box covers peds.
- T-2/P-9: tests/test_t2_coverage.py (16 fns), registry 73/73,
  R13.3 arbitration + R11 handoff pinned, include_internal atoms
  (ttc/threat/comfort/future/any) for tuning + P-6.
- T-8: min-boost accordion + blind-boost fixed (targeted tail
  stretch); converges cleanly.
- Real bugs fixed en route: R3.4 square overflow (severe swings never
  fired), R15.4 square overflow + steady-lead FP, #abs invalid in
  clingo 5.7 (broke whole catalog until sign-split rewrite).
- N-3(b): extrapolation DROPPED (20 artifact hits, fix 1.0->0.9).
- N-1: real map extraction (discretized lanes+connectors, widths,
  directions, intersections, tiled crossings/walkways, ego-frame
  transform). road_type/limits honestly omitted (unknown) -> default
  cap gated on bare trajs only. GT residuals ledger: scene-1 off-map
  start, scene-8 edge departure (genuine observations, not FPs).
- P-5: GPU-free sweep; min_score 0.3->0.5 (knee), rest flat.
  P-18: velocities kept, boxes->CV agent trajs in pipeline.run +
  calibration (score>=0.5, cap 20). 300 raw detections/frame noted.
- Perf (P-8 head start): demand gates + one-shot fact batching +
  #show atom scan: 20-agent 109->37 ms. Real scenes 169 ms (map
  tiles) + corrector 5.7 s under agents -> Week-4 P-6/P-8 work.
- Official baseline: GT 0.5 / pred 0.1 / fixed 0.7 / L2 11.54
  (conservative layer + noisy cold predictions; ledger in
  experiments/sweep_verdict.md).

- [x] P-4 geometry: longitudinal_gap, heading-gated about_to_enter (+stopline fallback), stop-line-relative stopped_before, velocity-projection closing_rate, R2.1 transition, stopped(V,0). Verified: 3 new edge tests + calibration re-run.
- [x] IF-2 remainder: rolling_avg_speed, conflict_points/oncoming_lanes keys, merge slots heuristic, approaching(V,ego), occupancy/current_time builders; evasive/hazard stay manual (tested); crosses_oncoming_path dropped (unused).
- [x] T-2 + P-9 coverage: tests/test_t2_coverage.py (16 fns), registry 73/73 green, R13.3 arbitration + R11 handoff pinned, include_internal atoms added.
- [x] T-8: min-boost convergence test + targeted tail stretch (no oscillation).
- [x] N-3 (b): labeled extrapolation + dual sweep -> DROP (experiments/extrapolation_verdict.md).
- [x] N-1 map data: real extraction (no posted limits exist in nuScenes data — honestly omitted); scene-7 urban FP gone; residuals ledgered.
- [x] P-5 sweep: GPU-free harness; min_score 0.3->0.5 knee; rest flat (experiments/sweep_verdict.md).
- [x] DC-4 + P-19: docs/rule_divergences.md (52 items with dispositions).
- [x] Gate W2-3: registry green, dual-horizon table (strict kept), updated baseline (GT 0.5 / pred 0.1 / fixed 0.7).

## WEEK 4 — corrector, pipeline, perf, spike, paper prep

Done 2026-09-26 (92 tests green):
- P-6/P-16: group_cap 40, dirty-set, stagnation break, brake fallback
  + pipeline on_failure passthrough, tail-rigid collision shift,
  duck-typed checker guard. 5 new corrector tests.
- T-9/P-14, T-10/T-11, P-17 (yaw/FDR/validation), P-18 (timing_ms,
  9-key error schema). All tested.
- PR/P-8: conflict culling + demand gates + batcher + #show:
  20-agent 109->37 ms. Multi-shot deferred (dynamics dominate).
- N-3(c) KILLED (retrain needed). RQ-4 splits + annotations
  (incl. fixing a double-solve from the surgery; no-duplicates test).
- Track 5: conventions.md, data_contracts refresh, default.yaml
  notes, run_evaluation scope, AGENTS Subdiv-3 banner, pyproject,
  testdoc refresh, replay_eval_design.md + report table.
- Deferred with reasons: multi-shot, parked-vs-live classes,
  real-scene perf headroom, CARLA bridge (install+smoke done S12), report.tex rewrite.

- [x] P-6 + P-16 corrector: group caps, dirty-set, stagnation break, brake fallback, tail shifts, duck guard. Verified: 5 new tests + idempotence + RuntimeError path.
- [x] T-8 (done W2-3), T-9 + P-14: hostile + reachable-red placeholder scenes, fire + correct clean.
- [x] T-10, T-11: real-output metrics tests; pipeline error-path tests (placeholder-fast).
- [x] P-17 metrics: yaw-aware collision, zip fix, matching keys, FDR rename, NaN/inf/shape validation.
- [x] P-18 batch/schema: boxes->agents (W2-3) + per-stage timing_ms + 9-key error schema.
- [x] PR-1/PR-2 + P-8/PR-3 perf: demand gates, fact batcher, #show scan (109->37 ms); conflict-aware culling + stagnation + group caps. Full multi-shot deferred (dynamics dominate). Budgets hold.
- [x] N-3 (c) spike: KILLED with evidence (plan_reg (12,256) baked).
- [x] RQ-4/RQ-5 refactor (as time allowed): _cull_agents + _shown_atom_to_dict extracted, annotations; _apply_correction split deferred (small, single-dispatch).
- [x] Track 5 docs: pyproject (no pytest section by design), default.yaml notes, run_evaluation scope, AGENTS Subdiv-3 banner, docs/conventions.md, data_contracts refresh, testdoc refresh.
- [x] P-11 prep: replay_eval_design.md (annotation-replay + CARLA) + measured report table; report.tex rewrite + harness build remain.
- [x] Gate W4: 92 green, month-end calibration JSON (GT 0.5 / pred 0.1 / fixed 0.7), paper-numbers drafted.

## Moving forward (2026-09-26 plan) — DONE 2026-09-26, see session S10

1. **Annotation-replay harness** — BUILT (`replay_eval.py`):
   184 tracked (92/92), parked explains ~60% collisions. DONE.
2. **Warm-by-default calibration** — DONE (`--warm` default,
   `--cold` reproduces; warm baseline GT 0.3/pred 0.2/fixed 0.5).
3. **Real-scene perf** — DONE: tile caps + constants, `use_asp_in_lane`
   gate, end-tolerance parity fix; ~100 ms real-scene checks.
4. **Parked-vs-live distinction** — ANSWERED via series variance
   (no contract change needed).
5. **CARLA closed-loop** — BLOCKED (no package/server; needs
   provisioning). Design doc updated.

## Backlog map (every futuretodoist item, where it lives above)

| ID | Item | Slot |
|---|---|---|
| T-2 | rule firing coverage | W2-3 |
| T-3 | asserts + CI | Day 1 + W1 |
| T-8 | min-speed oscillation test | done W2-3 (test + targeted fix) |
| T-9 / P-14 | hostile + reachable-red scenes | done W4 |
| T-10 | real-output metrics tests | done W4 |
| T-11 | pipeline error paths | done W4 |
| T-12 | determinism asserts | Day 1 |
| T-13 | usage.md fix | Day 1 |
| P-9 | registry + R12.4 + override audit | W2-3 |
| P-19 / DC-4 | divergences | W2-3 |
| P-4 | geometry | W2-3 |
| N-1 | map limits/types | W2-3 |
| P-5 | threshold sweep | W2-3 |
| N-2 | temporal inference | W1 |
| N-3 | horizon (a->b->c) | W1 / W2-3 / W4 |
| P-11 | closed-loop + report | design + numbers done; CARLA installed + smoke OK (S12), bridge beyond |
| P-6 / P-16 | corrector guarantees/realism | done W4 |
| P-15 | checker remainder | W1 |
| P-13 | planner cmd/ego-state | W1 |
| P-17 / P-18 | metrics / batch schema | done W4 |
| PR-1/PR-2, P-8/PR-3 | solving + culling perf | done W4 (multi-shot deferred w/ reasons) |
| RQ-3 / N-4 | requirements + env record | Day 1 |
| RQ-4/RQ-5 | splits + types | W4 if time |
| P-10 / P-20 / DC-1/DC-2 | hygiene, config/docs, conventions | W4 background |
| IF-2 rem | ungenerated facts | W2-3 |

## Done (do not re-add)

GAP-1-12; S1-S5 (blob fix, weather wiring, strategy table, derived facts, projection fix, fail-loud postprocess, STEP_DT pin, CPU+CUDA inference, real loader, in_lane_fact, reset_temporal, calibration v1, R2.3 guard, TG-2/3/4/5, T-6/T-7, DC-3). Detail: `session.md`.

---

## P1 constraint emission — DONE 2026-10-03 (see session S13)

- All 15 rule files emit constraint atoms (6 heads); checker
  `include_constraints` path + `tests/test_constraint_emission.py`
  (11 green). 104 tests green total, heuristic behavior unchanged.
- Next: P3 QP optimizer (projected-gradient MVP) + P4 CorrectorQP +
  P6 ablation on trajs.json. P2 checker extension shipped inside P1.
  See docs/IMPLEMENTATION_PLAN_ASP_CONSTRAINTS.md.

---

## P3/P4 constrained correction — DONE 2026-10-03 (see session S14)

- `src/constraint_optimizer.py` (projected-gradient MVP, T0 pinned) +
  `CorrectorQP` in `src/corrector.py` (3 solve/re-validate rounds,
  heuristic fallback with flagged head entry) +
  `tests/test_qp_feasibility.py` (10 green). 114 tests green total.
  Heuristic still default; no pipeline wiring.
- Next: pipeline `mode` flag + `--corrector` calibration option, then
  P6 ablation (full vs speed-only vs collision-only vs none) and
  real-scene QP latency. See docs/IMPLEMENTATION_PLAN_ASP_CONSTRAINTS.md.

---

## Pipeline QP mode + P6-lite — DONE 2026-10-03 (see session S15)

- `corrector_mode` flag on `SafetyLayerPipeline` (+ result stamp,
  + error-schema parity, + mode test); `--corrector` on
  `run_calibration.py` (wired, stamped; GPU run stays a laptop task).
  115 tests green, heuristic default everywhere.
- `experiments/ablate_corrector.py` + `qp_ablation.json`: 8/10 unsafe
  frozen scenes, fix parity 0.38/0.38, QP mean latency slightly lower
  (2536 vs 2683 ms), scene-7 10x faster. Both give up on the same 5
  re-route cases.
- Next: GPU `--corrector qp` calibration on the laptop, warm-data +
  tracked-replay ablation, A2/A3 constraint subsets.

---

## Bound-family ablation — DONE 2026-10-03 (see session S16)

- `CorrectorQP(bounds=full|caps|boxes)` + ablation variants (fallback
  off for purity) + 3 bounds tests. 118 green. Full 0.38, caps 0.00,
  boxes 0.12 on 8 unsafe frozen scenes -- joint solve required, no
  family sufficient alone. Evidence: `experiments/qp_ablation.json`.
- Next: GPU `--corrector qp` calibration (laptop), warm-data +
  tracked-replay ablation. A-loop on frozen data is done.

---

## QP sweep + hold-back fix — DONE 2026-10-03 (see session S17)

- Param sweep flat (1/8 all 12 combos): defaults kept.
  Attribution correction: S15 "qp 0.38" = 1 pure fix + 2 fallback
  completions; pure-QP edge is latency (scene 7), not rate.
- Centered clearance boxes were optimizer attractors -- fixed to
  one-sided hold-back (R4.2/R3.2/R15.2) with exact-tuple tests.
  Dump fix rates unchanged (moving-threat + tail-coupling needs
  horizon-joint solving; filed as P6-full). 119 green.
- Next: GPU `--corrector qp` calibration (laptop), warm tracked
  replay, horizon-joint/A7 work.

---

## Day log 2026-10-03 (S13-S17)

ASP-constrained correction end to end: emission (P1) -> optimizer +
CorrectorQP (P3/P4) -> pipeline/calibration wiring -> family ablation
-> param sweep -> hold-back fix. 93 -> 119 tests green, heuristic
default everywhere, verdicts parity (rate) with QP latency edge
(scene 7). Full ledger: `session.md` day summary 2026-10-03.

---

## Day log 2026-10-21 (S18 — closure session)

Closed the open-ledger items: T-14/RQ-6 (CI + pytest entry) verified,
N-5 doc refresh (README/whatodo/test_coverage/testdoc + DC-5 env
docs + shipped banners), P-21/P-22 GPU runs (warm corrector A/Bs,
fix-rate parity 0.50 = 0.50 both protocols), P-24 report rewrite
(report.tex, measured tables), futuretodoist §5 closure with
P-23/P-25 dispositions kept open. 120 green + test_all 7/7.
Full ledger: session.md S18.

## Day log 2026-10-22 (S19 -- P25-A)

Full P-25 arc started (approved plan frozen in nextstep.md; unattended
run). P25-A probe DONE: carla_launch.py gained --map/--kill;
carla_probe.py verified the 6-cam rig, K-from-fov intrinsics and the
C_i = S @ T_v_c @ F coordinate chain by projection sanity (12 m actor:
LEFT u=246.7 < cx=352 < RIGHT u=457.3, in-image), VRAM headroom 4.03
GB after a warm predict_frame (>=1.5 GB gate OK). Artifacts:
experiments/probe_out/{probe_calib.npz,cam_front.png}. Full ledger:
session.md S19.

## Day log 2026-10-22 (S20–S22 — P25-B/C/D, arc complete)

Bridge v1 landed and iterated to the S1 exit criteria; P25-C shield
dispatch + carla_metrics.py: heuristic/qp arms verified on S1;
P25-D: S2–S5 scenario classes, light-facts debugging (trigger_volume
fix, cluster radius redefinition, group pinning), choreography
re-parameterizations (S3 cut-in, S4 walker, S5 turn), corrector
dispatch fixes (longitudinal_clearance -> follow remap,
on_failure='brake', lateral_shift sign), harness hardening (actor
sweep, hold-aware stall detector). Frozen 15-run pilot matrix +
carla_metrics.json: collision rate 13.3% (shielded arms survive the
  two colliding off arms), frame drops 0, give-up 0 throughout.
  Deviations ledger: docs/carla_harness.md. Ledger: session.md S20–S22.

## Day log 2026-10-22 (S23 — visible window + Epic verdict)

User wanted to watch the sim live. Researched the quality tiers
(0.9.15 = Low/Epic only), added `--quality/--windowed/--stream-pool`
to carla_launch.py, built experiments/carla_vram_probe.py (VRAM gate
>=1.5 GB), and added the ego-follow spectator (`--view`,
`BridgeConfig.view_ego`). **Epic is infeasible on this rig**: four boot
crashes at map load regardless of VRAM (5.9 GB free on one attempt) —
packaging/driver, not budget. Fallback to windowed Low validated by the
probe (4.03 GB post-predict). Demo pair in experiments/runs/_view_demo/:
S3 off collided live at sim 10.15 s; S3 heuristic survives (wall_cap,
0.025 km, 0 give-ups) — same shape as the frozen matrix.
Full ledger: session.md S23.

## Day log 2026-10-23 (S24 — P-25 closeout completion + Epic driver retry)

Closed the P-25 tail: nextstep §8 verification gates all recorded
(test_all 7/7 PASS, pytest 120 passed 145.48 s server-off; windowed
Low relaunched); session.md S23 dangling reference fixed; D3 rows in
docs/evaluation_design_asp_correction.md flipped from TBD/blocked to
implemented (metrics pointer + caveats); docs/carla_harness.md gained
the §8.1 sections (architecture/cadence, fact mapping, frozen metric
definitions, per-scenario verdict table verified against the frozen
JSON, caveats). Epic driver retry (617.42): early boot crash fixed but
full map load still dies at 65 MiB free VRAM — 6 GB is the binding
constraint; P-26 stays backlogged (re-entry: >=8 GB GPU). Demo pair
re-ran at windowed Low, same outcomes. Full ledger: session.md S24.

Next (master plan approved 2026-10-23): P-23 horizon-joint solving
(design plan first), then multi-seed CARLA matrix (seeds 2-3, 45-row),
then P-13 remainder.

## Day log 2026-10-23 (S25 — P-23 increment 1)

P-23 entry gate: design plan in docs/implementation_plan_p23.md
(more-rounds ruled out by the 3/5/10 evidence). Shipped the joint
projection loop in src/constraint_optimizer.py (alternate box/speed/
accel to joint feasibility; honest `projection_cycle` status) with 2
regression tests; suite 120 -> 122 green. Measured honest outcome:
aggregate rates unchanged (0.375 = 0.375) and pure-QP 1/8 before =
after — instrumented scene 8 shows the solve IS jointly feasible
every round, so the ceiling is constraint-emission coverage, not
projection convergence. Increment 2 (coupled/range emission) is next.
Operational: perf budget tests need the CARLA server OFF (windowed
GPU load trips the 50 ms budget). Full ledger: session.md S25.

## Day log 2026-10-23 (S26 — P-23 increment 2)

P-23 increment 2: moving box projection in src/constraint_optimizer.py
(detect consecutive y_bounds with linear center motion; project the
entire segment as rigid body + lateral offset). Suite 122 green
(120 + 2 new tests). Measured: aggregate ablation UNCHANGED
(heuristic 0.375 = QP 0.375, caps 0.00, boxes 0.00). Pure-QP
(no fallback) on 8 unsafe scenes: 1/8 at 3 rounds, **2/8 at 10
rounds** (scenes 0, 7 fixed at 10r) — improvement at higher round
counts. However, accumulated bounds hit `projection_cycle` at round
2+ → ceiling is EMISSION COVERAGE, not projection convergence.
Next: coupled/range emission in rules/*.lp + checker.py
_group_constraints (range/corridor constraints). Full ledger:

## Day log 2026-10-23 (S27 — P-23 increment 3)

P-23 increment 3: range constraint emission in rules/collision_avoidance.lp
(constraint_y_bounds_range/5 for lateral corridors via #min/#max
aggregates), checker.py (_group_constraints parses y_bounds_range),
constraint_optimizer.py (_index_range + _project_ranges). Suite 122 green
(120 + 2 new tests). Measured: aggregate ablation UNCHANGED
(heuristic 0.375 = QP 0.375, caps 0.00, boxes 0.00). Pure-QP
(no fallback) on 8 unsafe scenes: **0/8 at 3 rounds, 2/8 at 10
rounds** (scenes 0, 7 fixed at 10r) — improvement at higher rounds
but regression at 3 rounds (scene 7 regressed). Accumulated bounds
hit projection_cycle at round 2+ → ceiling is EMISSION COVERAGE.
Next: multi-seed CARLA matrix (seeds 2-3, 45-row), then P-13
remainder. Full ledger: session.md S27.
session.md S26.

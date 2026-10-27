# Data Contracts & Function Table

## 1. Master data-flow table

| Stage | Producer | Consumer | Object | Type | Shape | Meaning | Example |
|---|---|---|---|---|---|---|---|
| sample | `data_loader.get_scene_sample` | `pipeline.run` | `images` | ndarray uint8 | (6,3,256,704) | 6 cams FRONT/FL/FR/BACK/BL/BR | zeros/placeholder |
| sample | loader | pipeline | `intrinsics` | f32 | (6,3,3) | fx=fy=400,cx=352,cy=128 | eye-based |
| sample | loader | pipeline | `extrinsics` | f32 | (6,4,4) | cam-to-vehicle | eye |
| sample | loader | pipeline | `ego2global` | f32 | (4,4) | vehicle→global, x=idx*10 | eye |
| sample | loader | pipeline | `timestamp` | float | — | unix | 1234567890+idx*.5 |
| context | `get_map_data` | `checker` | `map_data` | dict | — | real lanes/widths/dirs + intersections + tiled crossings/walkways, ego-frame; NO road_type/limits (unknown) | N-1 |
| context | `get_traffic_lights` | `checker` | `traffic_lights` | dict | — | int_0 red times 0..5 (placeholder); {} on real data | honest no-knowledge |
| proposal | `planner.predict` | `pipeline.run` | `trajectory` | f32 | (6,2) | ego m, t=.5..3s | x=[2.5..15] |
| proposal | planner | pipeline | `boxes` | f32 | (N,7) | [x,y,z,w,l,h,yaw] | 2 boxes |
| proposal | planner | pipeline | `scores` | f32 | (N,) | [0,1] | [.95,.87] |
| proposal | planner | pipeline | `velocities` | f32 | (N,3) | [vx,vy,vz] planar estimates (P-18) | zeros placeholder |
| agents | `pipeline._detections_to_trajectories` | `checker` | `other_vehicles` | list of (H,2) | — | CV projections, score>=0.5, cap 20 (P-18/P-5) | filtered |
| facts | `checker._add_*` (one-shot `_FactBatcher`) | clingo | `pos(ego,T,X,Y).` etc. | str | — | cm ints (×100); gap=Euclidean, longitudinal=signed, closing=velocity-projection | `pos(ego,0,250,0).` |
| answer | clingo (`#show` block) | `on_model` | shown symbols | Model | — | violation/hazard/warning/insufficient (+ttc/threat/comfort/future/any on request) | `violation(red_light,ego,5)` |
| verdict | `checker.check` | pipeline/corrector | `(is_safe, violations)` | bool, List[Dict] | — | warnings + internals excluded; `include_internal=True` appends ttc/threat/comfort/future/any | `False, [{rule,time…}]` |
| fix | `corrector.correct` | pipeline | `(corrected, applied)` | (6,2), List[Dict] | — | dirty-set + group caps + stagnation break; 'raise' or flagged 'brake' hold | hold at T5 |
| result | `pipeline.run` | caller/batch | result dict | dict | 12 keys | both trajs + meta + timing_ms split + stable error schema | §5 arch doc |
| metrics | `MetricsCalculator` | `run_evaluation` | avg ms etc. | float/dict | — | offline summary; yaw-aware boxes, FDR name, validated l2 | safe-rate, p95 |

## 2. Function table (runtime participants)

| File | Function/Class | Called by | Calls | Input | Output | Side effects |
|---|---|---|---|---|---|---|
| pipeline.py | `SafetyLayerPipeline.__init__` | tests, eval | planner/checker/corrector ctors | paths, device, on_failure | None | none |
| pipeline.py | `run` | batch, tests, eval | predict, detections->trajs, check, correct | frame + context | result dict (12 keys + timing_ms) | timing only |
| pipeline.py | `run_batch` | tests, eval | loader getters, run | loader, indices, ctx | List[Dict] (stable error schema) | prints errors |
| pipeline.py | `_detections_to_trajectories` | run, calibration | — | boxes/vels/scores, horizon | agent traj list (score>=0.5, cap 20) | none |
| neural_planner.py | `SparseDrivePlanner.__init__` | pipeline, tests | `_load_model` | path, device | None | prints |
| neural_planner.py | `predict` (+`nav_cmd`, `ego_status`) | pipeline, tests | placeholder/inference | 5 arrays + opts | traj/boxes/scores/velocities | cold reset |
| neural_planner.py | `begin_sequence`/`predict_frame`/`end_sequence` | temporal harness | inference (warm) | frames + status | same dict | bank state |
| neural_planner.py | `_load_model` | `__init__` | build_detector, torch.load | config+ckpt | model | sys.path insert |
| neural_planner.py | `_run_inference` | `predict`/`predict_frame` | model() | tensors (+img_metas if warm) | raw dict | none |
| neural_planner.py | `_postprocess` | inference | — | raw list | traj/boxes/scores/velocities | none (fail-loud) |
| neural_planner.py | `_generate_placeholder` | `predict` | — | — | fixed dict | `seed(42)` global |
| data_loader.py | `NuScenesLoader.__init__` | pipeline, tests | devkit (guarded) | dir, version | None | prints |
| data_loader.py | `get_scene_sample` | pipeline, tests | placeholder/real | idx | sample dict | RNG seed |
| data_loader.py | `get_scene_frames` | temporal harness | token walk | idx, n_past | frame list | caches token |
| data_loader.py | `estimate_ego_status` | temporal harness | — | pose pair + dt | (18,) state | none |
| data_loader.py | `get_map_data` | pipeline, tests | placeholder/real extraction | idx | map dict (ego-frame, no posted limits) | caches map |
| data_loader.py | `get_traffic_lights` | pipeline, tests | placeholder | idx | lights dict ({} real) | none |
| checker.py | `TrajectoryChecker.__init__` | pipeline, tests | blob loader | rules_dir, range | None | parses blob |
| checker.py | `check` (+`include_internal`) | pipeline, corrector, tests | cull, builders, ground, solve | traj + ctx | (bool, List) | none (no input mutation) |
| checker.py | `_cull_agents` / `_shown_atom_to_dict` | check | — | trajs / atom | kept list / dict-or-None | none |
| checker.py | `_FactBatcher` | check | ctrl.add (once) | fact strings | — | single parse |
| checker.py | `get_safe_trajectory` | tests | check, correct | traj + ctx | array (identical iff safe) | none |
| corrector.py | `TrajectoryCorrector.__init__` | pipeline, checker, tests | — | checker, iters, on_failure | None | none |
| corrector.py | `correct` | pipeline, checker, tests | check, strategies | traj + ctx | (array, applied) | none |
| evaluation.py | `MetricsCalculator.*` | tests, eval (avg-time) | numpy | lists/arrays | floats | none (validated) |
| run_evaluation.py | `run_evaluation` | CLI | pipeline, loader | num/paths | summary dict | stdout |
| run_calibration.py | `run_calibration` | CLI | planner, checker, corrector | data/flags (+dump/extrapo-late) | summary + optional trajs dump | stdout/JSON |
| run_temporal.py | warm-vs-cold | CLI | planner, loader | scenes | delta table | stdout |
| sweep_thresholds.py | threshold sweep | CLI | checker + trajs dump | dump path | knees table + JSON | restores .lp files |

## 3. "If I change this…" map

- **Planner output format** → `_postprocess` + checker validation break; tests
  `test_sparse_drive/test_pipeline` fail first.
- **Trajectory shape** (≠(6,2)/T>6) → `check` raises `ValueError`; corrector
  grouping assumes per-T dicts.
- **ASP predicates/facts** → rules stop firing silently (clingo is
  open-world); `test_rules` + vacuous-guard catch it.
- **One rule file** → blob rebuild includes it automatically; missing file
  breaks `#include` parse (blob=None → glob fallback; total absence →
  `RuntimeError`).
- **Correction strategy** → convergence behavior changes; `test_corrector`
  + priority test are the tripwires.
- **Config YAML** → nothing (unread); change constructor args / code
  constants instead.
- **Checkpoint** → placeholder↔real flip; only shape contract matters
  downstream.
- **Data loader** → `run_batch` schema; placeholder geometry (int_0) drives
  whether red-fires coverage exists.
- **Clingo invocation** (flags/timeout) → solve latency/correctness; no
  timeout currently set.
- **Metrics** → only `run_evaluation` output + `test_evaluation`; no
  feedback into planning.

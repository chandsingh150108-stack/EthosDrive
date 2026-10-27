# API Reference

Conventions: `docs/conventions.md`. All coordinates meters in code,
integer centimeters (x100) in ASP. Box layout `[x,y,z,w,l,h,yaw]`.

---

## SparseDrivePlanner (`src/neural_planner.py`)

### __init__(model_path: str, device: str = "cuda")
Missing/unloadable weights fall back to the placeholder planner
(`use_placeholder=True`); they do NOT raise.

### predict(images, intrinsics, extrinsics, ego2global, timestamp, nav_cmd=None, ego_status=None) -> Dict
- `images` (6,3,256,704); `intrinsics` (6,3,3); `extrinsics` (6,4,4)
  lidar-to-camera; `ego2global` (4,4); `timestamp` float.
- `nav_cmd`: optional one-hot (3,) — command-head index convention
  is UNVERIFIED (see `experiments/cmd_sweep_results.md`); default
  straight `[1,0,0]`.
- `ego_status`: optional (18,) upstream-layout state
  `[ax,ay,az,wx,wy,wz,vx,vy,vz,steer,0*8]`; ignored by cold
  single-frame inference (temporal queue only).
- Returns `trajectory` (6,2) t=0.5..3.0s ego-frame, `boxes` (N,7),
  `scores` (N,), `velocities` (N,3) planar `[vx,vy,vz]`.
- Cold-resets temporal banks every call (consecutive warm calls
  would crash on missing `img_metas`). Raises `ValueError` on bad
  image shape; fail-loud `KeyError`/`ValueError` on bad model output.

### begin_sequence() / predict_frame(...) / end_sequence()
N-2 warm-queue API: `predict_frame` takes the same args as `predict`
but preserves temporal banks and builds `img_metas` (per-frame
`T_global`) internally. Caller supplies per-frame `ego_status`
(see `data_loader.estimate_ego_status`). Always bracket with
begin/end per scene; never carry state across scenes.

### reset_temporal()
Clears instance banks/queues (cold state).

---

## TrajectoryChecker (`src/checker.py`)

### __init__(rules_dir: str = "rules", max_agent_range_m: float = 150.0)
Pre-concatenates the catalog blob once. Raises `FileNotFoundError`
/ `RuntimeError` for bad/empty rules dir.

### check(trajectory, other_vehicles=None, map_data=None, traffic_lights=None, zones=None, pedestrians=None, weather=None, ego_intent=None, extras=None, include_internal=False, include_constraints=False) -> Tuple[bool, List[Dict]]
- `trajectory` (T,2), T <= 6, meters, ego-frame at `STEP_DT=1.0 s`
  (resample model output first; see N-3a).
- Extended inputs: `zones` (school/construction, explicit
  `active_times` honored), `pedestrians` (`positions` per T),
  `weather` (condition/confidence/severity + caps), `ego_intent`
  (turn/maneuver/lead/platoon/blind-spots/occupancy + durations),
  `extras` (`{'facts': [...]}` raw ASP).
- Returns `(is_safe, violations)`. Shapes: violation/3
  `{rule,vehicle,time}`; violation/4 adds `vehicle2`; hazard/warning/
  insufficient carry `type`. Warnings never count toward unsafe.
- `include_internal=True` appends solver-internal atoms (never
  unsafe): `ttc` (`band`), `threat` (`level`), `comfort` (`score`),
  `future` (`time`), `any`.
- `include_constraints=True` (S13/P1) appends a third tuple element:
  ASP-emitted correction bounds grouped by family
  `{v_max, v_min, x_bounds, y_bounds, a_max, lane_center}` with
  integer-cm entries `(t, ...)`. Constraints never count toward
  unsafe and never appear in `violations`; default `False` keeps the
  legacy 2-tuple.
- Agents beyond range are culled (distance + conflict-aware);
  malformed entries skipped. Never mutates inputs.

### get_safe_trajectory(trajectory, **kwargs) -> np.ndarray
Delegates to `TrajectoryCorrector`. Safe input returns the
IDENTICAL object (pinned); unsafe returns a new array.

---

## TrajectoryCorrector (`src/corrector.py`)

### __init__(checker, max_iterations=100, params=None, params_file=None, on_failure='raise')
- `checker` accepted by type or interface (dual-identity tolerant).
- `params` override `configs/correction_strategies.yaml` (7 keys +
  `group_cap`), which override built-ins.
- `on_failure`: `'raise'` (RuntimeError) or `'brake'` (flagged
  `emergency_hold_uncorrectable` hold for upstream takeover).

### correct(trajectory, **kwargs) -> Tuple[np.ndarray, List[Dict]]
Fixpoint: per-timestep batching with dirty-set guards, per-group
caps, stagnation break (3 identical sets). Advisory findings
(`late_signal`/warnings) pass through as `advisory_no_change`.
Returns `(corrected, applied)` with
`{rule, timestep, correction_type, original_value, corrected_value}`.

---

## CorrectorQP (`src/corrector.py`) — S14/S15, opt-in

### __init__(checker, optimizer=None, optimizer_params=None, max_rounds=3, on_failure='raise', heuristic_fallback=True, bounds='full')
- Constrained alternative to the heuristic fixpoint: calls
  `check(include_constraints=True)`, projects the trajectory with
  `ConstraintOptimizer`, then re-validates symbolically each round.
- `bounds`: `'full'` (all emitted bounds), `'caps'`
  (`v_max`/`v_min`/`a_max`), or `'boxes'` (`x_bounds`/`y_bounds`/
  `lane_center`) for the P-6 family ablation.
- `heuristic_fallback=True` delegates unresolved scenes to
  `TrajectoryCorrector` (same `on_failure` contract). **Default
  production corrector remains the heuristic**; QP measured at parity
  on mini (fix rate 0.375 both, `experiments/qp_ablation.json`) and is
  not yet GPU-calibrated (P-21).

### correct(trajectory, **kwargs) -> Tuple[np.ndarray, List[Dict]]
Same return schema as `TrajectoryCorrector.correct`. `include_*`
kwargs are managed internally and never forwarded to the fallback.

---

## ConstraintOptimizer (`src/constraint_optimizer.py`) — S14

### __init__(params=None)
Knobs: `lr` (0.2), `steps` (50), `tol_cm` (1.0), `lane_pull` (0.05).

### solve(trajectory, constraints, params=None) -> Tuple[np.ndarray, Dict]
Projected-gradient projection of a (T,2) trajectory onto the grouped
ASP bounds, keeping T=0 pinned and staying close to the original.
Returns `(corrected_m, info)` with `status`, `iterations`,
`solve_time_ms`, `pinned_t0`. Pure NumPy (no scipy/GPU); validates
shape/finiteness and raises `ValueError` on bad input. The projection
is per-timestep — horizon-joint solving is P-23.

---

## correction_strategies (`src/correction_strategies.py`)

Single dispatch table: `strategy_for(rule) -> (group, method)`.
Groups: speed/collision/lane/stop/follow/comfort/min_speed (+
`ADVISORY_RULES={'late_signal'}`). `all_known_rules()` feeds the
P-9 registry test. `checker.py` must not duplicate this table.

---

## NuScenesLoader (`src/data_loader.py`)

### __init__(data_dir: str, version: str = "v1.0-mini")
`use_placeholder=True` when the path is missing. Real path uses the
devkit + map expansion (`maps/expansion/*.json`, see
`docs/environment.md`).

### get_scene_sample(scene_idx) -> Dict
`images` (6,3,256,704) uint8, `intrinsics` (6,3,3), `extrinsics`
(6,4,4) cam_T_lidar, `ego2global` (4,4), `timestamp` float.
Middle sample; token cached per scene.

### get_scene_frames(scene_idx, n_past=4) -> list
Consecutive oldest->newest sample dicts ending at the canonical
sample (walks `prev`; repeats at scene start). Powers N-2 warm runs.

### get_trajectory(scene_idx) -> np.ndarray
GT ego poses every 2nd sample (1 s grid), ego-local frame.

### get_map_data(scene_idx) -> Dict
Real extraction around the ego pose (40 m): discretized
lanes+connectors (segments/widths/directions/bike types),
`is_intersection` polygons, tiled crossings/walkways — all in
ego frame. `road_type`/`speed_limit` OMITTED when unposted
(unknown); signal phases are `{}` (honest no-knowledge).
Placeholder: two urban/13.4 lanes + reachable red + scene-9
obstacle (T-9 hostile design).

### get_traffic_lights(scene_idx) -> Dict
`{id: {color, time/times, x, y, radius, arrows}}`; `{}` on real data.

### estimate_ego_status(prev_ego2global, ego2global, dt) -> np.ndarray
(18,) upstream-layout state with pose-derived planar velocity at
indices 6,7; zeros elsewhere (mini has no CAN bus).

---

## SafetyLayerPipeline (`src/pipeline.py`)

### __init__(planner_path, rules_dir="rules", device="cuda", on_failure="raise", corrector_mode="heuristic", optimizer_params=None)
- `corrector_mode`: `'heuristic'` (default `TrajectoryCorrector`
  fixpoint) or `'qp'` (`CorrectorQP` constrained solve + re-validation,
  heuristic fallback on failure).
- `optimizer_params`: forwarded to `CorrectorQP` (`lr`, `steps`,
  `tol_cm`, `lane_pull`); ignored in heuristic mode.

### run(images, intrinsics, extrinsics, ego2global, timestamp, map_data=None, traffic_lights=None, zones=None, pedestrians=None, weather=None, ego_intent=None, extras=None) -> Dict
Predict (6,2 native @0.5 s) -> resample to the checker grid (T=0
anchor + true +1/+2/+3 s, same N-3a grid as calibration) ->
detections to CV agent trajectories (score>=0.5, cap 20, dt=STEP_DT)
-> check -> correct. Returns the native `original_trajectory`
(6,2), the checked `safe_trajectory` (4,2), `horizon_s` (3),
violations, `is_safe`, `inference_time_ms` (+ alias
`pipeline_time_ms`), per-stage `timing_ms`
`{planner,checker,corrector}`, `corrector_mode`, boxes/scores.

### run_batch(data_loader, scene_indices, zones=None, pedestrians=None, weather=None, ego_intent=None, extras=None) -> List[Dict]
Per-scene (or broadcast) context; stable error dicts (same keys,
Nones) instead of raising; `corrector` RuntimeError propagates
from `run()` (pinned).

---

## MetricsCalculator (`src/evaluation.py`)

- `collision_rate(trajs, gt_boxes, pred_boxes)` — yaw-aware
  footprints over static boxes; mismatched batch lengths raise.
- `l2_error(pred, gt)` — validated (T,2) finite inputs; empty -> 0.0.
- `violation_detection_rate` / `false_discovery_rate` (alias kept:
  `false_positive_rate`) — rule+time match, vehicle/vehicle2/type
  compared when both sides carry them.
- `average_inference_time(ms_list)`.

# SparseDrive Integration — Boundary Only

> Deep internals (detector/map/motion decoders, CUDA kernels) out of scope.
> This doc covers exactly what `asp-safety` asks for and what it consumes.

## 1. How the wrapper reaches SparseDrive

`SparseDrivePlanner._load_model` (`src/neural_planner.py`): inserts
`<repo>/SparseDrive` on `sys.path` → `build_detector` from the mmdet
plugin registry (mmdet, NOT mmdet3d — mmcv version conflict) → config
`projects/configs/sparsedrive_small_stage2.py` via importlib →
`build_detector(cfg.model)` → `torch.load(path, weights_only=True)`
with `strict=False` (`{state_dict}`-or-raw) → `.to(device).eval()`
with CPU fp16 wrappers disabled. Config: `model.type="SparseDrive"`,
head `SparseDriveHead`, `task_config={det,map,motion_plan}`,
`ego_fut_ts=6/mode=6`, `fut_ts=12`, `queue_length=4`,
`input_shape=(704,256)`.
Missing file or load exception → placeholder (`use_placeholder=True`);
`ImportError` → `RuntimeError` with setup hints. Synthetic
checkpoint-exact anchors under `SparseDrive/data/kmeans/*.py`
(overwritten by `load_state_dict`); no-compile fallbacks for the
CUDA-only `deformable_aggregation` op (pure-torch, verified 8.3e-07)
and `flash_attn` (torch SDPA path).

## 2. What goes in

`predict(images, intrinsics, extrinsics, ego2global, timestamp,
nav_cmd=None, ego_status=None)`: `images (6,3,256,704)` order
FRONT/FL/FR/BACK/BL/BR (shape-validated; uint8 or float32).
`_run_inference` builds (batched): `img (1,6,3,256,704)`
nuScenes-normalized (mean `[123.675,116.28,103.53]`, std
`[58.395,57.12,57.375]`); `projection_mat (1,6,4,4)` as K@[R|t]
embedded in 4x4 (RQ-2 fix — the old `(4,4)@(3,3)` form was a shape
crash); `image_wh=[[704,256]]×6`; `ego_status` zeros unless supplied
(cold inference ignores it); `gt_ego_fut_cmd=[1,0,0]` unless
`nav_cmd` given (head-selection semantics UNVERIFIED — see
`experiments/cmd_sweep_results.md`); `timestamp`. Model called as
`model(**inputs)` under `no_grad`.

Temporal state: the banks need per-frame `img_metas` (`T_global`)
that single-frame callers don't supply, so a warm bank crashes the
second call (`KeyError: 'img_metas'`). `predict()` therefore
cold-resets every call. `predict_frame()` (between
`begin_sequence()`/`end_sequence()`) supplies metas + per-frame
`ego_status` instead — the N-2 warm path. Never mix state across
scenes. `ego2global` IS the lidar-frame `T_global`.

## 3. What comes out (and what is actually used)

`_postprocess` reads `raw_output[0]['img_bbox']`, fail-loud
(`KeyError`/`ValueError`, never placeholder fallback):

| Key | Becomes | Fallback |
|---|---|---|
| `final_planning` | `trajectory (6,2)` ego meters, t=0.5..3.0 s | raise |
| `boxes_3d` (N,10 `[x,y,z,w,l,h,yaw,vx,vy,vz]`) | `boxes (N,7)` + `velocities (N,3)` | `zeros((0,7))` / `zeros((0,3))` |
| `scores_3d` | `scores (N,)` (length must match boxes) | `zeros((0,))` |

Map vectors, motion forecasts, mode scores are ignored. Pinned by
the T-5 golden test (incl. sliced velocities).

## 4. Placeholder (offline stand-in)

`_generate_placeholder`: `seed(42)`; 5 m/s × 0.5 s cruise with
slight lateral drift; boxes `[20,0,.5,2,4.5,1.5,0]`,
`[40,5,.5,2,4.5,1.5,0]` (static, zero velocities); scores
`[.95,.87]`. Deterministic (pinned).

## 5. Canonical launch (upstream, for reference)

`SparseDrive/tools/test.py`: `Config.fromfile` → plugin import →
`build_detector` → `load_checkpoint` → dataloader + `single_gpu_test` →
`dataset.evaluate`. The wrapper shortcuts all of it (direct build +
`torch.load`, no Config/dataloader/`load_checkpoint`) — **VERIFIED**.

## 6. What each side knows

- SparseDrive knows **nothing** about ASP (no import, no predicate).
- ASP layer knows **nothing** about SparseDrive (only the `(6,2)`/`(N,7)`
  arrays plus calibration passthrough).
- Interface = `predict()` return dict. Change `final_planning` shape/keys
  and `_postprocess` breaks; change trajectory `(6,2)` and checker
  validation rejects it.
- Horizon fact: the head emits 6×0.5 s = 3 s (`plan_reg_branch`
  output dim 12 is baked in; longer horizons need retraining —
  N-3(c) verdict). The checker contract is 6×1.0 s; both the
  calibration harness and `pipeline.run()` resample with a T=0
  anchor and stamp `horizon_s` (unified — previously `run()` checked
  native and read speeds halved).

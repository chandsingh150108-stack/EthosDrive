# Warm-vs-cold temporal results (N-2, 2026-09-26)

CUDA, 10 mini scenes, n_past=4 history frames + canonical eval frame.
Cold = single predict() on eval frame. Warm = begin_sequence +
predict_frame x5 (img_metas + pose-derived ego_status) + end_sequence.

| scene | cold L2 | warm L2 |
|-------|---------|---------|
| 0 |  8.50 |  7.78 |
| 1 |  7.89 |  6.28 |
| 2 |  4.53 |  2.39 |
| 3 | 12.91 | 11.41 |
| 4 |  4.19 |  2.31 |
| 5 | 21.47 | 21.31 |
| 6 | 12.04 |  8.94 |
| 7 | 26.27 | 24.65 |
| 8 | 14.76 | 13.98 |
| 9 |  2.89 |  2.77 |

Means: cold 11.54 m / warm 10.18 m (delta +1.36 m, warm wins 10/10).

Notes:
- Consecutive-frame inference without img_metas crashes with
  KeyError 'img_metas' (temporal banks); predict() cold-resets per
  call, predict_frame() supplies metas. Never mix warm state across
  scenes (end_sequence/begin_sequence per scene).
- ego_status layout follows the upstream converter
  (tools/data_converter/nuscenes_converter.py::get_ego_status):
  [ax,ay,az, wx,wy,wz, vx,vy,vz, steer, 0*8]; planar velocity from
  consecutive ego poses (mini has no CAN bus; upstream falls back to
  zeros there).
- Nav-command conditioning stays weak under warm inference too
  (see cmd_sweep_results.md); intent work remains deferred.
- Cost: warm = 5x inference (~12 s/scene GPU). Wiring warm predictions
  into run_calibration proper is a P-5 decision, not done here.

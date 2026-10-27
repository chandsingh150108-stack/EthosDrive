# Replay Evaluation Design (P-11)

Open-loop flagging is not safety evidence. This doc designs the two
harnesses that get us there. **Section A (annotation replay) is
implemented** in `experiments/replay_eval.py` with
`experiments/replay_baseline.json`; **Section B (CARLA closed-loop) is
implemented** (P-25, 2026-10-22 bridge + scenarios + frozen 15-run
pilot matrix; deviations ledger `docs/carla_harness.md`).

## A. nuScenes annotation replay (primary, ~1 week)

Replace single-frame detections with TRACKED agents from
`sample_annotation` records, and score the layer over full scenes:

1. Per scene, walk samples via `sample['next']` from the first token.
2. At each sample, read `sample['anns']` -> annotation records
   (`translation`, `rotation`, `size`, `instance_token`,
   `category_name`). Keep `vehicle.*` + `human.*` categories.
3. Group by `instance_token` across samples -> per-agent position
   series in global frame; transform to the eval sample's ego frame
   (same `_pose_to_mat` math as the loader).
4. Eval frame = middle sample (matches calibration): ego plan =
   planner prediction (cold or N-2 warm), agents = tracked series
   resampled to STEP_DT, map = N-1 extraction, GT = log ego.
5. Metrics per scene: findings on pred vs findings on GT (FP proxy),
   fix rate, timing. Aggregate exactly like `run_calibration.py`
   (reuse its summary schema + `horizon_s` stamps).

Why this beats the current setup: CV-projected single-frame
detections hallucinate motion (parked cars drift, FPs persist one
frame); tracked annotations give true agent histories, so R3/R4/R13
verdicts become evidence instead of noise. Parked-vs-live falls out
of the series variance (no box classes needed).

Cost estimate: annotation walk is devkit-only (no GPU); per-scene
overhead is one extra sample sweep (~10 samples x 6 cameras is NOT
needed -- annotations come from records, no images).

## B. CARLA closed-loop (implemented 2026-10-22, P-25)

Feasibility verdict (probed 2026-09-26, zero changes made): runnable.
No admin rights (prebuilt package needs none), RTX 4050 Laptop 6 GB
(exactly the stated minimum; 8 GB recommended), Ryzen 5 6600H,
16 GB RAM, 76 GB free (packaged server ~= 20 GB; source build at
165 GB+ ruled out), VCRedist x64 present, ports 2000/2001 free,
conda Python 3.10 (cp310 wheels exist).

Installed state: 0.9.15 stable at `C:\CARLA` (outside repo),
client in env, default map Town10HD_Opt smoke-verified
(`SMOKE OK`, planner-resolution frame). Expected rates: server
render ~= 10-20 FPS (6-camera rig); closed-loop wall rate
~= 0.3-0.4 steps/s (planner-bound), so the bridge replans
every 40th tick under synchronous fixed-timestep (deterministic
sim-time; replan wall is sim-time-frozen).

Implemented harness:
1. `experiments/carla_bridge.py`: 6-camera rig -> planner inputs
   (or the lane-plan analyzer behind `--plan-engine lane`), ASP
   checker verdict -> heuristic/qp/shield-off dispatch, plan
   execution by a pure-pursuit speed controller, run JSON per run.
2. `experiments/carla_scenarios.py`: S1 free drive, S2 red-light
   run (cluster pinned red), S3 cut-in, S4 jaywalker, S5 oncoming
   left turn; deterministic spawn, cleanup resets all map lights.
3. `experiments/carla_metrics.py`: frozen aggregate
   (`experiments/carla_metrics.json`, 15 runs).

Measured pilot numbers (seed 1, Town10HD_Opt, plan-engine lane):

| Run | km | status | collisions | shield verdict |
|---|---|---|---|---|
| S1 off / heuristic / qp | 0.60 x3 | distance_reached | 0 | no hazard on clean track; forcings all pass-through |
| S2 off | 0.60 | distance_reached | 0 | runs the red junction (81 red_light findings) |
| S2 heuristic / qp | ~0 | wall_cap (holds) | 0 | holds at the red box edge, give-up 0.0 |
| S3 off | 0.03 | stalled+collided (rear-ends cut-in) | 1 | NA |
| S3 heuristic / qp | 0.03 | wall_cap (follows at 4 m/s) | 0 | survives; emergency-hold brake fallback |
| S4 off | 0.60 | distance_reached | 0* | kinematic walker produces no collision event |
| S4 heuristic / qp | 0.47/0.52 | wall_cap | 0 | waits out the jaywalker |
| S5 off | 0.03 | stalled+collided (turn conflict) | 1 | NA |
| S5 heuristic / qp | ~0.02 | wall_cap | 0 | holds; emergency-hold fallback |

Aggregate (frozen `experiments/carla_metrics.json`): collision rate
13.3% (2/15, both off arms), frame drops 0, give-up 0 across all
frozen arms, intervention rate ~0.52 (dominated by hazard replans on
the hazard-heavy closed ends). Caveats: VRUs/NPCs are kinematic (no
physical impulses); evidence limited to seed 1 for the pilot; Lane
plan engine (not the OOD neural plan) drives EGO relative to the
CARLA imagery — neural plans remain logged in the run JSONs.

## C. Measured numbers for the report (warm official baseline, 10 mini scenes)

Source of truth: `experiments/calibration_baseline.json` (warm,
`horizon_s=3`, tracked replay variants in `replay_baseline.json`).

| Metric | Value | Source |
|---|---|---|
| Planner L2 (cold / warm) | 11.54 m / 10.18 m | replay / calibration |
| GT safe rate | 0.30 (residuals ledgered) | calibration_baseline.json |
| Pred safe rate | 0.20 | same |
| Pred fixed rate | 0.50 (agent-free 0.90 on the earlier cold run) | same |
| Planner / checker / corrector ms | ~13026 / ~60 / ~2562 | same (loaded laptop) |
| 20-agent check (synthetic) | 37 ms min | test_performance |
| Catalog coverage | 73/73 findings fire | test_rule_registry |
| Nav-command conditioning | weak (no mappable semantics) | cmd_sweep_results.md |
| Horizon | 3 s verified shield, blind beyond | N-3(a); extrapolation dropped |
| Corrector A/B (mini, unsafe=8) | heuristic 0.375 = QP 0.375 | qp_ablation.json |
| Bound-family ablation | full 0.375 > boxes 0.125 > caps 0.0 | qp_ablation.json |
| Warm GPU A/B (calibration) | fixed 0.50 = 0.50 | calibration_qp.json |
| Warm tracked-replay A/B | fixed 0.50 = 0.50, same scene sets (5 fixed / 5 give-ups) | replay_warm_heuristic.json / replay_warm_qp.json |

`paper/report.tex` now reflects the 63-rule catalog and these measured
numbers; the numbers are mini-split evidence, not a safety claim.

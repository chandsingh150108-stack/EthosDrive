"""Open-loop calibration harness (P-5 option b, v1 baseline).

For each nuScenes mini scene this script measures, on REAL data:
  - L2(pred, GT): pred is 6x0.5 s waypoints; GT is interpolated to the
    same 0.5 s grid for comparison.
  - Rule findings on GT (6x1.0 s waypoints, T=0 is now): a proxy for the
    FALSE-POSITIVE rate, since log trajectories are assumed safe-ish.
  - Rule findings on pred RESAMPLED to the checker grid: T=0 is the
    current ego pose ([0, 0] in ego frame) plus pred indices 1,3,5 as
    T=1,2,3 (N-3a anchor). The checker contract is STEP_DT=1.0 s (see
    checker.STEP_DT); feeding raw 0.5 s waypoints would halve every
    computed speed (P-1). Without the T=0 anchor, pred[1] (true +1.0 s)
    would be asserted as T=0 ("now"), shifting every speed/braking/TTC
    computation by a second.
  - Horizon: the resampled pred path spans true 0..3 s (4 points), not
    0..5 s -- a product limitation surfaced here, not hidden: blind
    beyond 3 s. Every summary carries horizon_s=3 so GT-vs-pred
    comparisons stay honest.
  - Timing split: planner (GPU) vs checker vs corrector.

Output: per-rule finding table + JSON summary for threshold tuning.
GT findings point at over-sensitive thresholds; pred findings point at
planner behavior. Neither is ground truth -- both are inputs to the
calibration decision, not conclusions.
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_planner import SparseDrivePlanner
from checker import TrajectoryChecker, STEP_DT
from corrector import TrajectoryCorrector, CorrectorQP
from data_loader import NuScenesLoader
from evaluation import MetricsCalculator
from pipeline import SafetyLayerPipeline


def interp_gt(gt: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Linear-interpolate GT (1 s grid from t=0) to query times."""
    gt_t = np.arange(len(gt), dtype=float)
    x = np.interp(times, gt_t, gt[:, 0].astype(float))
    y = np.interp(times, gt_t, gt[:, 1].astype(float))
    return np.column_stack([x, y]).astype(np.float32)


def run_calibration(data_root, device="cuda", scenes=None,
                    rules_dir=None, model_path=None, extrapolate=False,
                    dump_trajs=None, warm=True,
                    corrector_mode="heuristic"):
    base_dir = Path(__file__).parent.parent
    if rules_dir is None:
        rules_dir = str(base_dir / "rules")
    if model_path is None:
        model_path = str(
            base_dir / "SparseDrive" / "ckpt" / "sparsedrive_stage2.pth")

    loader = NuScenesLoader(data_root, version="v1.0-mini")
    if loader.use_placeholder:
        raise RuntimeError(f"no nuScenes data at {data_root}")
    if scenes is None:
        scenes = list(range(loader.num_scenes))

    planner = SparseDrivePlanner(model_path, device=device)
    checker = TrajectoryChecker(rules_dir)
    if corrector_mode == "qp":
        corrector = CorrectorQP(checker)
    elif corrector_mode == "heuristic":
        corrector = TrajectoryCorrector(checker)
    else:
        raise ValueError("corrector_mode must be 'heuristic' or 'qp'")

    assert abs(STEP_DT - 1.0) < 1e-9, "thresholds assume STEP_DT=1.0"

    gt_rule_counts = Counter()
    pred_rule_counts = Counter()
    pred_rule_counts_meas = Counter()  # N-3(b): violations at T<=3
    pred_rule_counts_extrap = Counter()  # N-3(b): violations at T>=4
    l2_errors = []
    t_plan, t_check, t_correct = [], [], []
    n_gt_safe = n_pred_safe = n_pred_fixed = 0

    print(f"Calibrating on {len(scenes)} scenes (device={device}, "
          f"{'warm' if warm else 'cold'} predictions, "
          f"corrector={corrector_mode})...")
    for idx in scenes:
        planner.reset_temporal()  # scenes are independent; no carryover
        sample = loader.get_scene_sample(idx)
        gt = loader.get_trajectory(idx)
        map_data = loader.get_map_data(idx)

        t0 = time.perf_counter()
        if warm:
            # N-2 warm path: walk 4 history frames + eval frame with
            # pose-derived ego status; evaluate the last output.
            from data_loader import estimate_ego_status
            frames = loader.get_scene_frames(idx, n_past=4)
            planner.begin_sequence()
            out = None
            prev_g, prev_t = None, None
            for fr in frames:
                status = (np.zeros(18, dtype=np.float32)
                          if prev_g is None else estimate_ego_status(
                              prev_g, fr['ego2global'],
                              fr['timestamp'] - prev_t))
                out = planner.predict_frame(
                    fr['images'].astype(np.float32), fr['intrinsics'],
                    fr['extrinsics'], fr['ego2global'], fr['timestamp'],
                    ego_status=status)
                prev_g, prev_t = fr['ego2global'], fr['timestamp']
            planner.end_sequence()
        else:
            out = planner.predict(
                sample['images'].astype(np.float32),
                sample['intrinsics'], sample['extrinsics'],
                sample['ego2global'], sample['timestamp'])
        t_plan.append((time.perf_counter() - t0) * 1000.0)
        pred = out['trajectory']

        # L2 on the model's native 0.5 s grid (GT interpolated).
        pred_times = (np.arange(len(pred)) + 1) * 0.5
        l2_errors.append(MetricsCalculator.l2_error(
            pred, interp_gt(gt, pred_times)))

        # GT check at native 1 s steps. Agent context included: the log
        # ego should not collide with detected traffic either (R4 FP
        # check). Detections are single-frame; project them over the GT
        # horizon the same way.
        gt_agents = SafetyLayerPipeline._detections_to_trajectories(
            out.get('boxes'), out.get('velocities'), out.get('scores'),
            len(gt))
        t0 = time.perf_counter()
        ok_gt, v_gt = checker.check(gt, map_data=map_data,
                                    other_vehicles=gt_agents or None)
        t_check.append((time.perf_counter() - t0) * 1000.0)
        n_gt_safe += ok_gt
        for v in v_gt:
            if v.get('type', 'violation') != 'warning':
                gt_rule_counts[v['rule']] += 1

        # Pred check resampled to the checker grid (N-3a): T=0 is the
        # current ego pose, then pred indices 1,3,5 (true +1/+2/+3 s)
        # as T=1,2,3. Native 6x0.5 s schema is asserted, not assumed.
        assert len(pred) == 6 and pred.shape[1] == 2, \
            f"native pred schema drifted: {pred.shape}"
        pred_1s = np.vstack([
            np.zeros((1, 2), dtype=np.float32),  # T=0: ego pose, now
            pred[1::2][:3],                      # T=1,2,3: true +1/+2/+3 s
        ])
        horizon_s = 3
        if extrapolate:
            # N-3(b) LABELED experiment: constant-velocity extension to
            # T=4,5 from the last two measured seconds. Invented physics
            # for the blind half -- every summary stamps horizon_s=6 and
            # the report splits violations by measured (T<=3) vs
            # extrapolated (T>=4) timesteps. Keep only if signal > noise.
            vel = pred_1s[3] - pred_1s[2]
            extra = np.array([pred_1s[3] + vel, pred_1s[3] + 2 * vel],
                             dtype=np.float32)
            pred_1s = np.vstack([pred_1s, extra])
            horizon_s = 6
        t0 = time.perf_counter()
        # Tracked agents from planner detections (P-18): without these,
        # R3/R4/R13 never see real traffic. Same projection as
        # pipeline.run (score-filtered, capped, checker-culled).
        agent_trajs = SafetyLayerPipeline._detections_to_trajectories(
            out.get('boxes'), out.get('velocities'), out.get('scores'),
            len(pred_1s))
        ok_p, v_p = checker.check(pred_1s, map_data=map_data,
                                  other_vehicles=agent_trajs or None)
        t_check.append((time.perf_counter() - t0) * 1000.0)
        n_pred_safe += ok_p
        for v in v_p:
            if v.get('type', 'violation') != 'warning':
                pred_rule_counts[v['rule']] += 1
                if extrapolate and v.get('time', 0) >= 4:
                    pred_rule_counts_extrap[v['rule']] += 1
                else:
                    pred_rule_counts_meas[v['rule']] += 1

        t0 = time.perf_counter()
        try:
            fixed, _ = corrector.correct(pred_1s, map_data=map_data,
                                         other_vehicles=agent_trajs or None)
            t_correct.append((time.perf_counter() - t0) * 1000.0)
            ok_f, _ = checker.check(fixed, map_data=map_data,
                                    other_vehicles=agent_trajs or None)
        except RuntimeError as e:
            # Uncorrectable within budget (common on extrapolated
            # horizons): record the failure, don't crash the sweep.
            t_correct.append((time.perf_counter() - t0) * 1000.0)
            ok_f = False
            print(f"    corrector gave up: {e}")
        n_pred_fixed += ok_f
        if dump_trajs is not None:
            dump_trajs[str(idx)] = {
                'gt': gt.tolist(),
                'pred_1s': pred_1s.tolist(),
                'map_data': map_data,
                'boxes': np.asarray(
                    out.get('boxes', np.zeros((0, 7)))).tolist(),
                'velocities': np.asarray(
                    out.get('velocities', np.zeros((0, 3)))).tolist(),
                'scores': np.asarray(
                    out.get('scores', np.zeros((0,)))).tolist(),
            }
        print(f"  scene {idx}: L2={l2_errors[-1]:.2f}m "
              f"gt_safe={ok_gt} pred_safe={ok_p} fixed={ok_f}")

    summary = {
        'scenes': len(scenes),
        'extrapolated': bool(extrapolate),
        'warm_predictions': bool(warm),
        'corrector_mode': corrector_mode,
        'horizon_s': 6 if extrapolate else 3,
        'horizon_steps_pred': 6 if extrapolate else 4,
        'l2_mean': float(np.mean(l2_errors)),
        'l2_values': [float(x) for x in l2_errors],
        'gt_safe_rate': n_gt_safe / len(scenes),
        'pred_safe_rate': n_pred_safe / len(scenes),
        'pred_fixed_rate': n_pred_fixed / len(scenes),
        'gt_rule_counts': dict(gt_rule_counts),
        'pred_rule_counts': dict(pred_rule_counts),
        'pred_rule_counts_measured_Tle3': dict(pred_rule_counts_meas),
        'pred_rule_counts_extrapolated_Tge4': dict(pred_rule_counts_extrap),
        'ms_planner_mean': float(np.mean(t_plan)),
        'ms_checker_mean': float(np.mean(t_check)),
        'ms_corrector_mean': float(np.mean(t_correct)),
    }
    print("\n=== Calibration baseline ===")
    print(f"  horizon: {summary['horizon_s']} s pred "
          f"({summary['horizon_steps_pred']} steps, T=0 anchored)")
    print(f"  L2 mean: {summary['l2_mean']:.2f} m")
    print(f"  GT safe rate (want ~1.0): {summary['gt_safe_rate']:.2f}")
    print(f"  pred safe rate: {summary['pred_safe_rate']:.2f}")
    print(f"  pred fixed rate: {summary['pred_fixed_rate']:.2f}")
    print("  GT rule hits (false-positive suspects):")
    for rule, cnt in gt_rule_counts.most_common():
        print(f"    {rule}: {cnt}")
    print("  pred rule hits:")
    for rule, cnt in pred_rule_counts.most_common():
        print(f"    {rule}: {cnt}")
    if extrapolate:
        print("  pred hits at measured T<=3:")
        for rule, cnt in pred_rule_counts_meas.most_common():
            print(f"    {rule}: {cnt}")
        print("  pred hits at EXTRAPOLATED T>=4 (invented physics):")
        for rule, cnt in pred_rule_counts_extrap.most_common():
            print(f"    {rule}: {cnt}")
    print(f"  timing ms planner/checker/corrector: "
          f"{summary['ms_planner_mean']:.1f} / "
          f"{summary['ms_checker_mean']:.1f} / "
          f"{summary['ms_corrector_mean']:.1f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Open-loop calibration baseline")
    ap.add_argument("--data-root", default="C:/data/sets/nuscenes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene indices, default all")
    ap.add_argument("--out", default=None, help="JSON summary path")
    ap.add_argument("--extrapolate", action="store_true",
                    help="N-3(b): extend pred to T=4,5 by constant velocity "
                         "(labeled experiment; see summary 'extrapolated')")
    ap.add_argument("--cold", action="store_true",
                    help="single-frame cold predictions (default is the "
                         "N-2 warm path)")
    ap.add_argument("--corrector", default="heuristic",
                    choices=["heuristic", "qp"],
                    help="correction engine: heuristic fixpoint (default) "
                         "or CorrectorQP constrained solve + re-validation")
    ap.add_argument("--dump-trajs", default=None,
                    help="P-5: save per-scene {gt, pred_1s, map_data} JSON "
                         "for GPU-free threshold sweeps")
    args = ap.parse_args()
    scenes = ([int(x) for x in args.scenes.split(",")]
              if args.scenes else None)
    dump = {} if args.dump_trajs else None
    summary = run_calibration(args.data_root, args.device, scenes,
                               extrapolate=args.extrapolate,
                               dump_trajs=dump,
                               warm=not args.cold,
                               corrector_mode=args.corrector)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")
    if args.dump_trajs and dump is not None:
        Path(args.dump_trajs).write_text(json.dumps(dump))
        print(f"wrote {args.dump_trajs} "
              f"({len(dump)} scenes for threshold sweeps)")

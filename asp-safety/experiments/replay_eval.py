"""Annotation-replay evaluation (forward plan item 1).

Same schema as run_calibration.py, but agent context comes from
TRACKED annotation histories (loader.get_tracked_agents) instead of
CV-projected single-frame detections. Parked agents (displacement
< 0.5 m) are reported separately and excluded from the moving-only
pass, which answers the parked-vs-live question without box classes.

Predictions: warm multi-frame walk by default (`--cold` reproduces the
old single-frame behavior). Corrector engine selectable (`--corrector
heuristic|qp`) for the P-22 warm tracked-replay ablation.

Usage: python experiments/replay_eval.py [--scenes 0,1] [--out JSON]
"""

import sys
import time
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_planner import SparseDrivePlanner
from checker import TrajectoryChecker, STEP_DT
from corrector import TrajectoryCorrector, CorrectorQP
from data_loader import NuScenesLoader
from evaluation import MetricsCalculator
from run_calibration import interp_gt


def _count(violations, counts):
    for v in violations:
        if v.get('type', 'violation') != 'warning':
            counts[v['rule']] += 1


def run_replay(data_root, device="cuda", scenes=None,
               rules_dir=None, model_path=None,
               warm=True, corrector_mode="heuristic"):
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

    gt_counts = Counter()
    gt_counts_moving = Counter()
    pred_counts = Counter()
    pred_counts_moving = Counter()
    l2_errors = []
    t_plan, t_check, t_correct = [], [], []
    n_gt_safe = n_pred_safe = n_pred_fixed = 0
    n_parked = n_moving = 0

    print(f"Replay eval on {len(scenes)} scenes (device={device}, "
          f"{'warm' if warm else 'cold'} predictions, "
          f"corrector={corrector_mode})...")
    for idx in scenes:
        planner.reset_temporal()
        sample = loader.get_scene_sample(idx)
        gt = loader.get_trajectory(idx)
        map_data = loader.get_map_data(idx)
        agents = loader.get_tracked_agents(idx)
        moving = [a['trajectory'] for a in agents.values()
                  if not a['parked']]
        parked = [a['trajectory'] for a in agents.values()
                  if a['parked']]
        n_moving += len(moving)
        n_parked += len(parked)
        all_agents = moving + parked

        t0 = time.perf_counter()
        if warm:
            # N-2 warm path (same as run_calibration): walk 4 history
            # frames + eval frame with pose-derived ego status.
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

        pred_times = (np.arange(len(pred)) + 1) * 0.5
        l2_errors.append(MetricsCalculator.l2_error(
            pred, interp_gt(gt, pred_times)))

        t0 = time.perf_counter()
        ok_gt, v_gt = checker.check(
            gt, map_data=map_data, other_vehicles=all_agents or None)
        t_check.append((time.perf_counter() - t0) * 1000.0)
        n_gt_safe += ok_gt
        _count(v_gt, gt_counts)
        _, v_gtm = checker.check(
            gt, map_data=map_data,
            other_vehicles=moving or None)
        _count(v_gtm, gt_counts_moving)

        pred_1s = np.vstack([
            np.zeros((1, 2), dtype=np.float32),
            pred[1::2][:3],
        ])
        t0 = time.perf_counter()
        ok_p, v_p = checker.check(
            pred_1s, map_data=map_data,
            other_vehicles=all_agents or None)
        t_check.append((time.perf_counter() - t0) * 1000.0)
        n_pred_safe += ok_p
        _count(v_p, pred_counts)
        _, v_pm = checker.check(
            pred_1s, map_data=map_data,
            other_vehicles=moving or None)
        _count(v_pm, pred_counts_moving)

        t0 = time.perf_counter()
        try:
            fixed, _ = corrector.correct(
                pred_1s, map_data=map_data,
                other_vehicles=all_agents or None)
            t_correct.append((time.perf_counter() - t0) * 1000.0)
            ok_f, _ = checker.check(
                fixed, map_data=map_data,
                other_vehicles=all_agents or None)
        except RuntimeError as e:
            t_correct.append((time.perf_counter() - t0) * 1000.0)
            ok_f = False
            print(f"    corrector gave up: {e}")
        n_pred_fixed += ok_f
        print(f"  scene {idx}: L2={l2_errors[-1]:.2f}m "
              f"agents={len(all_agents)} (moving {len(moving)}) "
              f"gt_safe={ok_gt} pred_safe={ok_p} fixed={ok_f}")

    summary = {
        'scenes': len(scenes),
        'horizon_s': 3,
        'horizon_steps_pred': 4,
        'agents': 'tracked-annotations',
        'warm_predictions': bool(warm),
        'corrector_mode': corrector_mode,
        'n_tracked_total': n_moving + n_parked,
        'n_moving': n_moving,
        'n_parked': n_parked,
        'l2_mean': float(np.mean(l2_errors)),
        'l2_values': [float(x) for x in l2_errors],
        'gt_safe_rate': n_gt_safe / len(scenes),
        'pred_safe_rate': n_pred_safe / len(scenes),
        'pred_fixed_rate': n_pred_fixed / len(scenes),
        'gt_rule_counts': dict(gt_counts),
        'gt_rule_counts_moving_only': dict(gt_counts_moving),
        'pred_rule_counts': dict(pred_counts),
        'pred_rule_counts_moving_only': dict(pred_counts_moving),
        'ms_planner_mean': float(np.mean(t_plan)),
        'ms_checker_mean': float(np.mean(t_check)),
        'ms_corrector_mean': float(np.mean(t_correct)),
    }
    print("\n=== Replay baseline (tracked agents) ===")
    print(f"  agents: {n_moving + n_parked} tracked "
          f"({n_moving} moving, {n_parked} parked)")
    print(f"  L2 mean: {summary['l2_mean']:.2f} m")
    print(f"  GT safe rate: {summary['gt_safe_rate']:.2f}")
    print(f"  pred safe rate: {summary['pred_safe_rate']:.2f}")
    print(f"  pred fixed rate: {summary['pred_fixed_rate']:.2f}")
    print("  GT rule hits:")
    for rule, cnt in gt_counts.most_common():
        print(f"    {rule}: {cnt}")
    print("  GT rule hits (moving agents only):")
    for rule, cnt in gt_counts_moving.most_common():
        print(f"    {rule}: {cnt}")
    print("  pred rule hits (all agents):")
    for rule, cnt in pred_counts.most_common():
        print(f"    {rule}: {cnt}")
    print("  pred rule hits (moving agents only):")
    for rule, cnt in pred_counts_moving.most_common():
        print(f"    {rule}: {cnt}")
    print(f"  timing ms planner/checker/corrector: "
          f"{summary['ms_planner_mean']:.1f} / "
          f"{summary['ms_checker_mean']:.1f} / "
          f"{summary['ms_corrector_mean']:.1f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Annotation-replay evaluation")
    ap.add_argument("--data-root", default="C:/data/sets/nuscenes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene indices, default all")
    ap.add_argument("--out", default=None, help="JSON summary path")
    ap.add_argument("--cold", action="store_true",
                    help="single-frame cold predictions (default is the "
                         "N-2 warm path)")
    ap.add_argument("--corrector", default="heuristic",
                    choices=["heuristic", "qp"],
                    help="correction engine: heuristic fixpoint or "
                         "CorrectorQP constrained solve + re-validation")
    args = ap.parse_args()
    scenes = ([int(x) for x in args.scenes.split(",")]
              if args.scenes else None)
    summary = run_replay(args.data_root, args.device, scenes,
                         warm=not args.cold,
                         corrector_mode=args.corrector)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")

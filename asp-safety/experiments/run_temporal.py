"""Warm-vs-cold temporal harness (N-2).

Per scene: walk 4 history frames + the canonical eval frame through
predict_frame() (banks warm, img_metas + pose-derived ego_status
supplied), and compare L2 against single-frame cold predict() on the
eval frame. GT and map come from the canonical sample, so the only
difference is temporal history.

Usage: python experiments/run_temporal.py [--scenes 0,1] [--n-past 4]
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_planner import SparseDrivePlanner
from data_loader import NuScenesLoader, estimate_ego_status
from evaluation import MetricsCalculator
from run_calibration import interp_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="C:/data/sets/nuscenes")
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--n-past", type=int, default=4)
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    loader = NuScenesLoader(args.data_root, version="v1.0-mini")
    if loader.use_placeholder:
        raise RuntimeError(f"no nuScenes data at {args.data_root}")
    scenes = ([int(x) for x in args.scenes.split(",")]
              if args.scenes else list(range(loader.num_scenes)))
    planner = SparseDrivePlanner(
        str(base / "SparseDrive" / "ckpt" / "sparsedrive_stage2.pth"),
        device="cuda")

    cold_l2, warm_l2 = [], []
    print(f"Temporal warm-vs-cold on {len(scenes)} scenes "
          f"(n_past={args.n_past})...")
    for idx in scenes:
        frames = loader.get_scene_frames(idx, n_past=args.n_past)
        gt = loader.get_trajectory(idx)
        last = frames[-1]

        # Cold: single independent inference on the eval frame.
        out_c = planner.predict(
            last['images'].astype(np.float32), last['intrinsics'],
            last['extrinsics'], last['ego2global'], last['timestamp'])
        pred_times = (np.arange(6) + 1) * 0.5
        l2_c = MetricsCalculator.l2_error(
            out_c['trajectory'], interp_gt(gt, pred_times))
        cold_l2.append(l2_c)

        # Warm: walk the sequence without resetting; pose-derived status.
        planner.begin_sequence()
        out_w = None
        prev_g = None
        prev_t = None
        for fr in frames:
            if prev_g is None:
                status = np.zeros(18, dtype=np.float32)
            else:
                dt = fr['timestamp'] - prev_t
                status = estimate_ego_status(prev_g, fr['ego2global'], dt)
            out_w = planner.predict_frame(
                fr['images'].astype(np.float32), fr['intrinsics'],
                fr['extrinsics'], fr['ego2global'], fr['timestamp'],
                ego_status=status)
            prev_g, prev_t = fr['ego2global'], fr['timestamp']
        planner.end_sequence()
        l2_w = MetricsCalculator.l2_error(
            out_w['trajectory'], interp_gt(gt, pred_times))
        warm_l2.append(l2_w)
        print(f"  scene {idx}: cold L2={l2_c:.2f}m warm L2={l2_w:.2f}m")

    print("\n=== Temporal result ===")
    print(f"  cold mean L2: {np.mean(cold_l2):.2f} m")
    print(f"  warm mean L2: {np.mean(warm_l2):.2f} m")
    print(f"  delta (cold-warm): {np.mean(cold_l2) - np.mean(warm_l2):+.2f} m")


if __name__ == "__main__":
    main()

"""Nav-command sweep (P-13): which cmd head matches which GT geometry?

For each mini scene, runs real inference under cmd 0/1/2 and reports L2
vs GT plus the GT lateral displacement at +3 s. Goals:
  1. Verify the index convention (NUM_COMMANDS claims 0=straight,1=left,2=right).
  2. Measure how much of the 11.5 m L2 baseline is wrong-intent vs motion error.

Usage: python experiments/cmd_sweep.py [--scenes 0,1,2]
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_planner import SparseDrivePlanner
from data_loader import NuScenesLoader
from evaluation import MetricsCalculator
from run_calibration import interp_gt


def gt_intent(gt: np.ndarray) -> str:
    """Intent label from GT lateral displacement at +3 s (ego frame, y=left)."""
    y3 = float(gt[min(3, len(gt) - 1), 1])
    if y3 > 2.0:
        return "left"
    if y3 < -2.0:
        return "right"
    return "straight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="C:/data/sets/nuscenes")
    ap.add_argument("--scenes", default=None)
    args = ap.parse_args()

    base = Path(__file__).parent.parent
    loader = NuScenesLoader(args.data_root, version="v1.0-mini")
    scenes = ([int(x) for x in args.scenes.split(",")]
              if args.scenes else list(range(loader.num_scenes)))
    planner = SparseDrivePlanner(
        str(base / "SparseDrive" / "ckpt" / "sparsedrive_stage2.pth"),
        device="cuda")

    cmds = [np.array([1, 0, 0], dtype=np.float32),
            np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, 1], dtype=np.float32)]
    print("scene | gt_y3 | intent   | L2(cmd0) L2(cmd1) L2(cmd2) | winner")
    for idx in scenes:
        sample = loader.get_scene_sample(idx)
        gt = loader.get_trajectory(idx)
        intent = gt_intent(gt)
        y3 = float(gt[min(3, len(gt) - 1), 1])
        l2s = []
        for c in cmds:
            out = planner.predict(
                sample['images'].astype(np.float32),
                sample['intrinsics'], sample['extrinsics'],
                sample['ego2global'], sample['timestamp'], nav_cmd=c)
            pred = out['trajectory']
            pred_times = (np.arange(len(pred)) + 1) * 0.5
            l2s.append(MetricsCalculator.l2_error(
                pred, interp_gt(gt, pred_times)))
            time.sleep(0.2)
        winner = int(np.argmin(l2s))
        print(f"{idx:5d} | {y3:+6.2f} | {intent:8s} | "
              f"{l2s[0]:7.2f} {l2s[1]:7.2f} {l2s[2]:7.2f} | cmd{winner}")


if __name__ == "__main__":
    main()

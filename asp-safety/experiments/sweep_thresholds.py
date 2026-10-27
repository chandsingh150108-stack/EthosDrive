"""Threshold sweep proper (P-5): finding-rate-vs-value knees.

GPU-free: loads per-scene {gt, pred_1s, map_data} dumped by
run_calibration.py --dump-trajs, rewrites one .lp default at a time,
and counts GT false-positive scenes vs pred hit scenes per value.
Rule files are backed up and restored (try/finally).

Usage:
  python experiments/run_calibration.py --dump-trajs experiments/trajs.json ...
  python experiments/sweep_thresholds.py --dump experiments/trajs.json
"""

import sys
import re
import json
import shutil
import argparse
import itertools
import numpy as np
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from checker import TrajectoryChecker
from pipeline import SafetyLayerPipeline

BASE = Path(__file__).parent.parent
RULES = BASE / "rules"

# Agent-context params (no .lp rewrite; passed to run_all).
AGENT_SWEEP = [
    ("min_score", [0.1, 0.3, 0.5, 0.7]),
    ("max_agents", [5, 10, 20, 40]),
]

# (label, file, pattern-with-one-group, formatter, [values])
SWEEP = [
    ("bubble_5m", "collision_avoidance.lp",
     r"default_collision_dist2\((\d+)\)\.", "default_collision_dist2({}).",
     [90000, 160000, 250000, 360000]),
    ("long_gap", "collision_avoidance.lp",
     r"min_longitudinal_gap\((\d+)\)\.", "min_longitudinal_gap({}).",
     [200, 300, 400]),
    ("lat_clear", "collision_avoidance.lp",
     r"min_lateral_clearance\((\d+)\)\.", "min_lateral_clearance({}).",
     [75, 100, 150]),
    ("default_cap", "speed_regulation.lp",
     r"S2 > (\d+)", "S2 > {}",
     [1562500, 1929321, 2500000]),
    ("wrong_way_min", "lane_discipline.lp",
     r"min_wrong_way_speed2\((\d+)\)\.", "min_wrong_way_speed2({}).",
     [625, 2500, 10000]),
    ("worker_clear2", "emergency.lp",
     r"worker_clearance2\((\d+)\)\.", "worker_clearance2({}).",
     [22500, 90000, 360000]),
]


def set_param(fname, pattern, fmt, value):
    p = RULES / fname
    src = p.read_text()
    new, n = re.subn(pattern, fmt.format(value), src, count=1)
    assert n == 1, f"pattern not found once in {fname}: {pattern}"
    p.write_text(new)


def backup():
    bak = {}
    for _, fname, _, _, _ in SWEEP:
        p = RULES / fname
        if fname not in bak:
            bak[fname] = p.read_text()
    return bak


def restore(bak):
    for fname, src in bak.items():
        (RULES / fname).write_text(src)


def run_all(checker, scenes, min_score=0.3, max_agents=20):
    """Returns (gt_fp_scenes, pred_hit_scenes, gt_counts, pred_counts)."""
    gt_fp = pred_hit = 0
    gt_counts, pred_counts = Counter(), Counter()
    for idx, entry in scenes.items():
        gt = np.array(entry["gt"], dtype=np.float32)
        pred = np.array(entry["pred_1s"], dtype=np.float32)
        m = entry["map_data"]
        boxes = np.array(entry.get("boxes", np.zeros((0, 7))),
                         dtype=np.float32)
        vels = np.array(entry.get("velocities", np.zeros((0, 3))),
                        dtype=np.float32)
        scores = np.array(entry.get("scores", np.zeros((0,))),
                          dtype=np.float32)
        gt_agents = SafetyLayerPipeline._detections_to_trajectories(
            boxes, vels, scores, len(gt), max_agents=max_agents,
            min_score=min_score)
        pred_agents = SafetyLayerPipeline._detections_to_trajectories(
            boxes, vels, scores, len(pred), max_agents=max_agents,
            min_score=min_score)
        ok, v = checker.check(gt, map_data=m,
                              other_vehicles=gt_agents or None)
        if not ok:
            gt_fp += 1
            for d in v:
                if d.get("type", "violation") != "warning":
                    gt_counts[d["rule"]] += 1
        ok, v = checker.check(pred, map_data=m,
                              other_vehicles=pred_agents or None)
        if not ok:
            pred_hit += 1
            for d in v:
                if d.get("type", "violation") != "warning":
                    pred_counts[d["rule"]] += 1
    return gt_fp, pred_hit, gt_counts, pred_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    scenes = json.loads(Path(args.dump).read_text())
    print(f"sweeping {len(scenes)} dumped scenes "
          f"({', '.join(sorted(scenes))})")
    bak = backup()
    rows = []
    try:
        checker = TrajectoryChecker(str(RULES))
        # Baseline (current defaults) first.
        gt_fp, pred_hit, gc, pc = run_all(checker, scenes)
        rows.append(("baseline", "-", gt_fp, pred_hit, dict(gc),
                     dict(pc)))
        print(f"baseline: GT-FP scenes={gt_fp} pred-hit scenes={pred_hit}")
        for label, fname, pattern, fmt, values in SWEEP:
            for val in values:
                set_param(fname, pattern, fmt, val)
                c = TrajectoryChecker(str(RULES))
                gt_fp, pred_hit, gc, pc = run_all(c, scenes)
                rows.append((label, val, gt_fp, pred_hit, dict(gc),
                             dict(pc)))
                print(f"{label}={val}: GT-FP={gt_fp} pred-hit={pred_hit} "
                      f"GT={dict(gc)} pred={dict(pc)}")
        for label, values in AGENT_SWEEP:
            for val in values:
                kw = {"min_score": val} if label == "min_score" \
                    else {"max_agents": val}
                gt_fp, pred_hit, gc, pc = run_all(checker, scenes, **kw)
                rows.append((label, val, gt_fp, pred_hit, dict(gc),
                             dict(pc)))
                print(f"{label}={val}: GT-FP={gt_fp} pred-hit={pred_hit} "
                      f"GT={dict(gc)} pred={dict(pc)}")
    finally:
        restore(bak)

    print("\n=== Sweep knees (pick values where GT-FP drops without "
          "losing pred hits) ===")
    for label, val, gt_fp, pred_hit, gc, pc in rows:
        print(f"  {label}={val}: GT-FP scenes {gt_fp}, "
              f"pred-hit scenes {pred_hit}")
    if args.out:
        Path(args.out).write_text(json.dumps(
            [{"param": l, "value": v, "gt_fp": g, "pred_hit": p,
              "gt_counts": gc, "pred_counts": pc}
             for l, v, g, p, gc, pc in rows], indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

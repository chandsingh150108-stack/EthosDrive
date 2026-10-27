"""QP optimizer hyperparameter sweep (P6 follow-up, GPU-free).

Grid over projected-gradient knobs (lr, steps, lane_pull) on the
frozen trajs.json dump. Each combo runs CorrectorQP with
heuristic_fallback=False (pure optimizer power) over all 10 scenes;
fix rate, L2 delta on fixed scenes, and mean latency pick the knee.
Mirrors the P-5 threshold-sweep precedent: keep defaults unless a
combo shows a measured win with no regression.

Usage (from asp-safety/):
    python experiments/sweep_qp_params.py [--out FILE]
"""

import sys
import time
import json
import argparse
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from checker import TrajectoryChecker
from corrector import CorrectorQP
from pipeline import SafetyLayerPipeline

GRID = {
    'lr': [0.1, 0.2, 0.4],
    'steps': [30, 60],
    'lane_pull': [0.0, 0.05],
}


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


def run_sweep(dump_path: str):
    dump = json.loads(Path(dump_path).read_text())
    ids = sorted(dump.keys(), key=int)
    base_dir = Path(__file__).parent.parent
    checker = TrajectoryChecker(str(base_dir / "rules"))

    scenes = []
    for sid in ids:
        s = dump[sid]
        pred = np.asarray(s['pred_1s'], dtype=np.float32)
        gt = np.asarray(s['gt'], dtype=np.float32)
        boxes = np.asarray(s['boxes'], dtype=np.float32)
        vels = np.asarray(s.get('velocities', np.zeros((0, 3))),
                          dtype=np.float32)
        scores = np.asarray(s.get('scores', np.zeros((0,))),
                            dtype=np.float32)
        agents = SafetyLayerPipeline._detections_to_trajectories(
            boxes, vels, scores, len(pred))
        scenes.append((sid, pred, gt, s['map_data'], agents))

    combos = list(itertools.product(
        GRID['lr'], GRID['steps'], GRID['lane_pull']))
    rows = []
    for lr, steps, lane_pull in combos:
        qp = CorrectorQP(
            checker,
            optimizer_params={'lr': lr, 'steps': steps,
                              'lane_pull': lane_pull},
            heuristic_fallback=False)
        fixed = l2d = ms = n_unsafe = 0
        for sid, pred, gt, map_data, agents in scenes:
            kw = dict(map_data=map_data, other_vehicles=agents or None)
            ok_p, _ = checker.check(pred, **kw)
            if ok_p:
                continue
            n_unsafe += 1
            t0 = time.perf_counter()
            try:
                solved, _ = qp.correct(pred, **kw)
                dt = (time.perf_counter() - t0) * 1000.0
                ok_f, _ = checker.check(solved, **kw)
                if ok_f:
                    fixed += 1
                    l2d += _l2(pred, gt[:len(pred)]) - _l2(
                        solved, gt[:len(solved)])
                ms += dt
            except RuntimeError:
                ms += (time.perf_counter() - t0) * 1000.0
        row = {'lr': lr, 'steps': steps, 'lane_pull': lane_pull,
               'fixed': fixed, 'unsafe': n_unsafe,
               'fix_rate': fixed / n_unsafe if n_unsafe else 1.0,
               'l2_improve_mean': (l2d / fixed) if fixed else 0.0,
               'ms_mean': ms / n_unsafe if n_unsafe else 0.0}
        rows.append(row)
        print(f"  lr={lr} steps={steps} pull={lane_pull}: "
              f"fixed {fixed}/{n_unsafe} "
              f"L2+{row['l2_improve_mean']:.2f}m "
              f"{row['ms_mean']:.0f} ms")
    return {'grid': GRID, 'rows': rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QP param sweep")
    ap.add_argument("--dump", default=str(
        Path(__file__).parent / "trajs.json"))
    ap.add_argument("--out", default=None, help="JSON rows path")
    args = ap.parse_args()
    summary = run_sweep(args.dump)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")

"""P6-lite ablation: heuristic fixpoint vs CorrectorQP on frozen trajs.

GPU-free: replays experiments/trajs.json (GT + pred_1s + map + boxes
from a real calibration dump) through checker + both correctors and
reports fix rate, L2 delta, and corrector latency per engine. This is
the B1-vs-B2 evidence on logged plans; closed-loop rates need CARLA.

Usage (from asp-safety/):
    python experiments/ablate_corrector.py [--out FILE] [--scenes 0,1]
"""

import sys
import time
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from checker import TrajectoryChecker
from corrector import TrajectoryCorrector, CorrectorQP
from pipeline import SafetyLayerPipeline


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b))
    return float(np.mean(np.linalg.norm(a[:n] - b[:n], axis=1)))


def run_ablation(dump_path: str, scenes=None):
    base_dir = Path(__file__).parent.parent
    dump = json.loads(Path(dump_path).read_text())
    ids = sorted(dump.keys(), key=int)
    if scenes is not None:
        ids = [str(i) for i in scenes]

    checker = TrajectoryChecker(str(base_dir / "rules"))
    heur = TrajectoryCorrector(checker)
    qp = CorrectorQP(checker)
    # Bound-family variants run WITHOUT heuristic fallback so the fix
    # rate measures the family's own power (a fallback fix would be
    # credited to the heuristic, not the family).
    qp_caps = CorrectorQP(checker, bounds='caps',
                          heuristic_fallback=False)
    qp_boxes = CorrectorQP(checker, bounds='boxes',
                           heuristic_fallback=False)

    rows = []
    for sid in ids:
        s = dump[sid]
        gt = np.asarray(s['gt'], dtype=np.float32)
        pred = np.asarray(s['pred_1s'], dtype=np.float32)
        map_data = s['map_data']
        boxes = np.asarray(s['boxes'], dtype=np.float32)
        vels = np.asarray(s.get('velocities', np.zeros((0, 3))),
                          dtype=np.float32)
        scores = np.asarray(s.get('scores', np.zeros((0,))),
                            dtype=np.float32)
        agents = SafetyLayerPipeline._detections_to_trajectories(
            boxes, vels, scores, len(pred))
        kw = dict(map_data=map_data,
                  other_vehicles=agents or None)

        ok_p, v_p = checker.check(pred, **kw)
        l2_before = _l2(pred, gt[:len(pred)])
        row = {'scene': int(sid), 'pred_unsafe': not ok_p,
               'l2_before': l2_before,
               'pred_rules': sorted({v['rule'] for v in v_p})}

        for name, eng in (('heuristic', heur), ('qp', qp),
                            ('qp_caps', qp_caps), ('qp_boxes', qp_boxes)):
            t0 = time.perf_counter()
            try:
                fixed, entries = eng.correct(pred, **kw)
                ms = (time.perf_counter() - t0) * 1000.0
                ok_f, v_f = checker.check(fixed, **kw)
                row[f'{name}_fixed'] = bool(ok_f)
                row[f'{name}_ms'] = ms
                row[f'{name}_l2'] = _l2(fixed, gt[:len(fixed)])
                row[f'{name}_entries'] = len(entries)
                row[f'{name}_residual'] = sorted(
                    {v['rule'] for v in v_f})
            except RuntimeError as e:
                ms = (time.perf_counter() - t0) * 1000.0
                row[f'{name}_fixed'] = False
                row[f'{name}_ms'] = ms
                row[f'{name}_l2'] = None
                row[f'{name}_entries'] = 0
                row[f'{name}_residual'] = [f'GAVE_UP:{e}']
        rows.append(row)
        print(f"  scene {sid}: unsafe={row['pred_unsafe']} "
              f"heur_fixed={row['heuristic_fixed']} "
              f"({row['heuristic_ms']:.0f} ms) "
              f"qp_fixed={row['qp_fixed']} ({row['qp_ms']:.0f} ms) "
              f"caps={row['qp_caps_fixed']} boxes={row['qp_boxes_fixed']}")

    unsafe = [r for r in rows if r['pred_unsafe']]

    def _rate(key):
        return (sum(r[key] for r in unsafe) / len(unsafe)
                if unsafe else 1.0)

    def _mean(key):
        return float(np.mean([r[key] for r in rows]))
    summary = {
        'scenes': len(rows),
        'pred_unsafe': len(unsafe),
        'heuristic_fixed_rate': _rate('heuristic_fixed'),
        'qp_fixed_rate': _rate('qp_fixed'),
        'qp_caps_fixed_rate': _rate('qp_caps_fixed'),
        'qp_boxes_fixed_rate': _rate('qp_boxes_fixed'),
        'heuristic_ms_mean': _mean('heuristic_ms'),
        'qp_ms_mean': _mean('qp_ms'),
        'qp_caps_ms_mean': _mean('qp_caps_ms'),
        'qp_boxes_ms_mean': _mean('qp_boxes_ms'),
        'rows': rows,
    }
    print(f"\nunsafe {len(unsafe)}/{len(rows)} | "
          f"heuristic fixed {summary['heuristic_fixed_rate']:.2f} "
          f"({summary['heuristic_ms_mean']:.0f} ms) | "
          f"qp fixed {summary['qp_fixed_rate']:.2f} "
          f"({summary['qp_ms_mean']:.0f} ms) | "
          f"caps {summary['qp_caps_fixed_rate']:.2f} "
          f"boxes {summary['qp_boxes_fixed_rate']:.2f}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="QP vs heuristic ablation")
    ap.add_argument("--dump", default=str(
        Path(__file__).parent / "trajs.json"))
    ap.add_argument("--scenes", default=None,
                    help="comma-separated scene ids, default all")
    ap.add_argument("--out", default=None, help="JSON summary path")
    args = ap.parse_args()
    scenes = ([int(x) for x in args.scenes.split(",")]
              if args.scenes else None)
    summary = run_ablation(args.dump, scenes)
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"wrote {args.out}")

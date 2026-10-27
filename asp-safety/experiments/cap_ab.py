"""A/B: map budget caps vs verdicts + timing (forward item 3)."""

import sys
import time
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import data_loader as DL
from data_loader import NuScenesLoader
from checker import TrajectoryChecker

CONFIGS = [
    ("res2", None),
    ("res4", None),
]


def build_maps(cw_polys, sw_polys, cells):
    DL.MAP_CROSSWALK_POLYS = cw_polys
    DL.MAP_WALKWAY_POLYS = sw_polys
    DL.MAP_TILE_CELLS = cells
    loader = NuScenesLoader("C:/data/sets/nuscenes", version="v1.0-mini")
    return loader


def main():
    trajs = json.loads(
        Path(__file__).parent.joinpath("trajs.json").read_text())
    checker = TrajectoryChecker(
        str(Path(__file__).parent.parent / "rules"))
    for name, _ in CONFIGS:
        if name == "res2":
            DL.MAP_CENTERLINE_RES = 2.0
        else:
            DL.MAP_CENTERLINE_RES = 4.0
        DL.MAP_CROSSWALK_POLYS = 8
        DL.MAP_WALKWAY_POLYS = 8
        DL.MAP_TILE_CELLS = 48
        loader = NuScenesLoader("C:/data/sets/nuscenes",
                                version="v1.0-mini")
        n_tiles = 0
        times = []
        verdicts = {}
        for idx in ["0", "7", "1", "5", "8"]:
            m = loader.get_map_data(int(idx))
            n_tiles += len(m.get("crosswalks", [])) + len(
                m.get("sidewalks", []))
            gt = np.array(trajs[idx]["gt"], dtype=np.float32)
            pred = np.array(trajs[idx]["pred_1s"], dtype=np.float32)
            s = time.perf_counter()
            ok_g, v_g = checker.check(gt, map_data=m)
            ok_p, v_p = checker.check(pred, map_data=m)
            times.append((time.perf_counter() - s) * 1000.0)
            hard_g = sorted((d["rule"], d["time"]) for d in v_g
                            if d.get("type", "violation") != "warning")
            hard_p = sorted((d["rule"], d["time"]) for d in v_p
                            if d.get("type", "violation") != "warning")
            verdicts[idx] = (ok_g, ok_p, hard_g, hard_p)
        times.sort()
        print(f"{name}: tiles={n_tiles} "
              f"check_ms min={times[0]:.1f} p50={times[2]:.1f} "
              f"max={times[-1]:.1f}")
        for idx, (ok_g, ok_p, hard_g, hard_p) in verdicts.items():
            print(f"  scene {idx}: gt_safe={ok_g} pred_safe={ok_p} "
                  f"gt={hard_g} pred={hard_p}")


if __name__ == "__main__":
    main()

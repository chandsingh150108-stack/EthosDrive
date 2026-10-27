"""Probe: which cmd head curves which way (P-13 semantics check)."""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_planner import SparseDrivePlanner
from data_loader import NuScenesLoader

loader = NuScenesLoader("C:/data/sets/nuscenes", version="v1.0-mini")
planner = SparseDrivePlanner(
    str(Path(__file__).parent.parent
        / "SparseDrive" / "ckpt" / "sparsedrive_stage2.pth"),
    device="cuda")

for idx in [0, 2]:
    s = loader.get_scene_sample(idx)
    gt = loader.get_trajectory(idx)
    print(f"scene {idx} GT end: {gt[-1]}, GT[+3s]: {gt[min(3, len(gt) - 1)]}")
    for k in range(3):
        c = np.zeros(3, dtype=np.float32)
        c[k] = 1.0
        o = planner.predict(s["images"].astype(np.float32),
                            s["intrinsics"], s["extrinsics"],
                            s["ego2global"], s["timestamp"], nav_cmd=c)
        traj = o["trajectory"]
        print(f"  cmd{k} end: {traj[-1]} mid: {traj[2]}")

"""Axis-semantics check: model plan columns vs nuScenes GT future.

Question: does SparseDrive final_planning emit [x_forward, y_left]
(default assumption used by the ASP surface) or [y_lateral, x_forward]
(the swap seen on CARLA)?  Each scene's middle sample drives one cold
predict() call; its future ego poses (0.5 s cadence) are decomposed
into forward (ego +x) and lateral-left (ego +y) displacements and
printed next to the plan columns.

Run: python experiments/plan_axis_check.py [scene_start scene_end]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from data_loader import NuScenesLoader  # noqa: E402
from neural_planner import SparseDrivePlanner  # noqa: E402

DATA = "C:/data/sets/nuscenes"


def main():
    s0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    s1 = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    loader = NuScenesLoader(DATA, version="v1.0-mini")
    if loader.use_placeholder:
        raise RuntimeError(f"no nuScenes data at {DATA}")
    nusc = loader.nusc

    planner = SparseDrivePlanner(
        str(ROOT / "SparseDrive/ckpt/sparsedrive_stage2.pth"),
        device="cuda")

    for scene_idx in range(s0, min(s1, loader.num_scenes)):
        sample = loader.get_scene_sample(scene_idx)
        token = loader._sample_tokens[scene_idx]
        row = nusc.get("sample", token)

        lidar_sd = nusc.get(
            "sample_data", row["data"]["LIDAR_TOP"])
        ego_now = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
        p_now = np.array(ego_now["translation"], dtype=np.float64)

        # walk the future chain (0.5 s per sample)
        fut, token_f = [], token
        for _ in range(6):
            token_f = nusc.get("sample", token_f)["next"]
            if not token_f:
                break
            r = nusc.get("sample", token_f)
            sd = nusc.get("sample_data", r["data"]["LIDAR_TOP"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])
            fut.append(np.array(ep["translation"], dtype=np.float64))

        out = planner.predict(
            sample["images"].astype(np.float32),
            sample["intrinsics"], sample["extrinsics"],
            sample["ego2global"], sample["timestamp"])
        plan = out["trajectory"]
        nb = out["boxes"].shape[0]

        print(f"\nscene {scene_idx} (token {token[:8]}) nbox={nb}")
        print("  t     plan[c0]  plan[c1]   gt_fwd   gt_lat"
              "     err_std  err_swap")
        for i in range(1, 4):
            t_s = i * 1.0  # row mapping: row k = (k+1)*0.5 s
            idx = 2 * i - 1  # rows 1,3,5 = 1/2/3 s (replay convention)
            dpart = (fut[idx - 1] - p_now) if idx <= len(fut) else None
            if dpart is not None and sample["ego2global"] is not None:
                e2g = np.asarray(sample["ego2global"], np.float64)
                R = e2g[:3, :3]
                heading = R[:, 0]
                left = R[:, 1]
                fwd = float(dpart @ (heading / np.linalg.norm(heading)))
                lat = float(dpart @ (left / np.linalg.norm(left)))
            else:
                fwd = lat = np.nan
            c0, c1 = float(plan[idx][0]), float(plan[idx][1])
            err_std = (abs(c0 - fwd) + abs(c1 - lat)
                       if not np.isnan(fwd) else float("nan"))
            err_swap = (abs(c0 - lat) + abs(c1 - fwd)
                        if not np.isnan(fwd) else float("nan"))
            print(f"  {t_s:.1f}s  {c0:8.2f}  {c1:8.2f}"
                  f"  {fwd:8.2f}  {lat:8.2f}"
                  f"  {err_std:8.2f}  {err_swap:8.2f}")


if __name__ == "__main__":
    main()

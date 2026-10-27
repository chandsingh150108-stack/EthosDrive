"""VRAM gate probe for a visible (windowed, Epic) CARLA server.

Measures the shared-GPU budget exactly as the harness will spend it:
server rendering (window Epic) + 6-camera rig + planner model + one
warm predict_frame, then compares free VRAM against the >=1.5 GB gate
(P25-A retreat ladder).

    python experiments/carla_vram_probe.py --quality Epic --windowed
    python experiments/carla_vram_probe.py --quality Epic --windowed \
        --stream-pool 1200
    python experiments/carla_vram_probe.py --quality Low

Kill any previous server first (launch flags only apply to a fresh
process):  python experiments/carla_launch.py --kill

Final machine-readable line:
    VRAM_PROBE quality=... windowed=... stream_pool=...
               free_post_gb=... gate=... status=PASS|FAIL
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carla  # noqa: E402
import carla_launch  # noqa: E402
from carla_probe import (  # noqa: E402
    CAM_CONFIGS, IMG_H, IMG_W, build_extrinsic, transform_to_mat,
    try_spawn_near, world_to_ego_nu)

CAM_ORDER = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT",
             "BACK", "BACK_LEFT", "BACK_RIGHT"]


def vram_free_gb() -> float:
    import torch
    free, _ = torch.cuda.mem_get_info()
    return free / 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", default="Epic", choices=["Low", "Epic"])
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--stream-pool", type=int, default=None)
    ap.add_argument("--map", default=None)
    ap.add_argument("--gate", type=float, default=1.5,
                    help="GB that must remain free after warm predict")
    args = ap.parse_args()

    t0 = time.time()
    if not carla_launch.launch_server(
            args.map, quality=args.quality, windowed=args.windowed,
            stream_pool=args.stream_pool):
        print(f"VRAM_PROBE ... status=FAIL (no server; quality="
              f"{args.quality} windowed={args.windowed})")
        return 1
    client = carla.Client("localhost", 2000)
    client.set_timeout(300.0)
    world = client.get_world()
    print("map:", world.get_map().name, flush=True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn = sorted(world.get_map().get_spawn_points(),
                   key=lambda t: (t.location.x, t.location.y))[0]
    actors, cams, queues = [], [], {}
    pool_args = ("" if args.stream_pool is None
                 else f" pool={args.stream_pool}")
    tag = f"{args.quality}{pool_args} windowed={args.windowed}"
    try:
        ego = try_spawn_near(world, bp_lib,
                             bp_lib.find("vehicle.tesla.model3"),
                             spawn.location)
        if ego is None:
            print(f"VRAM_PROBE {tag} status=FAIL (ego spawn failed)")
            return 1
        actors.append(ego)
        for name, yaw, fov, dx, dy, dz in CAM_CONFIGS:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(IMG_W))
            bp.set_attribute("image_size_y", str(IMG_H))
            bp.set_attribute("fov", str(fov))
            transform = carla.Transform(
                carla.Location(x=dx, y=dy, z=dz),
                carla.Rotation(pitch=-8.0 if yaw == 0.0 else 0.0,
                               yaw=yaw))
            cam = world.spawn_actor(bp, transform, attach_to=ego)
            cams.append((name, cam))
        actors.extend(c for _, c in cams)

        queues = {name: [] for name, _ in cams}
        for name, cam in cams:
            cam.listen(queues[name].append)
        deadline = time.time() + 60.0
        while time.time() < deadline and any(
                len(q) == 0 for q in queues.values()):
            world.tick()
            time.sleep(0.05)
        missing = [n for n, q in queues.items() if not q]
        if missing:
            print(f"VRAM_PROBE {tag} status=FAIL (no frames {missing})")
            return 1

        free_server = vram_free_gb()
        print(f"free VRAM (server+window+rig): {free_server:.2f} GB",
              flush=True)

        model_path = (Path(__file__).resolve().parents[1]
                      / "SparseDrive/ckpt/sparsedrive_stage2.pth")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from neural_planner import SparseDrivePlanner  # noqa: E402
        planner = SparseDrivePlanner(str(model_path), device="cuda")
        free_model = vram_free_gb()
        print(f"free VRAM (after model load): {free_model:.2f} GB",
              flush=True)

        t_world_veh = transform_to_mat(ego.get_transform())
        ego2global = world_to_ego_nu(t_world_veh)
        K = np.zeros((6, 3, 3), dtype=np.float32)
        C = np.zeros((6, 4, 4), dtype=np.float32)
        stack = np.zeros((6, 3, IMG_H, IMG_W), dtype=np.float32)
        for idx, (name, cam) in enumerate(cams):
            fov = next(f for n, _, f, _, _, _ in CAM_CONFIGS
                       if n == name)
            fx = (IMG_W / 2) / math.tan(math.radians(fov) / 2)
            cx, cy = IMG_W / 2.0, IMG_H / 2.0
            K[idx] = [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
            C[idx] = build_extrinsic(t_world_veh,
                                     transform_to_mat(
                                         cam.get_transform()))
            img = queues[name][-1]
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
                IMG_H, IMG_W, 4)
            stack[idx] = np.ascontiguousarray(
                arr[:, :, 2::-1].transpose(2, 0, 1)).astype(np.float32)

        planner.begin_sequence()
        t_inf = time.time()
        out = planner.predict_frame(stack, K, C,
                                    ego2global.astype(np.float32),
                                    time.time())
        dt_inf = time.time() - t_inf
        planner.end_sequence()
        free_post = vram_free_gb()
        traj = out["trajectory"]
        print(f"free VRAM (after warm frame): {free_post:.2f} GB "
              f"(warm wall {dt_inf:.2f} s; traj {traj.shape})", flush=True)

        status = "PASS" if free_post >= args.gate else "FAIL"
        print(f"VRAM_PROBE quality={args.quality} "
              f"windowed={args.windowed} "
              f"stream_pool={args.stream_pool} "
              f"free_post_gb={free_post:.2f} "
              f"gate={args.gate:.2f} status={status}")
        return 0 if status == "PASS" else 1
    finally:
        for _, cam in cams:
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        for a in actors:
            try:
                if a and a.is_alive:
                    a.destroy()
            except Exception:
                pass
        settings = world.get_settings()
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)
        print(f"probe wall {time.time() - t0:.1f} s "
              "(server kept alive)", flush=True)


if __name__ == "__main__":
    sys.exit(main())

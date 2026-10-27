"""Planner sanity on CARLA imagery: repeated warm frames while ego drives.

Answers three questions before the next closed-loop S1 attempt:
  1. Does the plan stabilize after the first warm frame, or keep
     predicting a large lateral swerve on a straight road?
  2. Are detected boxes sane in the planner frame (front objects
     should sit on +x with |y| small)?
  3. Are the camera images right-side up (sky brighter than road
     on CAM_FRONT)?

Run: python experiments/carla_sanity.py
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import carla  # noqa: E402
import carla_launch  # noqa: E402
from carla_probe import (  # noqa: E402
    CAM_CONFIGS, IMG_H, IMG_W, build_extrinsic, transform_to_mat,
)
from carla_context import (  # noqa: E402
    to_planner_ego_matrix, transform_matrix,
)
from data_loader import estimate_ego_status  # noqa: E402
from neural_planner import SparseDrivePlanner  # noqa: E402

FIXED_DT = 0.05
FRAME_EVERY = 10   # 0.5 s sim between planner calls
N_FRAMES = 6


def grab_stack(world, queues):
    df = min(q[-1].frame for q in queues.values())
    stack = np.zeros((6, 3, IMG_H, IMG_W), np.float32)
    idx = 0
    for name, *_ in CAM_CONFIGS:
        ims = [im for im in queues[name] if im.frame == df]
        if not ims:
            return None
        a = (np.frombuffer(ims[-1].raw_data, np.uint8)
             .reshape(IMG_H, IMG_W, 4)[:, :, 2::-1])
        stack[idx] = np.ascontiguousarray(
            a.transpose(2, 0, 1)).astype(np.float32)
        idx += 1
    return stack


def main():
    carla_launch.launch_server(None)
    client = carla.Client("localhost", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FIXED_DT
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn = sorted(world.get_map().get_spawn_points(),
                   key=lambda t: (t.location.x, t.location.y))[0]

    actors = []
    queues = {}
    try:
        ego = world.spawn_actor(
            bp_lib.find("vehicle.tesla.model3"), spawn)
        if ego is None:
            print("[sanity] FAIL: ego spawn")
            return 1
        actors.append(ego)
        quo = {}
        for name, yaw, fov, dx, dy, dz in CAM_CONFIGS:
            bp = world.get_blueprint_library().find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(IMG_W))
            bp.set_attribute("image_size_y", str(IMG_H))
            bp.set_attribute("fov", str(fov))
            tf = carla.Transform(
                carla.Location(x=dx, y=dy, z=dz),
                carla.Rotation(pitch=-8.0 if yaw == 0.0 else 0.0,
                               yaw=yaw))
            cam = world.spawn_actor(bp, tf, attach_to=ego)
            actors.append(cam)
            quo[name] = []
            cam.listen(quo[name].append)
        queues = quo

        deadline = time.time() + 30.0
        while time.time() < deadline and any(
                len(q) == 0 for q in queues.values()):
            world.tick()
            time.sleep(0.02)
        missing = [n for n, q in queues.items() if not q]
        if missing:
            print(f"[sanity] FAIL: no frames for {missing}")
            return 1
        for q in queues.values():
            q.clear()
        world.tick()

        # K and C computed once, exactly like the probe/bridge setup
        import math
        K = np.zeros((6, 3, 3), dtype=np.float64)
        for idx, (name, yaw, fov, _, _, _) in enumerate(CAM_CONFIGS):
            fx = (IMG_W / 2) / math.tan(math.radians(fov) / 2)
            cx, cy = IMG_W / 2, IMG_H / 2
            K[idx] = [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
        t_world_veh = transform_to_mat(ego.get_transform())
        C = np.zeros((6, 4, 4), dtype=np.float64)
        for idx, (name, cam) in enumerate(
                [(n, a) for n, a in zip(
                    [c[0] for c in CAM_CONFIGS],
                    [a for a in actors[1:]])]):
            C[idx] = build_extrinsic(
                t_world_veh, transform_to_mat(cam.get_transform()))

        # orientation probe: sky vs road on CAM_FRONT
        deadline = time.time() + 10.0
        while time.time() < deadline and any(
                len(q) == 0 for q in queues.values()):
            world.tick()
            time.sleep(0.02)
        df = min(q[-1].frame for q in queues.values())
        front = [im for im in queues["FRONT"] if im.frame == df][-1]
        a = (np.frombuffer(front.raw_data, np.uint8)
             .reshape(IMG_H, IMG_W, 4)[:, :, 2::-1])
        top, bot = float(a[:20].mean()), float(a[-20:].mean())
        print(f"[sanity] CAM_FRONT top20={top:.1f} bottom20={bot:.1f} "
              + ("sky-top brighter: OK" if top > bot
                 else "ROAD-top brighter: FLIPPED?"), flush=True)

        model_path = (Path(__file__).resolve().parents[1]
                      / "SparseDrive/ckpt/sparsedrive_stage2.pth")
        planner = SparseDrivePlanner(str(model_path), device="cuda")
        planner.begin_sequence()

        prev_ego_g = None
        printed = 0
        tick_i = 0
        while printed < N_FRAMES:
            world.tick()
            tick_i += 1
            ctrl = ego.get_control()
            ctrl.throttle = 0.45
            ctrl.steer = 0.0
            ctrl.brake = 0.0
            ctrl.hand_brake = False
            ego.apply_control(ctrl)

            if tick_i % FRAME_EVERY:
                continue
            stack = grab_stack(world, queues)
            if stack is None:
                print(f"[sanity] frame{printed}: stack miss "
                      f"(frames not synced)", flush=True)
                continue
            ego_g = to_planner_ego_matrix(
                transform_matrix(ego.get_transform()))
            if prev_ego_g is None:
                status = np.zeros(18, dtype=np.float32)
            else:
                status = estimate_ego_status(prev_ego_g, ego_g, 0.5)
            prev_ego_g = ego_g
            t0 = time.perf_counter()
            out = planner.predict_frame(
                stack, K.astype(np.float32), C.astype(np.float32),
                ego_g.astype(np.float32), time.time(), ego_status=status)
            ms = (time.perf_counter() - t0) * 1000.0
            traj = out["trajectory"]
            nb = out["boxes"].shape[0]
            scores = (out["scores"][:3].tolist() if nb else [])
            box0 = (out["boxes"][0, :3].round(1).tolist() if nb else [])
            print(f"[sanity] frame{printed} {ms:.0f}ms "
                  f"traj={traj.round(2).tolist()} "
                  f"nbox={nb} scores={scores} box0={box0}", flush=True)
            printed += 1
        print("[sanity] DONE")
        return 0
    finally:
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        if world is not None:
            settings.synchronous_mode = False
            world.apply_settings(settings)


if __name__ == "__main__":
    sys.exit(main())

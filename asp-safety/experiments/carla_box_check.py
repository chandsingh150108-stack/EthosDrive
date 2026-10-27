"""Detector mirror-sign check on CARLA.

Places a static vehicle at:
  A) (x+10, y+0, z) relative to ego heading (dead ahead)
  B) (x+10, y-5, z)  (5 m to the ego's LEFT in the planner frame --
     CARLA world y is right-handed w.r.t. heading, so left = minus-y)
and prints the planner's top-scoring detected box (lat, fwd, z).

Verdict rules (lat column after our (lat,fwd) emission convention):
  A box  ~ (0, 10)      -> correct center
  B box  ~ (+5, 10)     -> planner +lat = CARLA-LEFT (convention OK)
  B box  ~ (-5, 10)     -> mirrored detections (planner sees our
                           left as its right; every downstream
                           y-sign must be re-derived)

Run: python experiments/carla_box_check.py
"""
import math
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
from neural_planner import SparseDrivePlanner  # noqa: E402

FIXED_DT = 0.05


def build_rig(world, bp_lib, ego):
    cams = []
    for name, yaw, fov, dx, dy, dz in CAM_CONFIGS:
        bp = bp_lib.find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(IMG_W))
        bp.set_attribute("image_size_y", str(IMG_H))
        bp.set_attribute("fov", str(fov))
        tf = carla.Transform(
            carla.Location(x=dx, y=dy, z=dz),
            carla.Rotation(pitch=-8.0 if yaw == 0.0 else 0.0,
                           yaw=yaw))
        cam = world.spawn_actor(bp, tf, attach_to=ego)
        cams.append((name, cam))
    return cams


def grab_stack(world, queues):
    df = min(q[-1].frame for q in queues.values())
    stack = np.zeros((6, 3, IMG_H, IMG_W), np.float32)
    for idx, (name, *_rest) in enumerate(CAM_CONFIGS):
        ims = [im for im in queues[name] if im.frame == df]
        if not ims:
            return None
        a = (np.frombuffer(ims[-1].raw_data, np.uint8)
             .reshape(IMG_H, IMG_W, 4)[:, :, 2::-1])
        stack[idx] = np.ascontiguousarray(
            a.transpose(2, 0, 1)).astype(np.float32)
    return stack


def run_case(world, planner, K, C, ego, queues, dv, dy_left):
    """Spawn proxy at (10 m fwd, dy_left meters to planner-LEFT):
    CARLA world left-of-heading = rotate fwd vector +90 deg?? no:
    CARLA heading fwd f; the LEFT side is f rotated +90 deg
    counterclockwise in the XY plane about z. Compute explicitly."""
    f = ego.get_transform().get_forward_vector()
    f = np.array([f.x, f.y])
    if np.linalg.norm(f) < 1e-6:
        return
    f /= np.linalg.norm(f)
    left = np.array([-f[1], f[0]])  # +90 deg CCW = left side of heading
    pos = ego.get_location()
    p = pos.x + 10.0 * f[0] + dy_left * left[0]
    q = pos.y + 10.0 * f[1] + dy_left * left[1]

    bp = world.get_blueprint_library().find("vehicle.audi.a2")
    tf = carla.Transform(carla.Location(x=p, y=q, z=0.5))
    proxy = world.try_spawn_actor(bp, tf)
    if proxy is None:
        print(f"[boxcheck] spawn failed p=({p:.1f},{q:.1f})")
        return None
    for _ in range(6):
        world.tick()
        time.sleep(0.02)
    stack = grab_stack(world, queues)
    if stack is None:
        proxy.destroy()
        print("[boxcheck] stack miss")
        return None
    ego_g = to_planner_ego_matrix(
        transform_matrix(ego.get_transform())).astype(np.float32)
    out = planner.predict_frame(
        stack, K.astype(np.float32), C.astype(np.float32), ego_g,
        time.time())
    boxes = out["boxes"]
    scores = out["scores"]
    order = np.argsort(scores)[::-1]
    print(f"[boxcheck] proxy at fwd=10 left={dy_left:+.0f}", flush=True)
    for k in order[:5]:
        b = boxes[k]
        print(f"    box{int(k)} score={scores[k]:.2f} "
              f"lat={b[0]:+.2f} fwd={b[1]:+.2f} z={b[2]:+.2f}",
              flush=True)
    proxy.destroy()
    for _ in range(3):
        world.tick()
    return True


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

    actors, queues = [], {}
    try:
        ego = world.spawn_actor(
            bp_lib.find("vehicle.tesla.model3"), spawn)
        actors.append(ego)
        rigs = build_rig(world, bp_lib, ego)
        cams = []
        queues = {}
        for name, cam in rigs:
            cams.append((name, cam))
            actors.append(cam)
            queues[name] = []
            cam.listen(queues[name].append)

        deadline = time.time() + 30.0
        while time.time() < deadline and any(
                len(q) == 0 for q in queues.values()):
            world.tick()
            time.sleep(0.02)
        for q in queues.values():
            q.clear()
        world.tick()

        import math as _m
        K = np.zeros((6, 3, 3), dtype=np.float64)
        for idx, (name, yaw, fov, *_r) in enumerate(CAM_CONFIGS):
            fx = (IMG_W / 2) / _m.tan(_m.radians(fov) / 2)
            K[idx] = [[fx, 0, IMG_W / 2], [0, fx, IMG_H / 2], [0, 0, 1]]
        t_world_veh = transform_to_mat(ego.get_transform())
        C = np.zeros((6, 4, 4), dtype=np.float64)
        for idx, (name, cam) in enumerate(cams):
            C[idx] = build_extrinsic(
                t_world_veh, transform_to_mat(cam.get_transform()))

        model_path = (Path(__file__).resolve().parents[1]
                      / "SparseDrive/ckpt/sparsedrive_stage2.pth")
        planner = SparseDrivePlanner(str(model_path), device="cuda")
        planner.begin_sequence()

        # warm frame on empty scene first
        run_case(world, planner, K, C, ego, queues, 0.0, 0.0)
        run_case(world, planner, K, C, ego, queues, 10.0, 0.0)
        run_case(world, planner, K, C, ego, queues, 10.0, +5.0)
        run_case(world, planner, K, C, ego, queues, 10.0, -5.0)
        print("[boxcheck] DONE")
        return 0
    finally:
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == "__main__":
    sys.exit(main())

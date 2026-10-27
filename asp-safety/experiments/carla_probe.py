"""One-shot CARLA rig + coordinate probe (P25-A).

Verifies the 6-camera nuScenes-like rig and the coordinate transform
chain BEFORE anything downstream consumes it:

  1. reuse-or-launch server, connect, print versions/map
  2. synchronous mode, fixed dt 0.05
  3. spawn ego + 6x rgb 704x256 cameras at nuScenes-like mounts
  4. K from fov (square pixels, cx/cy at image center)
  5. C_i: planner-ego -> CV camera frame, composed from the sensor
     transform with the CARLA(UE4)->nuScenes handedness adaptation
     (y-flip F: CARLA vehicle y-right -> nuScenes ego y-left, then the
     optical-axis remap S: CARLA cam (x fwd, y right, z up) -> CV cam
     (x right, y down, z fwd)):
         C_i = S @ T_v_c @ F          (4x4, camera-to-ego semantics)
     so the planner's projection_mat = K @ C_i[:3] projects EGO-frame
     points directly to pixels with u = fx*x/z + cx.
  6. projection sanity: actors 12 m ahead 2 m LEFT / 2 m RIGHT of ego
     must project into CAM_FRONT with u < cx / u > cx respectively
     (sign errors here poison every later phase).
  7. VRAM report around one warm predict_frame on a real CARLA stack.

Saves experiments/probe_out/cam_front.png + probe_calib.npz, prints
PROBE OK on success.

Run:  python experiments/carla_probe.py [--map NAME]
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carla_launch  # noqa: E402
import carla  # noqa: E402

IMG_W, IMG_H = 704, 256

# name -> (yaw_deg, fov_deg, dx, dy, dz) relative to ego, meters.
# y positive = LEFT in the planner frame; CARLA uses y-right, so the
# physical lateral offset is applied directly in CARLA convention.
CAM_CONFIGS = [
    ("FRONT", 0.0, 52.0, 1.8, 0.0, 2.0),
    ("FRONT_LEFT", 55.0, 52.0, 1.6, 0.10, 2.0),
    ("FRONT_RIGHT", -55.0, 52.0, 1.6, -0.10, 2.0),
    ("BACK", 180.0, 65.0, -1.8, 0.0, 2.0),
    ("BACK_LEFT", 110.0, 65.0, -1.6, 0.60, 2.0),
    ("BACK_RIGHT", -110.0, 65.0, -1.6, -0.60, 2.0),
]

FLIP = np.diag([1.0, -1.0, 1.0, 1.0])


def remap_s_mat() -> np.ndarray:
    """S: CARLA cam (x fwd, y right, z up) -> CV cam (x right, y down, z fwd)."""
    s = np.zeros((4, 4))
    s[0, 1] = 1.0    # x_cv = y_carla (right)
    s[1, 2] = -1.0   # y_cv = -z_carla (down)
    s[2, 0] = 1.0    # z_cv = x_carla (fwd)
    s[3, 3] = 1.0
    return s


def transform_to_mat(t) -> np.ndarray:
    return np.asarray(t.get_matrix(), dtype=np.float64)


def world_to_ego_nu(t_world_vehicle: np.ndarray) -> np.ndarray:
    """CARLA vehicle pose -> nuScenes ego2global matrix.

    p_world = T_world_veh @ F @ p_ego  =>  ego2global = T_world_veh @ F.
    """
    return t_world_vehicle @ FLIP


def build_extrinsic(t_world_veh, t_world_cam) -> np.ndarray:
    """C_i: maps planner-ego-frame points to the CV camera frame."""
    t_v_c = np.linalg.inv(t_world_veh) @ t_world_cam  # cam -> CARLA vehicle
    return remap_s_mat() @ t_v_c @ FLIP


def pinch_pixel(p_cam: np.ndarray, fx: float, fy: float,
                cx: float, cy: float) -> tuple:
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return float(u), float(v)


def try_spawn_near(world, bp_lib, blueprint, at, tries=5):
    """Spawn at `at`, nudging fallback offsets if the spot is blocked."""
    for i in range(tries):
        loc = carla.Location(at)  # noqa: F821
        if i:
            loc.x += 0.5 * i * (1 if i % 2 else -1)
            loc.y += 0.5 * i
        actor = world.try_spawn_actor(
            blueprint, carla.Transform(loc))  # noqa: F821
        if actor is not None:
            return actor
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None)
    args = ap.parse_args()

    t0 = time.time()
    done = []

    if not carla_launch.launch_server(args.map):
        print("PROBE FAIL: could not obtain a server")
        return 1

    client = carla.Client("localhost", 2000)
    client.set_timeout(300.0)
    print("server version:", client.get_server_version(), flush=True)
    world = client.get_world()
    print("map:", world.get_map().name, flush=True)
    if args.map and args.map not in world.get_map().name:
        print(f"loading map {args.map} ...", flush=True)
        world = client.load_world(args.map)
        print("map now:", world.get_map().name, flush=True)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    spawn = sorted(world.get_map().get_spawn_points(),
                   key=lambda t: (t.location.x, t.location.y))[0]

    actors = []
    cams = []
    queues = {}
    try:
        ego = try_spawn_near(world, bp_lib,
                             bp_lib.find("vehicle.tesla.model3"),
                             spawn.location)
        if ego is None:
            print("PROBE FAIL: ego spawn failed")
            return 1
        actors.append(ego)

        for name, yaw, fov, dx, dy, dz in CAM_CONFIGS:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(IMG_W))
            bp.set_attribute("image_size_y", str(IMG_H))
            bp.set_attribute("fov", str(fov))
            transform = carla.Transform(  # noqa: F821
                carla.Location(x=dx, y=dy, z=dz),  # noqa: F821
                carla.Rotation(pitch=-8.0 if yaw == 0.0 else 0.0,  # noqa: F821
                               yaw=yaw))
            cam = world.spawn_actor(bp, transform, attach_to=ego)
            cams.append((name, cam))
        actors.extend(c for _, c in cams)

        queues = {name: [] for name, _ in cams}
        for name, cam in cams:
            cam.listen(queues[name].append)

        world.tick()
        deadline = time.time() + 60.0
        while time.time() < deadline and any(
                len(q) == 0 for q in queues.values()):
            world.tick()
            time.sleep(0.05)
        missing = [n for n, q in queues.items() if not q]
        if missing:
            print(f"PROBE FAIL: no frames within 60 s for {missing}")
            return 1

        # ---- intrinsics ----
        K = np.zeros((6, 3, 3), dtype=np.float64)
        fovs = {}
        for idx, (name, _) in enumerate(cams):
            fov = next(f for n, _, f, _, _, _ in CAM_CONFIGS if n == name)
            fovs[name] = fov
            fx = (IMG_W / 2) / math.tan(math.radians(fov) / 2)
            cx, cy = IMG_W / 2, IMG_H / 2
            K[idx] = [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]

        # ---- extrinsics ----
        t_world_veh = transform_to_mat(ego.get_transform())
        ego2global = world_to_ego_nu(t_world_veh)
        C = np.zeros((6, 4, 4), dtype=np.float64)
        for idx, (name, cam) in enumerate(cams):
            C[idx] = build_extrinsic(t_world_veh,
                                     transform_to_mat(cam.get_transform()))
        print("per-cam (yaw_deg, fov_deg):",
              {name: (cfg[1], fovs[name])
               for name, cfg in zip([n for n, _ in cams], CAM_CONFIGS)},
              flush=True)

        # ---- projection sanity actors ----
        test_actors = []
        for label, y_left in (("LEFT", 2.0), ("RIGHT", -2.0)):
            loc = carla.Location(  # noqa: F821
                x=spawn.location.x + 12.0,
                y=spawn.location.y - y_left,  # planner +y(left) = CARLA -y
                z=spawn.location.z)
            a = try_spawn_near(
                world, bp_lib, bp_lib.find("vehicle.tesla.model3"), loc)
            if a is None:
                print(f"PROBE FAIL: {label} test actor spawn failed")
                return 1
            test_actors.append((label, a))
        actors.extend(a for _, a in test_actors)
        world.tick()
        time.sleep(0.2)

        t_world_veh = transform_to_mat(ego.get_transform())
        C_front = C[0]
        K_front = K[0]
        fx, fy = K_front[0, 0], K_front[1, 1]
        cx, cy = K_front[0, 2], K_front[1, 2]

        results = {}
        for label, actor in test_actors:
            loc = actor.get_location()
            p_world = np.array([loc.x, loc.y, loc.z, 1.0])
            p_ego_nu = FLIP @ (np.linalg.inv(t_world_veh) @ p_world)
            p_cam = C_front @ p_ego_nu
            u, v = pinch_pixel(p_cam[:3], fx, fy, cx, cy)
            inside = (0 <= u < IMG_W) and (0 <= v < IMG_H)
            results[label] = (u, v, inside, float(p_cam[2]))
            print(f"projection {label}: u={u:.1f} v={v:.1f} "
                  f"in-image={inside} z_cam={p_cam[2]:.2f}", flush=True)

        checks = [
            results["LEFT"][2] and results["RIGHT"][2],
            results["LEFT"][0] < cx < results["RIGHT"][0],
            results["LEFT"][3] > 0 and results["RIGHT"][3] > 0,
        ]
        if not all(checks):
            print("PROBE FAIL: projection checks "
                  "(in-image, left<u_center<right, z>0):", checks)
            return 1

        # ---- save calibration + front image ----
        out_dir = Path(__file__).resolve().parent / "probe_out"
        out_dir.mkdir(exist_ok=True)
        try:
            import cv2
            has_cv = True
        except ImportError:
            has_cv = False
        img = queues["FRONT"][0]
        arr_u8 = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
            IMG_H, IMG_W, 4)
        rgb = np.ascontiguousarray(arr_u8[:, :, 2::-1])  # BGRA -> RGB, top-down
        if has_cv:
            cv2.imwrite(str(out_dir / "cam_front.png"), rgb)
        else:
            rgb.dump(str(out_dir / "cam_front.npy"))
        np.savez(out_dir / "probe_calib.npz", K=K, C=C,
                 ego2global=ego2global,
                 camera_names=[n for n, _ in cams])

        # ---- VRAM around one warm predict_frame ----
        import torch
        free0, total0 = torch.cuda.mem_get_info()
        free0_gb, total0_gb = free0 / 2**30, total0 / 2**30
        print(f"VRAM before model load: free={free0_gb:.2f}/"
              f"{total0_gb:.2f} GB (server textures only)", flush=True)

        model_path = (Path(__file__).resolve().parents[1]
                      / "SparseDrive/ckpt/sparsedrive_stage2.pth")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from neural_planner import SparseDrivePlanner  # noqa: E402
        planner = SparseDrivePlanner(str(model_path), device="cuda")
        free1, _ = torch.cuda.mem_get_info()
        print(f"VRAM after model load:  free={free1 / 2**30:.2f} GB",
              flush=True)

        stack = np.zeros((6, 3, IMG_H, IMG_W), dtype=np.float32)
        order = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT",
                 "BACK", "BACK_LEFT", "BACK_RIGHT"]
        for idx, name in enumerate(order):
            img = queues[name][-1]
            arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
                IMG_H, IMG_W, 4)
            stack[idx] = np.ascontiguousarray(
                arr[:, :, 2::-1].transpose(2, 0, 1)).astype(np.float32)

        planner.begin_sequence()
        t_inf = time.time()
        out = planner.predict_frame(stack, K.astype(np.float32),
                                    C.astype(np.float32),
                                    ego2global.astype(np.float32),
                                    time.time())
        dt_inf = time.time() - t_inf
        planner.end_sequence()
        free2, _ = torch.cuda.mem_get_info()
        free2_gb = free2 / 2**30
        traj = out["trajectory"]
        print(f"VRAM after warm frame:  free={free2_gb:.2f} GB; "
              f"traj shape={traj.shape} dtype={traj.dtype}; "
              f"p0={traj[0].tolist()}", flush=True)
        print(f"warm predict_frame wall: {dt_inf:.2f} s", flush=True)
        print(f"VRAM headroom: {free2_gb:.2f} GB "
              + ("OK (>=1.5 GB)" if free2_gb >= 1.5
                 else "LOW (<1.5 GB; retreat ladder at nextstep.md 9)"),
              flush=True)
        if free2_gb >= 1.5:
            done.append("VRAM-headroom")

        done += ["rig+K+C", "projection-sanity", "warm-frame"]
        print(f"saved: {out_dir / 'probe_calib.npz'} (+ cam_front.png); "
              f"total wall {time.time() - t0:.1f} s")
        print("PROBE OK:", "; ".join(done))
        return 0
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


if __name__ == "__main__":
    sys.exit(main())

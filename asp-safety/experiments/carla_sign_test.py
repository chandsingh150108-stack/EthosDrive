"""Steering sign calibration (P25-B verification step).

Bridge convention: steer command +1 should turn the planner-ego
(y LEFT-positive) heading counterclockwise. This test applies
steer=+0.8, measures the planner-ego heading rotation, and prints
SIGN OK (+evidence) or SIGN FLIP NEEDED. Run once, then frozen.
"""

import sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))


def main() -> int:
    import carla_launch

    if not carla_launch.launch_server(None):
        return 1
    import carla

    client = carla.Client("localhost", 2000)
    client.set_timeout(300.0)
    world = client.get_world()
    s = world.get_settings()
    s.synchronous_mode = True
    s.fixed_delta_seconds = 0.05
    world.apply_settings(s)

    bp_lib = world.get_blueprint_library()
    spawn = sorted(world.get_map().get_spawn_points(),
                   key=lambda t: (t.location.x, t.location.y))[0]
    veh = world.try_spawn_actor(bp_lib.find("vehicle.tesla.model3"), spawn)
    try:
        for _ in range(10):
            world.tick()
        h0 = veh.get_transform().get_forward_vector()
        for _ in range(30):
            veh.apply_control(carla.VehicleControl(steer=0.8, throttle=0.4))
            world.tick()
        h1 = veh.get_transform().get_forward_vector()

        p0 = np.array([h0.x, -h0.y])
        p1 = np.array([h1.x, -h1.y])
        cross = p0[0] * p1[1] - p0[1] * p1[0]
        angle = np.degrees(np.arctan2(cross, float(np.dot(p0, p1))))
        w0 = np.array([h0.x, h0.y])
        w1 = np.array([h1.x, h1.y])
        wcross = w0[0] * w1[1] - w0[1] * w1[0]
        wangle = np.degrees(np.arctan2(wcross, float(np.dot(w0, w1))))
        print(f"steer=+0.8 -> planner-ego heading change: {angle:+.2f} deg "
              f"(CARLA world: {wangle:+.2f} deg)")
        if angle > 5:
            print("SIGN OK: +steer turns planner-LEFT; bridge steer_sign="
                  "+1 is correct")
            return 0
        if angle < -5:
            print("SIGN FLIP NEEDED: set bridge cfg steer_sign=-1")
            return 1
        print("SIGN AMBIGUOUS: vehicle may not have moved enough")
        return 2
    finally:
        try:
            veh.destroy()
        except Exception:
            pass
        s = world.get_settings()
        s.synchronous_mode = False
        s.fixed_delta_seconds = None
        world.apply_settings(s)


if __name__ == "__main__":
    sys.exit(main())

"""One-off CARLA smoke test: connect, spawn ego, read one camera."""

import sys
import time
import numpy as np

import carla

client = carla.Client("localhost", 2000)
client.set_timeout(300.0)
print("server version:", client.get_server_version())
world = client.get_world()  # default startup map (no load_world yet)
print("map:", world.get_map().name)
bp_lib = world.get_blueprint_library()
spawn = sorted(world.get_map().get_spawn_points(),
               key=lambda t: (t.location.x, t.location.y))[0]
ego = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), spawn)
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "704")
cam_bp.set_attribute("image_size_y", "256")
cam_bp.set_attribute("fov", "90")
cam = world.spawn_actor(cam_bp, carla.Transform(), attach_to=ego)
frames = []
cam.listen(frames.append)
deadline = time.time() + 30.0
while not frames and time.time() < deadline:
    time.sleep(0.5)
assert frames, "no camera frame within 30 s"
img = frames[0]
arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(
    img.height, img.width, 4)[:, :, :3]
print(f"camera frame: {arr.shape} dtype={arr.dtype}")
print(f"ego at: {ego.get_location()}")
cam.destroy()
ego.destroy()
print("SMOKE OK")

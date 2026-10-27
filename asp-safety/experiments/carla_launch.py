"""CARLA server launcher (P25-A).

Launch / reuse / kill the CARLA 0.9.15 server.

Usage:
    python experiments/carla_launch.py                 # reuse-or-launch, startup map
    python experiments/carla_launch.py --map Town10HD_Opt
    python experiments/carla_launch.py --kill          # kill stale servers, exit
    python experiments/carla_launch.py --windowed --quality Epic

Quality: 0.9.15 has exactly two tiers, Low (default here; shadows and
postprocess off, 50 m draw distance) and Epic (full detail). When the
server already answers on :2000 the flags are moot (reuse path).
"""

import argparse
import socket
import subprocess
import sys
import time

EXE = r"C:\CARLA\WindowsNoEditor\CarlaUE4.exe"
WORK = r"C:\CARLA\WindowsNoEditor"
SERVER_PORT = 2000


def port_open(port: int = SERVER_PORT, timeout: float = 2.0) -> bool:
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect(("localhost", port))
        s.close()
        return True
    except OSError:
        return False


def kill_stale_servers() -> bool:
    """Taskkill every CarlaUE4 server process; call before a fresh run."""
    killed = False
    for image in ("CarlaUE4.exe", "CarlaUE4-Win64-Shipping.exe"):
        r = subprocess.run(
            ["taskkill", "/F", "/IM", image],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            killed = True
    if killed:
        print("killed stale CarlaUE4 server process(es)", flush=True)
        time.sleep(5)
    else:
        print("no stale CarlaUE4 server processes to kill", flush=True)
    while port_open():
        time.sleep(3)
        print("waiting for :2000 to drain...", flush=True)
    return killed


def launch_server(map_name: str | None = None,
                  poll_s: int = 10, max_polls: int = 60,
                  quality: str = "Low", windowed: bool = False,
                  stream_pool: int | None = None) -> bool:
    """Start the server unless one already answers on :2000.

    map_name: bare asset name (Town10HD_Opt) or full /Game/... path.
    Default None keeps the startup map.

    quality: "Low" (default) or "Epic" -- the only two 0.9.15 tiers.
    windowed: render to a visible window (default offscreen saves VRAM
    and is what every harness run uses). Resolution 1280x720.
    stream_pool: trim UE4 r.Streaming.PoolSize (MB) via -ExecCmds
    (fallback knob when Epic windowed approaches the VRAM gate).
    """
    if port_open():
        print("server already responding on :2000; reusing", flush=True)
        return True
    args = [EXE, f"-quality-level={quality}"]
    if stream_pool is not None:
        args.append(f'-ExecCmds="r.Streaming.PoolSize {stream_pool}"')
    if windowed:
        args += ["-windowed", "-ResX=1280", "-ResY=720"]
    else:
        args.append("-RenderOffScreen")
    if map_name:
        if not map_name.startswith("/Game/"):
            map_name = f"/Game/Carla/Maps/{map_name}"
        args.insert(1, map_name)
    print(f"launching {' '.join(args)} ...", flush=True)
    subprocess.Popen(args, cwd=WORK)
    for i in range(max_polls):
        if port_open():
            print(f"port open after {i * poll_s}s", flush=True)
            return True
        time.sleep(poll_s)
    print("port never opened", flush=True)
    return False


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default=None,
                   help="map to start the server with")
    p.add_argument("--kill", action="store_true",
                   help="kill stale CarlaUE4.exe processes and exit")
    p.add_argument("--quality", default="Low", choices=["Low", "Epic"],
                   help="render tier; 0.9.15 offers Low and Epic only")
    p.add_argument("--windowed", action="store_true",
                   help="render to a visible window instead of off-screen")
    p.add_argument("--stream-pool", type=int, default=None,
                   help="UE4 r.Streaming.PoolSize in MB (VRAM trim knob)")
    a = p.parse_args()
    if a.kill:
        sys.exit(0 if kill_stale_servers() else 1)
    sys.exit(0 if launch_server(a.map, quality=a.quality,
                                windowed=a.windowed,
                                stream_pool=a.stream_pool) else 1)

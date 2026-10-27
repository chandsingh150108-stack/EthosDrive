"""CARLA closed-loop bridge (P25-B/C). Shield OFF default; engines
off | heuristic | qp dispatched per replan (P25-C).

One planner (CUDA), one ASP checker, one corrector dispatch table
share the run; NO second model load (6 GB VRAM budget, nextstep.md
5.1). The bridge is the pipeline's closed-loop analogue, not a second
pipeline (no SafetyLayerPipeline instantiation).

Cadence rule (nextstep.md 1): replan every `replan_ticks` ticks
(default 2 s sim); the controller END-ANCHORS the executed plan:
steer alpha to the plan's last waypoint, target speed from the plan's
end-segment speed; intermediate waypoints are lookahead references
only. The 0.5 s native grid is therefore never played back too fast
(documented divergence from naive waypoint queuing).

Warm semantics (nextstep.md 5.3): begin_sequence() once per run;
predict_frame() on replans 2 s sim apart (coarser than nuScenes'
0.5 s cadence; real timestamps carried by ego_status/img_metas).

Steering sign convention (calibrated via experiments/carla_sign_
test.py, P25-B evidence: steer=+0.8 rotates the CARLA-world heading
+16.6 deg = planner-ego -16.6 deg, i.e. CARLA steer>0 turns
planner-RIGHT). The bridge computes alpha in planner-ego frame
(y left-positive), so the command needs cfg.steer_sign = -1.0
(default, frozen from the sign test); overridable for experimentation.

RunLog rows per replan: engine, is_safe_raw, violations (compact),
checker_ms, corrector_ms, plan_ms, fix_flags, corrections_applied_
count, min_ttc_s, agents_count, map_iids, executed waypoints, "
v_target"". Per-tick series: ego world pose/speed; arc length
accumulates from ego positions (sim_km).

CLI: use experiments/run_closed_loop.py (one run -> one JSON row).
"""

import math
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

import carla  # noqa: E402
import carla_launch  # noqa: E402
from carla_probe import (  # noqa: E402
    CAM_CONFIGS, IMG_H, IMG_W, FLIP, build_extrinsic,
)
from carla_context import (  # noqa: E402
    build_agent_context, build_light_context, build_map_context,
    build_zone_context, transform_matrix, to_planner_ego_matrix,
)
from data_loader import estimate_ego_status  # noqa: E402
from neural_planner import SparseDrivePlanner  # noqa: E402
from checker import TrajectoryChecker  # noqa: E402
from corrector import TrajectoryCorrector, CorrectorQP  # noqa: E402
import correction_strategies as _cs  # noqa: E402

# Longitudinal findings are rear-end gaps: the effective remedy is
# slowing down (follow family), not a lateral tail shift. CARLA cut-in
# runs showed the lateral remedy ping-pongs forever -> give-ups. Move
# longitudinal_clearance to the follow family (experiments-side remap;
# frozen src/ untouched).
_cs.COLLISION_RULES = _cs.COLLISION_RULES - {'longitudinal_clearance'}
_cs.FOLLOW_RULES = _cs.FOLLOW_RULES | {'longitudinal_clearance'}
_cs.RULE_GROUPS = dict(_cs.RULE_GROUPS)
_cs.RULE_GROUPS['collision'] = ('_correct_collision', _cs.COLLISION_RULES)
_cs.RULE_GROUPS['follow'] = ('_correct_safe_following', _cs.FOLLOW_RULES)

FIXED_DT = 0.05
CAM_ORDER = ["FRONT", "FRONT_LEFT", "FRONT_RIGHT",
             "BACK", "BACK_LEFT", "BACK_RIGHT"]

CKPT = ROOT / "SparseDrive/ckpt/sparsedrive_stage2.pth"
RULES = ROOT / "rules"


@dataclass
class BridgeConfig:
    scenario: str = "S1"
    seed: int = 1
    engine: str = "off"          # off | heuristic | qp
    plan_engine: str = "neural"  # neural | lane (analytic reference)
    replan_ticks: int = 40       # 40 * 0.05 s = 2 s sim (plan 1)
    max_minutes: float = 6.0     # wall cap
    max_distance_m: float = 600.0
    device: str = "cuda"
    keep_server: bool = True
    # controller constants (nextstep.md 4.3; tuned once on S1)
    wheelbase_m: float = 2.85
    lookahead_factor: float = 0.6
    lookahead_min_m: float = 2.0
    lookahead_max_m: float = 6.0
    v_target_cap_ms: float = 13.9
    steer_gain: float = 2.0      # steer = steer_gain * wheelbase * sin(a)/L
    steer_deadband: float = 0.02
    throttle_gain: float = 1.2
    throttle_smooth: float = 0.2
    steer_sign: float = -1.0     # calibrated: steer sign test P25-B
    dbg_lines: bool = False      # [ctrl] diagnostics every ~2 s sim
    view_ego: bool = False       # spectator ego-follow (windowed server)


class CarlaBridge:
    """Sync-mode loop: CARLA -> planner -> [shield] -> controller."""

    def __init__(self, cfg: BridgeConfig):
        self.cfg = cfg
        self.client = None
        self.world = None
        self.ego = None
        self.cams: List = []
        self.collision_sensor = None
        self.col_queue: deque = deque()
        self.col_seen: set = set()
        self.queues: Dict[str, deque] = {}
        self.planner = None
        self.checker = None
        self.correctors: Dict[str, object] = {}
        self.scenario_obj = None
        self.K = None
        self.C = None
        self.stacks: Optional[np.ndarray] = None
        self.last_plan_ego: Optional[np.ndarray] = None
        self.last_plan_world: Optional[List[Tuple[float, float]]] = None
        self.prev_ego_global: Optional[np.ndarray] = None
        self.last_v_target = 0.0
        self.last_throttle = 0.0
        self.prev_world_pos: Optional[np.ndarray] = None
        self.last_progress_pos: Optional[Tuple[float, float]] = None
        self.stuck_tick: Optional[int] = None
        self.arc_len_m = 0.0
        self.sim_time_s = 0.0
        self.rows: List[dict] = []
        self.tick_series: List[dict] = []
        self.replans = 0
        self.frame_drops = 0
        self.give_ups = 0
        self.collision_events: List[dict] = []
        self.wall0 = 0.0
        self.dbg_count = 0
        self.hold_ticks = 0
        self.spectator = None
        self._final_summary: Optional[dict] = None

    # ------------------------------------------------------------------
    # setup / teardown

    def setup(self):
        if not carla_launch.launch_server(None):
            raise RuntimeError("CARLA server unavailable")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(300.0)
        self.world = self.client.get_world()
        print(f"[bridge] map: {self.world.get_map().name}")

        s = self.world.get_settings()
        s.synchronous_mode = True
        s.fixed_delta_seconds = FIXED_DT
        self.world.apply_settings(s)

        # Defensive sweep: crashed/wall-capped runs and ad-hoc probes
        # can leak NPC vehicles, walkers, cameras and collision
        # sensors; any leftover sits on the shared spawn routes and
        # poisons the next run (stalled/collided). Destroy everything
        # except lights and the spectator before spawning the ego.
        for a in self.world.get_actors():
            if a.type_id.startswith(("traffic.", "spectator")):
                continue
            if a.id < 1000:  # map-built props/signs keep their ids
                continue
            try:
                a.destroy()
            except Exception:
                pass

        bp_lib = self.world.get_blueprint_library()
        from carla_scenarios import get_scenario
        scenario = get_scenario(self.cfg.scenario)
        spawn, actor_hook = scenario.resolve_setup(self.world, self.cfg.seed)
        self.scenario_obj = scenario
        ego_bp = bp_lib.find("vehicle.tesla.model3")
        self.ego = self.world.try_spawn_actor(ego_bp, spawn)
        if self.ego is None:
            raise RuntimeError(f"ego spawn failed at {spawn}")
        print(f"[bridge] ego at {spawn.location.x:.1f}, "
              f"{spawn.location.y:.1f}")

        self.cams = []
        for name, yaw, fov, dx, dy, dz in CAM_CONFIGS:
            bp = bp_lib.find("sensor.camera.rgb")
            bp.set_attribute("image_size_x", str(IMG_W))
            bp.set_attribute("image_size_y", str(IMG_H))
            bp.set_attribute("fov", str(fov))
            tf = carla.Transform(
                carla.Location(x=dx, y=dy, z=dz),
                carla.Rotation(pitch=-8.0 if yaw == 0.0 else 0.0, yaw=yaw))
            cam = self.world.spawn_actor(bp, tf, attach_to=self.ego)
            self.cams.append((name, cam))
        self.queues = {name: deque(maxlen=4) for name, _ in self.cams}
        for name, cam in self.cams:
            cam.listen(self.queues[name].append)

        c_bp = bp_lib.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(
            c_bp, carla.Transform(), attach_to=self.ego)
        self.collision_sensor.listen(self.col_queue.append)

        # calibration is attachment-relative: pose-independent
        self.world.tick()
        t_world_veh = transform_matrix(self.ego.get_transform())
        self.K = np.zeros((6, 3, 3), dtype=np.float32)
        self.C = np.zeros((6, 4, 4), dtype=np.float32)
        for idx, (name, cam) in enumerate(self.cams):
            fov = next(f for n, _, f, _, _, _ in CAM_CONFIGS if n == name)
            fx = (IMG_W / 2) / math.tan(math.radians(fov) / 2)
            cx, cy = IMG_W / 2.0, IMG_H / 2.0
            self.K[idx] = [[fx, 0, cx], [0, fx, cy], [0, 0, 1]]
            self.C[idx] = build_extrinsic(
                t_world_veh, transform_matrix(cam.get_transform()))
        names = [n for n, _ in self.cams]
        assert names == CAM_ORDER, f"cam order {names}"

        print(f"[bridge] loading planner ({self.cfg.device})...")
        self.planner = SparseDrivePlanner(str(CKPT), device=self.cfg.device)
        self.checker = TrajectoryChecker(str(RULES))
        self.correctors = {
            "heuristic": TrajectoryCorrector(
                self.checker,
                params={"lateral_shift_m": -2.0},
                on_failure="brake"),
            "qp": CorrectorQP(self.checker, on_failure="brake"),
        }
        self.planner.begin_sequence()
        actor_hook(self.world, self.ego, self)
        self.world.tick()
        loc = self.ego.get_location()
        self.prev_world_pos = np.array([loc.x, loc.y])
        self.wall0 = time.time()
        if self.cfg.view_ego:
            self.spectator = self.world.get_spectator()
            self._view_apply()
            print("[bridge] view: ego-follow enabled")
        print("[bridge] setup complete")

    def teardown(self):
        if getattr(self, "scenario_obj", None) is not None:
            try:
                self.scenario_obj.cleanup(self.world)
            except Exception:
                pass
            self.scenario_obj = None
        try:
            if self.planner is not None:
                self.planner.end_sequence()
        except Exception:
            pass
        self.planner = None
        self.correctors = {}
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        for _, cam in reversed(self.cams):
            try:
                cam.stop()
                cam.destroy()
            except Exception:
                pass
        self.cams = []
        if self.collision_sensor is not None:
            try:
                self.collision_sensor.destroy()
            except Exception:
                pass
            self.collision_sensor = None
        if self.ego is not None:
            try:
                self.ego.destroy()
            except Exception:
                pass
            self.ego = None
        if self.world is not None:
            s = self.world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            self.world.apply_settings(s)
        print("[bridge] teardown complete (server kept alive)")

    # ------------------------------------------------------------------
    # main loop

    def run(self) -> dict:
        if self.ego is None:
            self.setup()
        status = "ok"
        try:
            status = self._main_loop()
        finally:
            self.teardown()
        return self._summary(status)

    def _main_loop(self) -> str:
        tick_i = 0
        while True:
            self.world.tick()
            tick_i += 1
            self.sim_time_s = tick_i * FIXED_DT

            while self.col_queue:
                ev = self.col_queue.popleft()
                other = getattr(ev, "other_actor", None)
                other_id = getattr(other, "id", None)
                other_type = getattr(other, "type_id", str(other))
                if other_id is not None and other_id in self.col_seen:
                    continue
                if other_id is not None:
                    self.col_seen.add(other_id)
                i_vec = getattr(ev, "normal_impulse", None)
                loc = self.ego.get_location() if self.ego else None
                self.collision_events.append({
                    "tick": tick_i,
                    "sim_s": round(self.sim_time_s, 2),
                    "other": other_type,
                    "other_id": other_id,
                    "ego_xyz": (None if loc is None
                                else (round(loc.x, 1), round(loc.y, 1),
                                      round(loc.z, 1))),
                    "impulse": float(np.linalg.norm(
                        np.array([getattr(i_vec, "x", 0.0),
                                  getattr(i_vec, "y", 0.0),
                                  getattr(i_vec, "z", 0.0)]))),
                })
                print(f"[bridge] COLLISION sim_s={self.sim_time_s:.2f} "
                      f"with {other_type}", flush=True)

            if self.scenario_obj is not None:
                self.scenario_obj.tick(self.world, self.sim_time_s)

            if self.cfg.view_ego:
                self._view_apply()

            if tick_i % self.cfg.replan_ticks == 0:
                self._replan(tick_i)

            self._control_tick()
            wall_min = (time.time() - self.wall0) / 60.0
            if self.arc_len_m >= self.cfg.max_distance_m:
                print(f"[bridge] distance target reached "
                      f"({self.arc_len_m:.0f} m)", flush=True)
                return "distance_reached"
            if wall_min >= self.cfg.max_minutes:
                print(f"[bridge] wall cap reached ({wall_min:.1f} min)",
                      flush=True)
                return "wall_cap"
            # stuck-break: no meaningful progress while throttle is up.
            # Skips windows that include planned holds (v <= 1 m/s, e.g.
            # red-light / jaywalker waits): the zero progress there is
            # intended, and the resume boundary would otherwise false-
            # positive a stall.
            if tick_i % 100 == 0 and self.hold_ticks == 0:
                loc = self.ego.get_location() if self.ego else None
                cur = (loc.x, loc.y) if loc else None
                if cur is None:
                    continue
                if self.last_progress_pos is not None:
                    d = float(np.hypot(
                        cur[0] - self.last_progress_pos[0],
                        cur[1] - self.last_progress_pos[1]))
                    if d < 0.5 and self.last_v_target > 1.0:
                        print(f"[bridge] stalled: {d:.2f} m progress "
                              f"in 5 s sim (target "
                              f"{self.last_v_target:.1f} m/s)",
                              flush=True)
                        return "stalled"
                self.last_progress_pos = cur

    # ------------------------------------------------------------------
    # spectator ego-follow (visible server)

    def _view_apply(self):
        """Point the spectator 12 m behind the ego at 7 m height.

        Applied per tick; the pose change takes effect on the next
        render tick (one 50 ms sim step behind -- imperceptible).
        No physics effect: pure rendering convenience.
        """
        if self.spectator is None or self.ego is None:
            return
        t = self.ego.get_transform()
        yaw = math.radians(t.rotation.yaw)
        loc = t.location
        self.spectator.set_transform(carla.Transform(
            carla.Location(
                x=loc.x - 12.0 * math.cos(yaw),
                y=loc.y - 12.0 * math.sin(yaw),
                z=loc.z + 7.0),
            carla.Rotation(pitch=-25.0, roll=0.0, yaw=t.rotation.yaw)))

    # ------------------------------------------------------------------
    # replan

    def _replan(self, tick_i):
        frames = self._sync_frames()
        if frames is None:
            self.frame_drops += 1
            print(f"[bridge] frame drop #{self.frame_drops} "
                  f"(previous plan kept)", flush=True)
            return
        t_start = time.perf_counter()

        ego_tf = self.ego.get_transform()
        ego_g = to_planner_ego_matrix(transform_matrix(ego_tf))
        if self.prev_ego_global is None:
            status = np.zeros(18, dtype=np.float32)
        else:
            dt_s = self.cfg.replan_ticks * FIXED_DT
            status = estimate_ego_status(self.prev_ego_global, ego_g, dt_s)
        self.prev_ego_global = ego_g
        ts = time.time()

        if self.cfg.plan_engine == "lane":
            plan_1s = self._lane_plan(ego_tf, ego_g)
            plan_ms = 0.0
            plan_native = None
        else:
            t0 = time.perf_counter()
            out = self.planner.predict_frame(
                self.stacks, self.K, self.C, ego_g, ts, ego_status=status)
            plan_ms = (time.perf_counter() - t0) * 1000.0
            plan_native = out["trajectory"]
            # Emission axis order (verified vs nuScenes gt + CARLA drive,
            # see experiments/plan_axis_check.py): plan columns are
            # (lateral, forward) at 0.5 s cadence. Swap to planner-ego
            # (x forward, y left) before any downstream consumption.
            plan_1s = np.vstack([
                np.zeros((1, 2), dtype=np.float32),
                plan_native[1::2][:3, ::-1],
            ])

        mapd = build_map_context(self.world.get_map(), ego_tf.location,
                                 ego_transform=ego_tf)
        inter, lights = build_light_context(
            self.world, ego_tf.location, ego_transform=ego_tf)
        if inter:
            merged = dict(mapd.get("intersections", {}))
            merged.update(inter)
            mapd["intersections"] = merged
        agents = build_agent_context(self.world, self.ego, plan_1s)
        zones = build_zone_context(self.cfg.scenario)
        ctx = dict(
            other_vehicles=agents or None,
            map_data=mapd or None,
            traffic_lights=lights or None,
            zones=zones,
        )

        t0 = time.perf_counter()
        try:
            is_safe_raw, violations = self.checker.check(plan_1s, **ctx)
        except Exception as e:
            is_safe_raw, violations = None, [{
                "rule": "check_error", "vehicle": "ego", "time": -1,
                "error": str(e)[:120],
            }]
        checker_ms = (time.perf_counter() - t0) * 1000.0

        fix_flags: List[str] = []
        corrections_count = 0
        corrector_ms = 0.0
        executed = plan_1s
        if self.cfg.engine != "off":
            if is_safe_raw:
                fix_flags.append("already_safe_corrector_skipped")
            else:
                t0 = time.perf_counter()
                try:
                    executed, applied = self.correctors.get(
                        self.cfg.engine,
                        TrajectoryCorrector(self.checker)).correct(
                            plan_1s, **ctx)
                    corrections_count = len(applied)
                except Exception as e:
                    fix_flags.append("revert_to_plan")
                    self.give_ups += 1
                    print(f"[bridge] corrector give-up: "
                          f"{e}", flush=True)
                corrector_ms = (time.perf_counter() - t0) * 1000.0

        min_ttc = self._min_ttc(plan_1s, agents)

        t_wv = transform_matrix(ego_tf)
        M = t_wv @ FLIP
        exec_arr = np.asarray(executed, dtype=float)
        world_pts = []
        for (x, y) in exec_arr:
            w = M @ np.array([x, y, 0.0, 1.0])
            world_pts.append((float(w[0]), float(w[1])))
        # up-sample the executed path at ~1 m resolution so the
        # controller's lookahead projection can track curves instead
        # of cutting from the plan end anchor.
        path = []
        for i, pt in enumerate(world_pts):
            if i == 0:
                path.append(pt)
                continue
            a = np.array(world_pts[i - 1])
            b = np.array(pt)
            seg = float(np.linalg.norm(b - a))
            n = max(1, int(seg / 1.0))
            for k in range(1, n + 1):
                path.append(tuple(a + (b - a) * (k / n)))
        self.last_plan_ego = exec_arr.astype(np.float32)
        self.last_plan_world = path

        if len(exec_arr) >= 2:
            self.last_v_target = float(np.linalg.norm(
                exec_arr[-1] - exec_arr[-2]))
        safe_delta = None
        if self.cfg.engine != "off" and "revert_to_plan" not in fix_flags:
            safe_delta = float(np.sqrt(np.mean(
                np.sum((exec_arr - plan_1s) ** 2, axis=1))))

        self.replans += 1
        self.rows.append({
            "replan": self.replans,
            "tick": tick_i,
            "sim_s": round(self.sim_time_s, 3),
            "engine": self.cfg.engine,
            "is_safe_raw": is_safe_raw,
            "violations": [
                {k: v.get(k)
                 for k in ("rule", "vehicle", "vehicle2", "time")}
                if isinstance(v, dict) else {"raw": str(v)}
                for v in violations
            ],
            "checker_ms": round(checker_ms, 2),
            "corrector_ms": round(corrector_ms, 2),
            "plan_ms": round(plan_ms, 2),
            "exec_ms": round((time.perf_counter() - t_start) * 1000.0, 2),
            "fix_flags": fix_flags,
            "corrections_applied_count": corrections_count,
            "min_ttc_s": min_ttc,
            "agents_count": len(agents),
            "map_iids": list(mapd.get("lanes", {})) + list(lights or {}),
            "plan_ego": exec_arr.round(2).tolist(),
            "executed_ego": (
                np.asarray(executed, dtype=float).round(2).tolist()),
            "v_target_ms": round(self.last_v_target, 3),
            "safe_vs_plan_l2": (
                None if safe_delta is None else round(safe_delta, 3)),
        })

    def _lane_plan(
        self, ego_tf, ego_g: np.ndarray,
    ) -> np.ndarray:
        """Analytic plan: current-lane centerline at a fixed v_ref.

        Fallback / reference-plan engine (plan doc fallback arm):
        walks the CARLA waypoint chain from the ego's road waypoint,
        samples the centerline at 1/2/3 s * v_ref along-path, and
        converts each world point into the planner-ego frame via
        inv(ego_g) so downstream ASP/consumption semantics stay
        identical to the neural plan path.
        """
        v_ref = float(min(7.0, self.cfg.v_target_cap_ms))
        mp = self.world.get_map()
        p0 = ego_tf.location
        wpt = mp.get_waypoint(
            carla.Location(x=p0.x, y=p0.y, z=p0.z + 0.3))
        chain = []
        if wpt is not None:
            acc = 0.0
            prev = (p0.x, p0.y)
            limit = v_ref * 3.0 + 2.0
            while acc < limit:
                nxt = wpt.next(0.5)
                if not nxt:
                    break
                wpt = nxt[0]
                wl = wpt.transform.location
                acc += float(np.hypot(wl.x - prev[0], wl.y - prev[1]))
                prev = (float(wl.x), float(wl.y))
                junction_speed = 5.0 if wpt.is_junction else v_ref
                chain.append((prev[0], prev[1], acc, junction_speed))

        def speed_at(d: float) -> float:
            return v_ref if not chain or d >= chain[-1][2] else (
                chain[0][3] if d <= chain[0][2] else next(
                    s for (x, y, a, s) in chain if a >= d))

        targets = []
        d_plan = 0.0
        for _ in range(3):
            d_plan = min(d_plan + speed_at(d_plan),
                         (chain[-1][2] if chain else v_ref * 3.0))
            targets.append(d_plan)
        inv = np.linalg.inv(ego_g)
        rows = []
        for d_target in targets:
            pt = None
            for (x, y, acc_d, _s) in chain:
                if acc_d >= d_target:
                    pt = (x, y)
                    break
            if pt is None and chain:
                pt = chain[-1][:2]
            if pt is None:
                rows.append([0.0, 0.0])
                continue
            w = np.array([pt[0], pt[1], p0.z, 1.0])
            e = inv @ w
            rows.append([float(e[0]), float(e[1])])
        return np.vstack([
            np.zeros((1, 2), dtype=np.float32),
            np.asarray(rows, dtype=np.float32),
        ])

    def _min_ttc(self, plan: np.ndarray,
                 agents: List[np.ndarray]) -> Optional[float]:
        """nextstep.md 6: min over agents, t in {1..3}, of
        dist(agent_t, plan_t) / closing_speed (m per ASP-step = s)."""
        ttc = []
        for agent in agents:
            n = min(len(plan), len(agent))
            for t in range(1, n):
                d_now = float(np.hypot(*(agent[t] - plan[t])))
                d_prev = float(np.hypot(*(agent[t - 1] - plan[t - 1])))
                closing = (d_prev - d_now) / 1.0
                if closing > 0.1:
                    ttc.append(d_now / closing)
        return min(ttc) if ttc else None

    def _sync_frames(self) -> Optional[np.ndarray]:
        deadline = time.time() + 0.5
        while time.time() < deadline:
            if all(self.queues[n] for n, _ in self.cams):
                f = min(self.queues[n][-1].frame for n, _ in self.cams)
                imgs = []
                for name, _ in self.cams:
                    got = [im for im in self.queues[name] if im.frame == f]
                    if got:
                        imgs.append(got[-1])
                if len(imgs) == len(self.cams):
                    stack = np.zeros((6, 3, IMG_H, IMG_W), dtype=np.float32)
                    for idx, im in enumerate(imgs):
                        arr = np.frombuffer(im.raw_data, np.uint8).reshape(
                            IMG_H, IMG_W, 4)
                        stack[idx] = np.ascontiguousarray(
                            arr[:, :, 2::-1].transpose(2, 0, 1)).astype(
                                np.float32)
                    self.stacks = stack
                    return stack
            time.sleep(0.02)
        return None

    # ------------------------------------------------------------------
    # controller (plan 4.3; consumption cadence note 1)

    def _control_tick(self):
        if self.last_plan_world is None or self.ego is None:
            return
        loc = self.ego.get_location()
        tr = self.ego.get_transform()
        fwd = tr.get_forward_vector()
        v = self.ego.get_velocity()
        v_now = float(np.hypot(v.x, v.y))
        pos = np.array([loc.x, loc.y])
        lookahead = float(np.clip(
            self.cfg.lookahead_factor * max(v_now, 1.0),
            self.cfg.lookahead_min_m, self.cfg.lookahead_max_m))

        path = np.asarray(self.last_plan_world, dtype=float)
        # Lookahead target: point on the plan path at arc-length
        # (lookahead) ahead of the NEAREST path point to the current
        # ego position. Falls back to the path end.
        diffs = np.linalg.norm(path - pos[None, :], axis=1)
        i_near = int(np.argmin(diffs))
        arc = np.concatenate([[0.0], np.cumsum(
            np.linalg.norm(np.diff(path, axis=0), axis=1))])
        s0 = float(arc[i_near])
        s_target = s0 + lookahead
        j = int(np.searchsorted(arc, s_target))
        target = path[min(j, len(path) - 1)]
        end_arc = float(arc[-1])
        gap = s_target - end_arc
        if j >= len(path) and len(path) >= 2 and gap > 0:
            d0 = path[-1] - path[-2]
            d0 = d0 / max(np.linalg.norm(d0), 1e-6)
            target = path[-1] + d0 * gap

        # planner-ego convention: y_left = -y_carla; alpha signed so that
        # alpha > 0 means the target lies to the planner-LEFT of heading.
        h = np.array([fwd.x, -fwd.y])
        d = target - pos
        d = np.array([d[0], -d[1]])
        norm_h = np.hypot(*h)
        norm_d = np.hypot(*d)
        if norm_h < 1e-6 or norm_d < 1e-6:
            alpha = 0.0
        else:
            cosang = float(np.clip(
                np.dot(h, d) / (norm_h * norm_d), -1.0, 1.0))
            cross = h[0] * d[1] - h[1] * d[0]
            alpha = math.acos(cosang)
            if cross < 0:
                alpha = -alpha

        steer_norm = float(np.clip(
            self.cfg.steer_gain * self.cfg.wheelbase_m * math.sin(alpha)
            / max(lookahead, 0.5), -1.0, 1.0))
        if abs(steer_norm) < self.cfg.steer_deadband:
            steer_norm = 0.0
        steer = float(np.clip(self.cfg.steer_sign * steer_norm, -1.0, 1.0))

        v_target = float(np.clip(self.last_v_target, 0.0,
                                 self.cfg.v_target_cap_ms))
        throttle = 0.0
        brake = 0.0
        if v_target >= 0.2:
            raw = float(np.clip(
                self.cfg.throttle_gain * (v_target - v_now)
                / max(v_target, 4.0), 0.0, 1.0))
            self.last_throttle = (
                self.cfg.throttle_smooth * raw
                + (1.0 - self.cfg.throttle_smooth) * self.last_throttle)
            throttle = self.last_throttle
        else:
            brake = 1.0

        self.ego.apply_control(carla.VehicleControl(
            steer=steer, throttle=throttle, brake=brake, reverse=False))
        if v_target <= 1.0:
            self.hold_ticks += 1
        else:
            self.hold_ticks = 0

        if self.cfg.dbg_lines and self.dbg_count % 40 == 0:
            print(f"[ctrl] tick={self.sim_time_s:.0f} pos=({loc.x:.1f},"
                  f"{loc.y:.1f}) alpha={math.degrees(alpha):+.0f}deg "
                  f"steer={steer:+.2f} thr={throttle:.2f} "
                  f"v={v_now:.2f}/{v_target:.1f} "
                  f"tgt=({target[0]:.1f},{target[1]:.1f})", flush=True)
        self.dbg_count += 1

        if self.prev_world_pos is not None:
            self.arc_len_m += float(np.hypot(pos[0] - self.prev_world_pos[0],
                                             pos[1] - self.prev_world_pos[1]))
        self.prev_world_pos = pos
        self.tick_series.append({
            "tick": int(round(self.sim_time_s / FIXED_DT)),
            "x": round(float(loc.x), 2), "y": round(float(loc.y), 2),
            "v": round(v_now, 2),
        })

    # ------------------------------------------------------------------
    # summary

    def _summary(self, status: str) -> dict:
        wall = time.time() - self.wall0
        give_rate = self.give_ups / self.replans if self.replans else 0.0
        return {
            "status": status,
            "scenario": self.cfg.scenario,
            "seed": self.cfg.seed,
            "engine": self.cfg.engine,
            "device": self.cfg.device,
            "map": (self.world.get_map().name
                    if self.world is not None else "n/a"),
            "replans": self.replans,
            "frame_drops": self.frame_drops,
            "sim_time_s": round(self.sim_time_s, 1),
            "wall_min": round(wall / 60.0, 2),
            "sim_km": round(self.arc_len_m / 1000.0, 3),
            "collision_events": self.collision_events,
            "collided": len(self.collision_events) > 0,
            "give_ups": self.give_ups,
            "give_up_rate": round(give_rate, 3),
            "rows": self.rows,
            "tick_series": self.tick_series,
            "config": asdict(self.cfg),
        }

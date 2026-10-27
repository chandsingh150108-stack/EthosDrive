"""Scripted closed-loop scenarios (P25-D: S1-S5).

Each scenario exposes:
    resolve_setup(world, seed) -> (carla.Transform spawn, actor_hook)
where actor_hook(world, ego, bridge) spawns/records the scripted
actors right after the ego and rig exist but before the first
physics tick of the run loop.
    tick(world, sim_time_s)    -> per-tick deterministic choreography
    cleanup(world)             -> despawn scripted actors / unfreeze

Choreography arm: scripted movers are kinematically teleported
(set_simulate_physics(False) + set_transform each fixed-dt tick), so
runs are deterministic end-to-end. Equivalent to the "ORACLE-CONTEXT"
flagged arm from nextstep.md 7.4: the ASP shield sees ground-truth
actor poses; planner camera detections are NOT the source for
bystanders' motion (logged in docs/carla_harness.md Deviations).

Scenarios (frozen in nextstep.md 7.1):
    S1 free drive         | no ambient traffic      | baseline
    S2 red-light run      | controlling light actors frozen RED;
                          | ego spawn selected so the stop line is
                          | 55-95 m ahead
    S3 cut-in             | NPC 2 lanes left, ~25-40 m ahead, same
                          | direction slower; cuts into ego's lane
                          | starting at t=3 s
    S4 jaywalker          | walker crosses ego's path outside a
                          | marked crossing, 15 m ahead, 1.2 m/s
    S5 oncoming left-turn | NPC starts in opposing lane, turns left
                          | across ego's path starting t=2.5 s

Coordinates are resolved ONCE per run from real carla.Map waypoint
queries (resolve_setup) so scripted geometry is always road-
consistent; seeds inject deterministic spawn selection.
"""

import math
import random
from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# generic helpers


def _chain(wp, total_m: float, step: float = 1.0) -> np.ndarray:
    """Forward waypoint chain [(x, y, z, yaw_deg, arc_m)].

    Row index i corresponds to arc i*step from the starting waypoint;
    walking stops where wp.next() is exhausted (dead ends).
    """
    pts = [(wp.transform.location.x, wp.transform.location.y,
            wp.transform.location.z, wp.transform.rotation.yaw, 0.0)]
    acc = 0.0
    prev = pts[0]
    while acc < total_m:
        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]
        wl = wp.transform.location
        acc += math.hypot(wl.x - prev[0], wl.y - prev[1])
        prev = (wl.x, wl.y, wl.z, wp.transform.rotation.yaw, acc)
        pts.append(prev)
    return np.asarray(pts, dtype=float)


def _at(chain: np.ndarray, i: int) -> np.ndarray:
    """Chain row with clamped index."""
    return chain[int(np.clip(i, 0, len(chain) - 1))]


def _yaw_deg(a, b) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def _carla_T(x: float, y: float, z: float, yaw: float):
    import carla
    return carla.Transform(carla.Location(x=x, y=y, z=z),
                           carla.Rotation(yaw=yaw))


def _append_curve(pts: List[Tuple], b0: Tuple, b1: Tuple, b2: Tuple,
                  n: int = 14) -> None:
    """Append a quadratic-bezier arc (b0 -> b1 -> b2); yaw per segment."""
    for t in np.linspace(0.0, 1.0, n + 1)[1:]:
        q = (1 - t) ** 2 * np.array(b0[:2]) + \
            2 * (1 - t) * t * np.array(b1[:2]) + \
            t * t * np.array(b2[:2])
        pts.append((float(q[0]), float(q[1]), b2[2], 0.0))
    for i in range(1, len(pts)):
        pts[i - 1] = tuple(list(pts[i - 1][:3]) + [_yaw_deg(
            pts[i - 1], pts[i])])
    pts[-1] = tuple(list(pts[-1][:3]) + [pts[-2][3]])


class _Poly:
    """Arc-length-parameterized polyline [(x,y,z,yaw)]."""

    def __init__(self, points: np.ndarray):
        self.pts = np.asarray(points, dtype=float)
        seg = self.pts[1:, :2] - self.pts[:-1, :2]
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.arc = np.concatenate([[0.0], np.cumsum(self.seg_len)])

    def pose(self, s: float) -> Tuple[float, float, float, float]:
        s = float(np.clip(s, 0.0, self.arc[-1]))
        i = int(np.searchsorted(self.arc, s, "right")) - 1
        i = min(max(i, 0), len(self.pts) - 2)
        a, b = self.pts[i], self.pts[i + 1]
        den = self.seg_len[i] if self.seg_len[i] > 1e-9 else 1.0
        f = (s - self.arc[i]) / den
        xy = a[:2] + (b[:2] - a[:2]) * float(f)
        z = a[2] + (b[2] - a[2]) * float(f)
        return float(xy[0]), float(xy[1]), float(z), float(_yaw_deg(a, b))


class _Kinematic:
    """Teleported actor following a polyline at a constant speed."""

    def __init__(self, actor, poly: _Poly, speed: float):
        self.actor = actor
        self.poly = poly
        self.speed = speed
        try:
            actor.set_simulate_physics(False)
        except Exception:
            pass

    def tick(self, sim_time_s: float) -> None:
        x, y, z, yaw = self.poly.pose(self.speed * sim_time_s)
        self.actor.set_transform(_carla_T(x, y, z, yaw))


def _reset_all_lights(world) -> None:
    """Deterministic reset of every map light to a plain cycle.

    Cross-run state must not leak: pinning red for one scenario would
    otherwise bleed into the next scenario's checks.
    """
    for a in world.get_actors():
        if "traffic_light" not in a.type_id:
            continue
        try:
            a.set_green_time(20.0)
            a.set_yellow_time(3.0)
            a.set_red_time(20.0)
            a.reset_group()
        except Exception:
            pass


class _ScriptedScenario:
    name = "?"
    description = ""

    def __init__(self):
        self.kin: List[_Kinematic] = []
        self.red_lights: List = []
        self._light_snapshot = []

    def resolve_setup(self, world, seed: int) -> Tuple:
        raise NotImplementedError

    def tick(self, world, sim_time_s: float) -> None:
        for k in self.kin:
            k.tick(sim_time_s)

    def cleanup(self, world) -> None:
        for k in self.kin:
            try:
                k.actor.destroy()
            except Exception:
                pass
        self.kin = []
        # restore any lights we pinned red
        for li, st, rt in self._light_snapshot:
            try:
                li.set_red_time(rt)
                li.set_state(st)
            except Exception:
                pass
        try:
            for li, _st, _rt in self._light_snapshot:
                li.reset_group()
        except Exception:
            pass
        self._light_snapshot = []
        self.red_lights = []
        _reset_all_lights(world)

    def _signal_centers(self, world) -> List:
        out = []
        for a in world.get_actors():
            if "traffic_light" not in a.type_id:
                continue
            try:
                tv = a.trigger_volume.location
                out.append((tv.x, tv.y))
            except Exception:
                pass
        return out

    def _route_signal_free(self, chain, centers, radius: float = 45.0):
        """True when no chained waypoint enters a signalized cluster."""
        if not centers:
            return True
        for (cx, cy) in centers:
            d = np.hypot(chain[:, 0] - cx, chain[:, 1] - cy)
            if float(d.min()) <= radius:
                return False
        return True

    def _pick_spawn(self, world, seed: int):
        """Spawn whose 130 m route avoids signalized clusters, so that
        S1/S3/S4/S5 evidence is dominated by the scripted event rather
        than by traffic-light phase timing (S2 pins red deliberately).
        """
        centers = self._signal_centers(world)
        mpc = world.get_map()
        spawns = sorted(mpc.get_spawn_points(),
                        key=lambda t: (t.location.x, t.location.y))
        if not spawns:
            raise RuntimeError("map has no spawn points")
        rng = random.Random(seed)
        off = rng.randrange(len(spawns))
        ordered = spawns[off:] + spawns[:off]
        for sp in ordered:
            chain = _chain(mpc.get_waypoint(sp.location), 130.0)
            if len(chain) > 20 and self._route_signal_free(chain, centers):
                return sp
        raise RuntimeError("no signal-free spawn")

    def _spawn_candidates(self, world, seed: int, limit: int = 8):
        """Seeded-rotation spawn candidates for spawn-failure retries.
        Signalized-cluster routes are skipped so lights never gate the
        scripted choreography."""
        centers = self._signal_centers(world)
        mpc = world.get_map()
        spawns = sorted(mpc.get_spawn_points(),
                        key=lambda t: (t.location.x, t.location.y))
        if not spawns:
            raise RuntimeError("map has no spawn points")
        rng = random.Random(seed)
        off = rng.randrange(len(spawns))
        ordered = spawns[off:] + spawns[:off]
        out = []
        for sp in ordered:
            chain = _chain(mpc.get_waypoint(sp.location), 130.0)
            if len(chain) > 20 and self._route_signal_free(chain, centers):
                out.append(sp)
                if len(out) >= limit:
                    break
        if not out:
            raise RuntimeError("no signal-free spawn candidates")
        return out


# ---------------------------------------------------------------------------
# S1


class S1FreeDrive(_ScriptedScenario):
    name = "S1"
    description = "free drive, no ambient traffic"

    def resolve_setup(self, world, seed):
        return self._pick_spawn(world, seed), (lambda w, ego, bridge: None)


# ---------------------------------------------------------------------------
# S2: red-light run


class S2RedLightRun(_ScriptedScenario):
    name = "S2"
    description = ("signalized junction held RED per tick; ego spawns "
                   "so the stop line is 55-95 m ahead")

    def resolve_setup(self, world, seed):
        import carla
        mp = world.get_map()
        lights = [a for a in world.get_actors()
                  if "traffic_light" in a.type_id]
        if not lights:
            raise RuntimeError("S2: no traffic lights on map")
        # a genuinely signalized approach: stop line near the light's
        # own trigger center (CARLA light actors reference approaches
        # across the map; only physical approaches count here)
        signalized = []
        for li in lights:
            tc = li.trigger_volume.location
            try:
                swps = li.get_stop_waypoints() or []
            except Exception:
                swps = []
            for w in swps:
                loc = (w.transform.location.x, w.transform.location.y,
                       w.transform.location.z)
                if math.hypot(loc[0] - tc.x, loc[1] - tc.y) <= 30.0:
                    signalized.append((li, loc))
        if not signalized:
            raise RuntimeError("S2: no signalized approaches")

        cand = []
        for sp in sorted(mp.get_spawn_points(),
                         key=lambda t: (t.location.x, t.location.y)):
            wp0 = mp.get_waypoint(sp.location)
            chain = _chain(wp0, 130.0)
            if len(chain) < 10:
                continue
            for _, loc in signalized:
                d = np.hypot(chain[:, 0] - loc[0], chain[:, 1] - loc[1])
                i_near = int(np.argmin(d))
                if float(d[i_near]) < 6.0 and 55.0 <= chain[i_near, 4] <= 95.0:
                    cand.append(sp)
        if not cand:
            raise RuntimeError(
                "S2: no spawn with a signalized stop line 55-95 m ahead")
        rng = random.Random(seed)
        sp = cand[rng.randrange(len(cand))]

        # hold the whole junction cluster RED from t=0 (lights on this
        # server have no set_freeze; re-asserted per tick below)
        chain = _chain(mp.get_waypoint(sp.location), 130.0)
        chosen_tc = None
        best_d = 1e9
        for li, loc in signalized:
            d = np.hypot(chain[:, 0] - loc[0], chain[:, 1] - loc[1])
            i_near = int(np.argmin(d))
            if float(d[i_near]) < 6.0 and chain[i_near, 4] <= best_d and \
                    float(chain[i_near, 4]) <= 95.0 + 30.0:
                best_d = float(chain[i_near, 4])
                chosen_tc = li.trigger_volume.location
        if chosen_tc is None:
            raise RuntimeError("S2: chosen spawn lost its approach")
        for other in lights:
            otc = other.trigger_volume.location
            if math.hypot(otc.x - chosen_tc.x, otc.y - chosen_tc.y) <= 30.0:
                try:
                    self._light_snapshot.append(
                        (other, other.get_state(), other.get_red_time()))
                except Exception:
                    pass
                try:
                    # pin the group's red phase: CARLA group timers
                    # otherwise exit red after their default duration
                    # (~4 min), leaving yellow/green too long to survive
                    # per-tick re-assertion
                    other.set_red_time(1e6)
                    other.reset_group()
                except Exception:
                    pass
                try:
                    other.set_state(carla.TrafficLightState.Red)
                    self.red_lights.append(other)
                except Exception:
                    pass
        return sp, (lambda w, ego, bridge: None)

    def tick(self, world, sim_time_s: float) -> None:
        import carla
        for li in self.red_lights:
            try:
                li.set_state(carla.TrafficLightState.Red)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# S3: cut-in (kinematic NPC)


class S3CutIn(_ScriptedScenario):
    name = "S3"
    description = ("NPC 2 lanes left, ~25-40 m ahead, slower; cuts "
                   "into ego's lane starting at t=3 s")

    def resolve_setup(self, world, seed):
        mp = world.get_map()
        bp_lib = world.get_blueprint_library()
        npc_bp = bp_lib.find("vehicle.tesla.model3")
        last_err = None
        for sp in self._spawn_candidates(world, seed):
            wp0 = mp.get_waypoint(sp.location)
            chain = _chain(wp0, 45.0)
            wpn = wp0
            for _ in range(2):
                nxt = wpn.get_left_lane()
                if nxt is None:
                    break
                wpn = nxt
            if wpn.lane_id == wp0.lane_id:
                one = wp0.get_left_lane()
                if one is not None:
                    wpn = one
            nchain = _chain(wpn, 60.0)
            if len(nchain) < 15:
                last_err = RuntimeError("S3: NPC lane chain too short")
                continue
            z = float(chain[0, 2])
            i0 = int(np.argmax(nchain[:, 4] >= 30.0))
            npc = world.try_spawn_actor(
                npc_bp, _carla_T(nchain[i0, 0], nchain[i0, 1],
                                 z + 0.05, nchain[i0, 3]))
            if npc is None:
                last_err = RuntimeError("S3: NPC spawn failed")
                continue
            # path: own lane ~4 m (1 s at 4 m/s), then swing into ego's
            # lane touching down at ego-arc 26 m at t ~ 2.4 s, i.e.
            # ~1 s before the 7 m/s ego arrives -> forced braking event
            pts = [(float(p[0]), float(p[1]), z, 0.0)
                   for p in nchain[i0:i0 + 2]]
            land = _at(chain, 26)
            _append_curve(
                pts, pts[-1],
                (0.5 * (pts[-1][0] + land[0]),
                 0.5 * (pts[-1][1] + land[1])),
                land, n=12)
            i_land = int(np.argmin(np.hypot(chain[:, 0] - land[0],
                                            chain[:, 1] - land[1])))
            pts.extend([tuple(p[:4]) for p in chain[i_land:40]])
            self.kin.append(_Kinematic(npc, _Poly(np.asarray(pts)), 4.0))
            return sp, (lambda w, ego, bridge: None)
        raise last_err or RuntimeError("S3: no workable spawn")


# ---------------------------------------------------------------------------
# S4: jaywalker (kinematic walker)


class S4Jaywalker(_ScriptedScenario):
    name = "S4"
    description = ("walker crosses ego's path 15 m ahead outside a "
                   "marked crossing, 1.2 m/s")

    def resolve_setup(self, world, seed):
        import carla
        mp = world.get_map()
        bp = world.get_blueprint_library().filter("walker.pedestrian.0001")
        last_err = None
        for sp in self._spawn_candidates(world, seed):
            wp0 = mp.get_waypoint(sp.location)
            chain = _chain(wp0, 40.0)
            if len(chain) < 20:
                last_err = RuntimeError("S4: ego chain too short")
                continue
            z = float(chain[0, 2])
            c = _at(chain, 15)
            wp15 = mp.get_waypoint(
                carla.Location(x=float(c[0]), y=float(c[1]), z=z))
            yaw = math.radians(wp15.transform.rotation.yaw)
            perp = np.array([-math.sin(yaw), math.cos(yaw)])
            center = np.array([float(c[0]), float(c[1])])
            a = center + perp * 5.5
            b = center - perp * 5.5
            pts = [(float(a[0]), float(a[1]), z, _yaw_deg(a, b))]
            for k in range(1, 12):
                p = a + (b - a) * (k / 11.0)
                pts.append((float(p[0]), float(p[1]), z, _yaw_deg(a, b)))
            walker = world.try_spawn_actor(
                bp[0], _carla_T(float(a[0]), float(a[1]), z + 0.30, 0.0))
            if walker is None:
                last_err = RuntimeError("S4: walker spawn failed")
                continue
            self.kin.append(_Kinematic(walker, _Poly(np.asarray(pts)), 1.2))
            return sp, (lambda w, ego, bridge: None)
        raise last_err or RuntimeError("S4: no workable spawn")


# ---------------------------------------------------------------------------
# S5: oncoming left turn (kinematic NPC)


class S5OncomingLeft(_ScriptedScenario):
    name = "S5"
    description = ("NPC enters opposing lane 15 m before its turn; "
                   "starts left turn across ego's path at t=2.5 s")

    def resolve_setup(self, world, seed):
        mp = world.get_map()
        bp_lib = world.get_blueprint_library()
        npc_bp = bp_lib.find("vehicle.tesla.model3")
        last_err = None
        for sp in self._spawn_candidates(world, seed):
            wp0 = mp.get_waypoint(sp.location)
            chain = _chain(wp0, 45.0)
            if len(chain) < 30:
                last_err = RuntimeError("S5: ego chain too short")
                continue
            z = float(chain[0, 2])
            wpo = wp0.get_left_lane()
            if wpo is None:
                last_err = RuntimeError("S5: no opposing lane at spawn")
                continue
            ochain = _chain(wpo, 45.0)
            turn = _at(chain, 10)
            land = _at(chain, 22)
            d = np.hypot(ochain[:, 0] - turn[0], ochain[:, 1] - turn[1])
            i_turn = int(np.argmin(d))
            pts = [(float(p[0]), float(p[1]), z, 0.0)
                   for p in ochain[max(0, i_turn - 15):i_turn + 1]]
            if len(pts) < 3:
                last_err = RuntimeError("S5: opposing chain too short")
                continue
            _append_curve(
                pts, pts[-1],
                (0.5 * (pts[-1][0] + land[0]),
                 0.5 * (pts[-1][1] + land[1]) + 3.5),
                land, n=16)
            i_land = int(np.argmin(np.hypot(chain[:, 0] - land[0],
                                            chain[:, 1] - land[1])))
            pts.extend([tuple(p[:4]) for p in chain[i_land:34]])
            poly = _Poly(np.asarray(pts))
            p0 = poly.pose(0.0)
            npc = world.try_spawn_actor(
                npc_bp, _carla_T(p0[0], p0[1], z + 0.05, p0[3]))
            if npc is None:
                last_err = RuntimeError("S5: NPC spawn failed")
                continue
            self.kin.append(_Kinematic(npc, poly, 6.0))
            return sp, (lambda w, ego, bridge: None)
        raise last_err or RuntimeError("S5: no workable spawn")


_REGISTRY = {s.name: s() for s in
             (S1FreeDrive, S2RedLightRun, S3CutIn, S4Jaywalker,
              S5OncomingLeft)}


def get_scenario(name: str) -> _ScriptedScenario:
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown scenario '{name}' (registered: "
            f"{sorted(_REGISTRY)})")
    return _REGISTRY[name]

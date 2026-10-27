"""CARLA sim state -> checker-ingress context builders (P25-B).

Converts live CARLA actors/lights/map into the dicts TrajectoryChecker
and the correctors consume at `check(...)`:

  build_agent_context  -> List[(6,2) float32] CV rollouts, ego frame
  build_light_context  -> (intersections, lights) dicts, ego frame
  build_map_context    -> lanes meta dict, ego frame
  build_zone_context   -> None in v1 (no scripted zones)

UNITS: every length in the exported dicts is METERS.
TrajectoryChecker scales to integer centimetres internally
(checker.SCALE_FACTOR = 100); the bridge passes dicts straight
through (deviation from nextstep.md's "x100 before export" phrasing
— the checker does the x100 — recorded in docs/carla_harness.md
Deviations).

FRAMES: CARLA world (x fwd, y RIGHT, z up) vs planner ego (x fwd,
y LEFT, z up): points convert with FLIP = diag(1,-1,1,1),
    p_ego = FLIP @ inv(T_world_veh) @ p_world.
Builders accept an optional ego_transform (carla.Transform); when it
is omitted they emit WORLD-frame coordinates (inspection only).
The bridge always passes the ego transform.

Selftest (no server, no GPU):
    python experiments/carla_context.py --selftest
runs duck-typed fixtures through the builders into a real
TrajectoryChecker ("rules"; clingo CPU) and prints check() verdicts
for one canned unsafe fixture.
"""

import fnmatch
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

FLIP = np.diag([1.0, -1.0, 1.0, 1.0])
FLIP3 = np.diag([1.0, -1.0, 1.0])
CULL_RANGE_M = 160.0
MAX_AGENTS = 20
AGENT_PATTERNS = ("vehicle.*", "walker.*")
LIGHT_COLOR_MAP = {"Green": "green", "Yellow": "yellow", "Red": "red"}


def world_point_to_ego(p_world: np.ndarray, t_world_veh: np.ndarray):
    """World point (3,) -> planner-ego point (3,)."""
    p_veh = np.linalg.inv(t_world_veh) @ np.append(p_world[:3], 1.0)
    return (FLIP @ p_veh)[:3]


def world_vector_to_ego(v_world: np.ndarray, t_world_veh: np.ndarray):
    """World velocity (3,) -> planner-ego velocity (3,)."""
    r = np.linalg.inv(t_world_veh)[:3, :3]
    return FLIP3 @ (r @ v_world[:3])


def to_planner_ego_matrix(t_world_veh: np.ndarray) -> np.ndarray:
    """CARLA vehicle pose -> nuScenes ego2global (p_world = M @ p_ego)."""
    return t_world_veh @ FLIP


def transform_matrix(t) -> np.ndarray:
    return np.asarray(t.get_matrix(), dtype=np.float64)


# ---------------------------------------------------------------------------
# agents


def build_agent_context(world, ego_vehicle, ego_plan,
                        horizon_s: float = 3.0, dt: float = 1.0,
                        cull_range_m: float = CULL_RANGE_M,
                        max_agents: int = MAX_AGENTS,
                        force_ids=None) -> List[np.ndarray]:
    """Tracked vehicles/walkers -> CV-rollout (6,2) ego-frame trajs.

    Mirrors SafetyLayerPipeline._detections_to_trajectories semantics
    (steps = arange(6) * dt from the T=0 anchor; parked actors keep
    whatever velocity they have) and the checker's range/closing
    conflict retention (_cull_agents policy; range 160 m per
    nextstep.md 4.1). force_ids keeps scenario actors regardless of
    range and sorts them first.
    """
    del ego_plan, horizon_s  # rollout length is fixed at 6 @ dt
    try:
        t_world_veh = transform_matrix(ego_vehicle.get_transform())
        actors = world.get_actors()
    except Exception:
        return []

    force = set(force_ids or [])
    rows = []
    for a in actors:
        try:
            type_id = getattr(a, "type_id", "")
            if not any(fnmatch.fnmatch(type_id, p) for p in AGENT_PATTERNS):
                continue
            if a.id == ego_vehicle.id:
                continue
            loc = a.get_location()
            p0 = world_point_to_ego(
                np.array([loc.x, loc.y, loc.z], dtype=np.float64),
                t_world_veh)
            vel = a.get_velocity()
            v3 = world_vector_to_ego(
                np.array([vel.x, vel.y, vel.z], dtype=np.float64),
                t_world_veh)
            vx, vy = float(v3[0]), float(v3[1])
        except (RuntimeError, AttributeError):
            continue  # actor died mid-read (garbage collection race)

        steps = np.arange(6, dtype=np.float32) * dt
        traj = np.column_stack([
            p0[0] + vx * steps, p0[1] + vy * steps]).astype(np.float32)

        d0 = float(np.hypot(p0[0], p0[1]))
        in_conflict = bool(
            np.min(np.hypot(traj[:, 0], traj[:, 1])) <= cull_range_m)
        if a.id in force:
            d_sort = 0.0
        elif not in_conflict:
            continue
        else:
            d_sort = d0
        rows.append((d_sort, a.id, traj))

    rows.sort(key=lambda r: r[0])
    return [traj for _, _, traj in rows[:max_agents]]


# ---------------------------------------------------------------------------
# lights


def _light_trigger_center(li) -> Optional[np.ndarray]:
    try:
        loc = li.trigger_volume.location
        return np.array([loc.x, loc.y, loc.z], dtype=np.float64)
    except (RuntimeError, AttributeError):
        return None


def _light_state_color(li) -> Optional[str]:
    try:
        state = li.get_state()
        name = getattr(state, "name",
                       str(state).replace("TrafficLightState.", ""))
        return LIGHT_COLOR_MAP.get(name)
    except (RuntimeError, AttributeError):
        return None


def _light_stop_points(li) -> list:
    try:
        return [(wp.transform.location.x, wp.transform.location.y)
                for wp in (li.get_stop_waypoints() or [])]
    except (RuntimeError, AttributeError):
        return []


def build_light_context(world, ego_location, cluster_radius: float = 60.0,
                        ego_transform=None) -> Tuple[Dict, Dict]:
    """Nearby traffic lights -> pseudo-intersections + light facts.

    Lights whose trigger-volume center lies within cluster_radius of
    ego are union-find clustered by mutual proximity; each cluster is
    ONE iid (mean trigger center; radius = max distance from center
    to member stop waypoints + 10 m). Off/Unknown states are skipped
    entirely (absent facts = no rule claim).

    ego_transform: carla.Transform of the ego vehicle; when given,
    intersections/lights export PLANNER-EGO-frame meters (the bridge
    passes it; the signature extension vs the frozen nextstep.md
    signature is logged in docs/carla_harness.md Deviations). When
    omitted, WORLD-frame meters.
    """
    t_wv = transform_matrix(ego_transform) if ego_transform is not None \
        else None

    try:
        lights_actors = [a for a in world.get_actors()
                         if "traffic_light" in getattr(a, "type_id", "")]
    except Exception:
        return {}, {}

    eg = np.array([ego_location.x, ego_location.y, ego_location.z])

    near = []
    for li in lights_actors:
        c = _light_trigger_center(li)
        if c is None:
            continue
        if float(np.hypot(c[0] - eg[0], c[1] - eg[1])) <= cluster_radius:
            color = _light_state_color(li)
            if color is None:
                continue
            near.append((li, c, color))

    parent = list(range(len(near)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            if np.hypot(*(near[i][1][:2] - near[j][1][:2])) <= cluster_radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    clusters: Dict[int, list] = {}
    for i in range(len(near)):
        clusters.setdefault(find(i), []).append(near[i])

    intersections: Dict[str, Dict] = {}
    lights: Dict[str, Dict] = {}
    for members in clusters.values():
        center_w = np.stack([c for _, c, _ in members]).mean(axis=0)
        iid = f"light_{min(a.id for a, _, _ in members)}"
        # Cluster radius: the tightest member's own stop-line ring.
        # CARLA light actors reference stop waypoints across the map;
        # max-over-all-stops inflates the box to 100+ m, which paints
        # "inside the red junction" violations from far away and makes
        # hold-position corrections unfixable. Min member ring covers
        # the physical crossing without the far references.
        member_rings = []
        for a, c_mem, _ in members:
            own = _light_stop_points(a)
            if own:
                own = np.asarray(own)
                ring = float(np.max(np.hypot(
                    own[:, 0] - c_mem[0], own[:, 1] - c_mem[1])))
                member_rings.append(ring)
        if member_rings:
            radius = min(member_rings) + 10.0
        else:
            radius = 30.0
        p = center_w
        if t_wv is not None:
            p = world_point_to_ego(p, t_wv)
        entry = {"x": float(p[0]), "y": float(p[1]), "radius": radius}
        intersections[iid] = entry
        lights[iid] = {"color": members[0][2], "times": [0, 1, 2, 3, 4, 5],
                       "x": entry["x"], "y": entry["y"],
                       "radius": radius}
        lights[iid].pop("time", None)
    return intersections, lights


# ---------------------------------------------------------------------------
# map


def build_map_context(world_map, ego_location, ego_transform=None,
                      forward_m: float = 60.0, backward_m: float = 20.0,
                      step_m: float = 3.0) -> Dict:
    """Current-lane centerline -> lanes meta dict (meters).

    OpenDrive walk along the ego's projected driving-lane waypoint
    (forward_m / backward_m at step_m). Emits lane segments,
    lane_width, lane_direction, road_type='urban'. Intersections and
    lights come from build_light_context, not here (Option B retreat
    candidate; retreat decision logged in docs/carla_harness.md).

    ego_transform: when given, vertices are PLANNER-EGO frame meters.
    """
    try:
        import carla
        driving = carla.LaneType.Driving
    except Exception:
        driving = 1

    wp = world_map.get_waypoint(ego_location, project_to_road=True,
                                lane_type=driving)
    if wp is None:
        return {}

    lid = f"carla_r{wp.road_id}_l{wp.lane_id}"
    seq = [wp.transform.location]
    cur = wp
    dist = 0.0
    while dist < forward_m:
        nxt = cur.next(step_m)
        if not nxt:
            break
        loc = nxt[0].transform.location
        if loc.x == seq[-1].x and loc.y == seq[-1].y:
            cur = nxt[0]
            break
        seq.append(loc)
        cur = nxt[0]
        dist += step_m
    cur = wp
    dist = 0.0
    while dist < backward_m:
        prev = cur.previous(step_m)
        if not prev:
            break
        loc = prev[0].transform.location
        if loc.x == seq[0].x and loc.y == seq[0].y:
            cur = prev[0]
            break
        seq.insert(0, loc)
        cur = prev[0]
        dist += step_m

    t_wv = transform_matrix(ego_transform) if ego_transform is not None \
        else None
    pts = []
    for loc in seq:
        p = np.array([loc.x, loc.y, loc.z], dtype=np.float64)
        if t_wv is not None:
            p = world_point_to_ego(p, t_wv)
        pts.append((float(p[0]), float(p[1])))

    segments = [{"x1": pts[i][0], "y1": pts[i][1],
                 "x2": pts[i + 1][0], "y2": pts[i + 1][1]}
                for i in range(len(pts) - 1)]

    mapd: Dict = {"lanes": {lid: segments},
                  "lane_width": {lid: float(wp.lane_width)},
                  "road_type": {lid: "urban"}}
    if len(pts) >= 2:
        d = np.array([pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]])
        norm = np.hypot(*d)
        if norm > 1e-6:
            mapd["lane_direction"] = {lid: [
                float(d[0] / norm), float(d[1] / norm)]}  # [dx, dy] list
    return mapd


def build_zone_context(scenario_cfg) -> None:
    """v1: always None (no school/construction zones scripted)."""
    del scenario_cfg
    return None


# ---------------------------------------------------------------------------
# selftest (no server, no GPU)


class _Location:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

    def __eq__(self, other):
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)


class _Waypoint:
    def __init__(self, road_id, lane_id, x, y, width=3.5):
        self.road_id, self.lane_id = road_id, lane_id
        self.lane_width = width
        self.transform = type("T", (), {"location": _Location(x, y)})()
        self._next: list = []
        self._previous: list = []

    def next(self, dist):
        if self._next:
            return self._next
        return []

    def previous(self, dist):
        if self._previous:
            return self._previous
        return []


class _ActorList(list):
    def filter(self, pattern):
        return [a for a in self if fnmatch.fnmatch(a.type_id, pattern)]

    def get_actors(self):
        return self


class _Transform:
    def get_matrix(self):
        return np.eye(4)


class _BoundingBox:
    def __init__(self, loc):
        self.location = loc


class _State:
    def __init__(self, name):
        self.name = name


class _Actor:
    _counter = 0

    def __init__(self, type_id, x, y, vx=0.0, vy=0.0, actor_id=None):
        _Actor._counter += 1
        self.id = actor_id if actor_id is not None else _Actor._counter
        self.type_id = type_id
        self._loc = _Location(x, y)
        self._vel = _Location(vx, vy)

    def transform(self):
        return _Transform()

    def get_transform(self):
        return _Transform()

    def get_location(self):
        return self._loc

    def get_velocity(self):
        return self._vel


class _Light(_Actor):
    def __init__(self, x, y, color):
        super().__init__("traffic.traffic_light", x, y,
                         actor_id=None)
        self._color = color

    def get_state(self):
        return _State(self._color)

    def get_trigger_volume(self):
        return _BoundingBox(self._loc)

    def get_stop_waypoints(self):
        return [_Waypoint(1, -1, self._loc.x, self._loc.y)]


class _Map:
    def __init__(self, wp):
        self._wp = wp

    def get_waypoint(self, location, project_to_road=True, lane_type=1):
        return self._wp


def _run_selftest() -> int:
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "src"))
    from checker import TrajectoryChecker

    print("[selftest] fixtures (no server, no GPU)")

    ego = _Actor("vehicle.tesla.model3", 0.0, 0.0, actor_id=1)

    # CARLA world (y RIGHT): agent 6 m ahead 1 m right of ego lane,
    # closing 3 m/s -> planner ego (6, +1).
    agent = _Actor("vehicle.tesla.model3", 6.0, -1.0, vx=-3.0, actor_id=2)
    far = _Actor("vehicle.tesla.model3", 200.0, 200.0, vx=0.5, actor_id=3)
    forced = _Actor("vehicle.tesla.model3", 200.0, 0.0, vx=-2.0, actor_id=4)
    walker = _Actor("walker.pedestrian.0001", 5.0, -2.0, vx=-1.0, actor_id=5)
    world = _ActorList([ego, agent, far, forced, walker])

    plan = np.column_stack([
        np.arange(1, 7, dtype=np.float32), np.zeros(6, dtype=np.float32)])

    agents = build_agent_context(world, ego, plan, force_ids=[4])
    order = [float(tr[0, 0]) for tr in agents]
    print(f"[selftest] agent_context: {len(agents)} trajs "
          f"(expect 3: forced(200), walker(5), vehicle(6); far dropped): "
          f"T0 x={order}")
    assert len(agents) == 3, "agent context culling/cap failed"
    assert abs(order[0] - 200.0) < 1e-4, "forced id not first"
    assert abs(agents[1][0, 1] - 2.0) < 1e-4, \
        "world->ego y flip wrong (walker CARLA -2 must be +2)"

    # light with ego transform -> planner-ego frame conversion
    light = _Light(10.0, -5.0, "Red")   # CARLA -y = planner +y (left)
    inter, lights = build_light_context(_ActorList([light]),
                                        _Location(0, 0),
                                        ego_transform=_Transform())
    iid = next(iter(lights))
    print(f"[selftest] light_context: iid={iid} "
          f"color={lights[iid].get('color')} "
          f"egoxy=({inter[iid]['x']:.1f}, {inter[iid]['y']:.1f})")
    assert lights[iid].get("color") == "red"
    assert abs(inter[iid]["x"] - 10.0) < 1e-6
    assert abs(inter[iid]["y"] - 5.0) < 1e-6, \
        "light ego-frame conversion wrong"

    off_inter, off_lights = build_light_context(
        _ActorList([_Light(3.0, 3.0, "Off")]),
        _Location(0, 0), ego_transform=_Transform())
    assert not off_lights, "Off state must be skipped"

    # chain
    w0 = _Waypoint(1, -1, 0.0, 0.0)
    w1 = _Waypoint(1, -1, 3.0, 0.0)
    w2 = _Waypoint(1, -1, 6.0, 0.0)
    w3 = _Waypoint(1, -1, 9.0, 0.0)
    w3._previous = []
    w2._previous = [w1]
    w0._next = [w1]
    w1._next = [w2]
    w2._next = [w3]
    mapd = build_map_context(_Map(w0), _Location(0, 0),
                             ego_transform=_Transform())
    lid = list(mapd["lanes"])[0]
    print(f"[selftest] map_context: lane={lid} "
          f"segments={len(mapd['lanes'][lid])} "
          f"width={mapd['lane_width'][lid]:.1f}")
    assert len(mapd["lanes"][lid]) >= 3, "lane segment walk broken"
    assert abs(mapd["lane_width"][lid] - 3.5) < 1e-6

    checker = TrajectoryChecker(str(ROOT / "rules"))
    is_safe, violations = checker.check(plan, other_vehicles=agents,
                                        map_data=mapd,
                                        traffic_lights=lights)
    print(f"[selftest] check(unsafe fixture): is_safe={is_safe}, "
          f"violations={violations}")
    assert not is_safe, "expected violations on the unsafe fixture"

    safe = np.column_stack([
        np.linspace(0.0, 0.4, 6, dtype=np.float32),
        np.zeros(6, dtype=np.float32)])
    is_safe2, violations2 = checker.check(
        safe, other_vehicles=[], map_data=mapd,
        traffic_lights=build_light_context(
            _ActorList([]), _Location(0, 0))[1])
    print(f"[selftest] check(safe fixture): is_safe={is_safe2}, "
          f"violations={violations2}")
    assert is_safe2 and not violations2, "safe fixture flagged"

    print("SELFTEST OK")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_run_selftest())
    print(__doc__)
    sys.exit(1)

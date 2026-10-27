"""Checker edge-semantics tests (TG-4 / T-7).

Shape validation, malformed-agent tolerance, cull-radius behavior,
warning-vs-violation separation, light timing keys, and determinism.
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from checker import TrajectoryChecker

_RULES_DIR = str(Path(__file__).parent.parent / "rules")


def _checker(**kwargs):
    return TrajectoryChecker(_RULES_DIR, **kwargs)


def test_shape_validation():
    """Wrong shapes must raise ValueError, never silently ground."""
    c = _checker()
    for bad in [
        np.zeros((6, 3), dtype=np.float32),   # wrong width
        np.zeros((7, 2), dtype=np.float32),   # too many steps
        np.zeros((2, 2, 2), dtype=np.float32),  # wrong ndim
    ]:
        try:
            c.check(bad)
        except ValueError:
            continue
        raise AssertionError(f"no ValueError for shape {bad.shape}")
    print("  [PASS] Shape validation rejects bad inputs")


def test_malformed_agents_skipped():
    """Empty/malformed other_vehicles entries are skipped, not fatal."""
    c = _checker()
    ego = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        dtype=np.float32,
    )
    ok, v = c.check(
        ego,
        other_vehicles=[
            np.zeros((0, 2), dtype=np.float32),  # empty
            np.zeros((6, 3), dtype=np.float32),  # wrong width
            "junk",                              # wrong type
        ],
    )
    assert ok, f"malformed agents should be skipped, got {v}"
    print("  [PASS] Malformed agents skipped without crash")


def test_cull_radius():
    """Agents beyond the cull radius are ignored."""
    c = _checker(max_agent_range_m=150.0)
    ego = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        dtype=np.float32,
    )
    far = np.array(
        [[200 + t, 0] for t in range(6)],
        dtype=np.float32,
    )
    ok, v = c.check(ego, other_vehicles=[far])
    assert ok, f"200 m agent should be culled, got {v}"
    near = np.array(
        [[0, 1] for _ in range(6)],
        dtype=np.float32,
    )
    ok2, v2 = c.check(ego, other_vehicles=[near])
    assert not ok2, "1 m agent must not be culled"
    print("  [PASS] Cull radius keeps near agents, drops far ones")


def test_warnings_never_unsafe():
    """Advisory warnings must not flip is_safe (R2.5 lane_departure)."""
    c = _checker()
    # y=1.8m: outer ~20% band of a 4m lane -> warning only, still in lane.
    traj = np.array(
        [[0, 1.8], [1, 1.8], [2, 1.8], [3, 1.8], [4, 1.8], [5, 1.8]],
        dtype=np.float32,
    )
    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 100, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
    }
    ok, v = c.check(traj, map_data=map_data)
    warns = [x for x in v if x.get('type', 'violation') == 'warning']
    hard = [x for x in v if x.get('type', 'violation') != 'warning']
    assert warns, f"expected lane_departure warning, got {v}"
    assert any(d['rule'] == 'lane_departure' for d in warns), \
        f"warning must be named lane_departure, got {v}"
    assert ok and not hard, f"warning flipped is_safe: {v}"
    print("  [PASS] Warnings reported but never unsafe")


def test_light_time_keys():
    """Both 'time' and 'times' light keys must produce findings."""
    c = _checker()
    traj = np.array(
        [[14, 0], [16, 0], [18, 0], [20, 0], [22, 0], [24, 0]],
        dtype=np.float32,
    )
    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 50, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
        'controls_approach': {'i1': 'l1'},
    }
    for lights in [
        {'i1': {'color': 'red', 'time': 3, 'x': 20, 'y': 0, 'radius': 3}},
        {'i1': {'color': 'red', 'times': [3], 'x': 20, 'y': 0, 'radius': 3}},
    ]:
        ok, v = c.check(traj, map_data=map_data, traffic_lights=lights)
        assert not ok and any(
            d['rule'] == 'red_light' and d['time'] == 3 for d in v
        ), f"expected red_light@3, got {v}"
    print("  [PASS] 'time' and 'times' keys both work")


def test_determinism():
    """Repeated checks of the same scene must agree exactly."""
    c = _checker()
    ego = np.array(
        [[0, 0], [5, 0], [10, 0], [15, 0], [20, 0], [25, 0]],
        dtype=np.float32,
    )
    other = np.array(
        [[10, 3], [11, 3], [12, 3], [13, 3], [14, 3], [15, 3]],
        dtype=np.float32,
    )
    r1 = c.check(ego, other_vehicles=[other])
    r2 = c.check(ego, other_vehicles=[other])
    assert r1[0] == r2[0] and r1[1] == r2[1], f"{r1} != {r2}"
    print("  [PASS] Checks are deterministic")


def test_explicit_times_honored():
    """Zone guard/worker facts fire only at listed timesteps (P-15).

    No invented phases: a stop guard at [3] must yield exactly
    school_guard_stop@3; a guard at [1] must yield exactly @1 (not @3).
    """
    c = _checker()
    traj = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        dtype=np.float32,
    )

    def _zones(guard_times):
        return [{
            'id': 'z1', 'x1': -5.0, 'y1': -5.0, 'x2': 15.0, 'y2': 5.0,
            'type': 'school', 'active': True,
            'crossing_guard': 'stop',
            'crossing_guard_times': guard_times,
        }]

    ok, v = c.check(traj, zones=_zones([3]))
    hits = sorted(d['time'] for d in v if d['rule'] == 'school_guard_stop')
    assert not ok and hits == [3], f"expected guard hit @3 only, got {v}"

    ok, v = c.check(traj, zones=_zones([1]))
    hits = sorted(d['time'] for d in v if d['rule'] == 'school_guard_stop')
    assert not ok and hits == [1], f"expected guard hit @1 only, got {v}"
    print("  [PASS] Explicit guard times honored, no invented phases")


def test_check_aliasing():
    """P-15: check() never mutates input; safe path returns identical object."""
    c = _checker()
    safe = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        dtype=np.float32,
    )
    snapshot = safe.copy()
    ok, _ = c.check(safe)
    assert ok
    np.testing.assert_array_equal(safe, snapshot)

    out = c.get_safe_trajectory(safe)
    assert out is safe, "safe path must return the identical object"

    fast = np.array(
        [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]],
        dtype=np.float32,
    )
    snapshot = fast.copy()
    out = c.get_safe_trajectory(fast)
    np.testing.assert_array_equal(fast, snapshot)
    assert out is not fast, "unsafe path must return a new object"
    print("  [PASS] check()/correction aliasing contract pinned")


def test_approach_vs_leave():
    """P-4: about_to_enter fires approaching, silent leaving/holding."""
    c = _checker()
    imap = {
        'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}]},
        'lane_width': {'l1': 4.0},
        'intersections': {'i1': {'x': 20.0, 'y': 0.0, 'radius': 2.0}},
    }
    ex = {'facts': [
        'queue_ahead(vehicle_0, i1, 2).', 'queue_ahead(vehicle_0, i1, 3).',
        'space_beyond_box(ego, vehicle_0, i1, 2, 100).',
        'space_beyond_box(ego, vehicle_0, i1, 3, 100).',
        'ego_length(400).', 'vehicle(vehicle_0).',
    ]}
    appr = np.array(
        [[10, 0], [12, 0], [14, 0], [16, 0], [18, 0], [20, 0]],
        dtype=np.float32,
    )
    ok, v = c.check(appr, map_data=imap, extras=ex)
    assert not ok and any(d['rule'] == 'box_blocking' for d in v), \
        f"approaching must fire box_blocking, got {v}"

    leave = np.array(
        [[20, 0], [22, 0], [24, 0], [26, 0], [28, 0], [30, 0]],
        dtype=np.float32,
    )
    ok, v = c.check(leave, map_data=imap, extras=ex)
    assert not any(d['rule'] == 'box_blocking' for d in v), \
        f"leaving must not fire box_blocking, got {v}"

    hold = np.array([[16, 0]] * 6, dtype=np.float32)
    ok, v = c.check(hold, map_data=imap, extras=ex)
    assert not any(d['rule'] == 'box_blocking' for d in v), \
        f"holding in band must not fire box_blocking, got {v}"
    print("  [PASS] about_to_enter distinguishes approach/leave/hold")


def test_stopped_before_proximity():
    """P-4: stopped_before needs a stop NEAR the feature (R6.5)."""
    c = _checker()
    m2 = {'crosswalks': [{'id': 'c1', 'x1': 10.0, 'y1': -3.0,
                           'x2': 14.0, 'y2': 3.0}]}
    ped = [{'id': 'p1',
            'positions': {t: (12.0, 0.0) for t in range(6)}}]
    ex = {'facts': ['in_crosswalk(1200, 0, c1).']}

    stop_far = np.array([[100, 0]] * 6, dtype=np.float32)
    ok, v = c.check(stop_far, map_data=m2, pedestrians=ped, extras=ex)
    assert ok, f"stopped 86 m away has nothing to yield to, got {v}"

    stop_near = np.array([[11, 0]] * 6, dtype=np.float32)
    ok, v = c.check(stop_near, map_data=m2, pedestrians=ped, extras=ex)
    assert ok, f"stop at the crosswalk already yields, got {v}"

    moving = np.array([[0, 0], [4, 0], [8, 0], [12, 0], [16, 0], [20, 0]],
                      dtype=np.float32)
    ok, v = c.check(moving, map_data=m2, pedestrians=ped, extras=ex)
    assert not ok and any(d['rule'] == 'crosswalk_yield' for d in v), \
        f"moving through must yield, got {v}"
    print("  [PASS] stopped_before is stop-line-relative")


def test_longitudinal_clearance_adjacency():
    """P-4: R4.2 is along-track; adjacent-lane proximity is R4.1's job."""
    c = _checker()
    m = {
        'lanes': {
            'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}],
            'l2': [{'x1': -50.0, 'y1': 5.0, 'x2': 50.0, 'y2': 5.0}],
        },
        'lane_width': {'l1': 4.0, 'l2': 4.0},
    }
    ego = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], dtype=np.float32)
    side = np.array(
        [[2, 5], [3, 5], [4, 5], [5, 5], [6, 5], [7, 5]], dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[side], map_data=m)
    assert not any(d['rule'] == 'longitudinal_clearance' for d in v), \
        f"adjacent lane must not fire R4.2, got {v}"

    tail = np.array(
        [[0, 0], [0.5, 0], [1, 0], [1.5, 0], [2, 0], [2.5, 0]],
        dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[tail], map_data=m)
    assert not ok and any(d['rule'] == 'longitudinal_clearance'
                          for d in v), f"same-lane tail must fire R4.2, got {v}"
    print("  [PASS] R4.2 uses along-track separation")


def test_conflict_aware_culling():
    """P-8/PR-3: fast closers beyond 150 m are kept; receders culled."""
    c = _checker(max_agent_range_m=150.0)
    ego = np.array([[0, 0]] * 6, dtype=np.float32)
    closer = np.array([[160 - 24 * t, 0] for t in range(6)],
                      dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[closer], include_internal=True)
    assert any(d.get('type') == 'ttc' for d in v), \
        f"24 m/s closer must be kept (ttc derived), got {v}"
    receder = np.array([[160 + 24 * t, 0] for t in range(6)],
                       dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[receder], include_internal=True)
    assert not any(d.get('type') == 'ttc' for d in v), \
        f"receder must be culled (no ttc), got {v}"
    print("  [PASS] Culling is conflict-aware")


def test_no_duplicate_findings():
    """RQ-4 regression: one solve per check, no duplicated dicts."""
    c = _checker()
    ego = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        dtype=np.float32,
    )
    other = np.array([[2.0, 0.0]] * 6, dtype=np.float32)
    _, v = c.check(ego, other_vehicles=[other])
    assert v, "overlap must produce findings"
    keys = [(d.get('rule'), d.get('vehicle'), d.get('vehicle2'),
             d.get('time'), d.get('type')) for d in v]
    assert len(keys) == len(set(keys)), f"duplicated findings: {v}"
    print("  [PASS] No duplicated findings")


def test_lane_end_tolerance():
    """Python in_lane_fact mirrors the ASP past-the-end tolerance."""
    c = _checker()
    m = {'lanes': {'l1': [{'x1': 0.0, 'y1': 0.0, 'x2': 10.0, 'y2': 0.0}]},
         'lane_width': {'l1': 4.0}}
    # 1 m past the end: e*(e+L) = 1*11 <= 16 -> in lane.
    near = np.array([[t, 0] for t in range(5)] + [[11.0, 0.0]],
                    dtype=np.float32)
    ok, v = c.check(near, map_data=m)
    assert not any(d['rule'] == 'lane_keeping' and d['time'] == 5
                   for d in v), f"1 m past end must hold lane, got {v}"
    # 10 m past the end: e*(e+L) = 10*20 > 16 -> out.
    far = np.array([[t, 0] for t in range(5)] + [[20.0, 0.0]],
                   dtype=np.float32)
    ok, v = c.check(far, map_data=m)
    assert not ok and any(d['rule'] == 'lane_keeping' and d['time'] == 5
                          for d in v), f"10 m past end must leave, got {v}"
    print("  [PASS] Lane end tolerance mirrors ASP clause")


if __name__ == "__main__":
    test_shape_validation()
    test_malformed_agents_skipped()
    test_cull_radius()
    test_warnings_never_unsafe()
    test_light_time_keys()
    test_determinism()
    test_explicit_times_honored()
    test_check_aliasing()
    test_approach_vs_leave()
    test_stopped_before_proximity()
    test_longitudinal_clearance_adjacency()
    test_conflict_aware_culling()
    test_no_duplicate_findings()
    test_lane_end_tolerance()
    print("Checker edge tests passed!")

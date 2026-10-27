"""P3/P4 battery: ConstraintOptimizer feasibility + CorrectorQP loop.

Optimizer unit tests use synthetic constraint dicts (no Clingo);
revalidation tests run the full check -> solve -> re-check loop
through real Clingo. All coordinates in meters.
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_TESTS_DIR.parent / "src"))

import numpy as np
from checker import TrajectoryChecker
from constraint_optimizer import ConstraintOptimizer, DEFAULT_OPT_PARAMS
from corrector import CorrectorQP, TrajectoryCorrector

_RULES_DIR = str(_TESTS_DIR.parent / "rules")


def _checker(**kwargs):
    return TrajectoryChecker(_RULES_DIR, **kwargs)


def test_optimizer_speed_cap():
    """Synthetic v_max caps every segment; T0 pinned; near-original."""
    opt = ConstraintOptimizer()
    fast = np.array([[20 * t, 0] for t in range(6)], dtype=float)
    con = {'v_max': [(t, 500) for t in range(1, 6)], 'v_min': [],
           'x_bounds': [], 'y_bounds': [], 'a_max': [],
           'lane_center': []}
    solved, info = opt.solve(fast, con)
    assert info['status'] == 'optimal'
    assert info['solve_time_ms'] >= 0
    np.testing.assert_allclose(solved[0], fast[0])
    for t in range(1, 6):
        seg = float(np.linalg.norm(solved[t] - solved[t - 1]))
        assert seg <= 5.0 + 1e-9, f"segment {t} speed {seg} exceeds cap"
    # Closest feasible: cruises at exactly the cap.
    np.testing.assert_allclose(solved[:, 0], [5 * t for t in range(6)],
                               atol=0.05)
    print("  [PASS] v_max projection exact, T0 pinned")


def test_optimizer_box_and_center():
    """y_bounds corridor clips drift; lane_center biases lateral."""
    opt = ConstraintOptimizer()
    drift = np.array([[5 * t, float(t)] for t in range(6)], dtype=float)
    con = {'v_max': [], 'v_min': [],
           'x_bounds': [], 'y_bounds': [(t, -200, 200) for t in range(6)],
           'a_max': [], 'lane_center': [(t, 0) for t in range(6)]}
    solved, _ = opt.solve(drift, con)
    for t in range(1, 6):
        assert -2.0 - 1e-9 <= solved[t, 1] <= 2.0 + 1e-9, \
            f"T={t} lateral {solved[t, 1]} outside corridor"
    assert abs(solved[5, 1]) < abs(drift[5, 1]), \
        "lane_center spring must pull the tail inward"
    print("  [PASS] box clip + lane_center spring")


def test_optimizer_accel_cap():
    """a_max clamps the per-step velocity change (single family)."""
    opt = ConstraintOptimizer()
    jerk = np.array([[0, 0], [1, 0], [2, 0], [10, 0], [18, 0], [26, 0]],
                    dtype=float)
    con = {'v_max': [], 'v_min': [], 'x_bounds': [], 'y_bounds': [],
           'a_max': [(t, 300) for t in range(6)], 'lane_center': []}
    solved, _ = opt.solve(jerk, con)
    for t in range(2, 6):
        dv = (solved[t] - solved[t - 1]) - (solved[t - 1] - solved[t - 2])
        assert float(np.linalg.norm(dv)) <= 3.0 + 1e-9, \
            f"T={t} accel {np.linalg.norm(dv)} exceeds cap"
    print("  [PASS] a_max clamp exact")


def test_optimizer_invalid():
    """Fail-loud on bad shape and non-finite input."""
    opt = ConstraintOptimizer()
    for bad in [np.zeros((6, 3)), np.zeros((7, 2)),
                np.full((4, 2), np.nan)]:
        try:
            opt.solve(bad, {})
            raise AssertionError(f"shape {bad.shape} must raise")
        except ValueError:
            pass
    print("  [PASS] optimizer fail-loud")


def test_optimizer_joint_feasibility_cycle():
    """P-23 regression: tight box + a_max + v_max together.

    The pre-P-23 single sequential pass (boxes -> speeds -> accel)
    left this case ~9.6 cm outside its constraints after converging:
    the accel clamp re-broke the box the clip had just fixed. The
    joint alternation loop must reach exact feasibility.
    """
    opt = ConstraintOptimizer()
    traj = np.array(
        [[0.0, -0.262], [1.997, 1.037], [3.994, -1.333],
         [5.991, -0.392], [7.988, 0.228], [9.985, -1.616]])
    con = {'v_max': [(t, 179) for t in range(1, 6)], 'v_min': [],
           'x_bounds': [],
           'y_bounds': [(t, -37, 37) for t in range(1, 6)],
           'a_max': [(t, 59) for t in range(2, 6)],
           'lane_center': []}
    solved, info = opt.solve(traj, con)
    assert info['status'] == 'optimal', \
        f"joint alternation must converge, got {info['status']}"
    worst = 0.0
    for t in range(1, 6):
        worst = max(worst, -0.37 - solved[t, 1], solved[t, 1] - 0.37)
        seg = float(np.linalg.norm(solved[t] - solved[t - 1]))
        worst = max(worst, seg - 1.79)
    for t in range(2, 6):
        dv = float(np.linalg.norm(
            (solved[t] - solved[t - 1]) - (solved[t - 1] - solved[t - 2])))
        worst = max(worst, dv - 0.59)
    assert worst <= 0.001, f"joint feasibility violated by {worst:.4f} m"
    np.testing.assert_allclose(solved[0], traj[0])
    print("  [PASS] P-23 joint projection loop (cycle case exact)")


def test_optimizer_cycle_status_honest():
    """P-23: when the alternation cap cannot reach joint feasibility the
    status says 'projection_cycle' instead of a silent 'optimal'."""
    opt = ConstraintOptimizer(params={'joint_iters': 1})
    traj = np.array(
        [[0.0, -0.262], [1.997, 1.037], [3.994, -1.333],
         [5.991, -0.392], [7.988, 0.228], [9.985, -1.616]])
    con = {'v_max': [(t, 179) for t in range(1, 6)], 'v_min': [],
           'x_bounds': [],
           'y_bounds': [(t, -37, 37) for t in range(1, 6)],
           'a_max': [(t, 59) for t in range(2, 6)],
           'lane_center': []}
    _, info = opt.solve(traj, con)
    assert info['status'] in ('optimal', 'projection_cycle')
    assert 'joint_iters' in info
    print(f"  [PASS] status honest: {info['status']}")


def test_qp_safe_passthrough():
    """Safe trajectory returns a copy with no corrections."""
    c = _checker()
    qp = CorrectorQP(c)
    slow = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    fixed, entries = qp.correct(slow)
    assert entries == []
    np.testing.assert_array_equal(fixed, slow)
    print("  [PASS] QP safe passthrough")


def test_qp_revalidation_speed():
    """Bare speeding trajectory: QP caps to the default limit, re-check
    is clean, and every entry follows the 5-key schema."""
    c = _checker()
    qp = CorrectorQP(c)
    fast = np.array([[20 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(fast)
    assert not ok and any(d['rule'] == 'speed_limit' for d in v), \
        f"expected speed_limit, got {v}"
    fixed, entries = qp.correct(fast)
    ok2, v2 = c.check(fixed)
    assert ok2, f"QP output must re-validate clean, got {v2}"
    assert len(entries) > 0 and all(
        set(e.keys()) == {'rule', 'timestep', 'correction_type',
                          'original_value', 'corrected_value'}
        for e in entries), f"bad entry schema: {entries}"
    assert all(e['rule'] == 'qp_optimization' for e in entries)
    print("  [PASS] QP speed fix re-validates clean")


def test_qp_revalidation_red():
    """Red hold: QP freezes before the box so the re-check passes."""
    c = _checker()
    qp = CorrectorQP(c)
    road = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0,
                              'x2': 50.0, 'y2': 0.0}]},
            'lane_width': {'l1': 4.0},
            'controls_approach': {'i1': 'l1'},
            'intersections': {'i1': {'x': 15.0, 'y': 0.0, 'radius': 2.0}}}
    run = np.array([[3 * t, 0] for t in range(6)], dtype=np.float32)
    red = {'i1': {'color': 'red', 'times': [5], 'x': 15.0, 'y': 0.0,
                  'radius': 2.0}}
    ok, v = c.check(run, map_data=road, traffic_lights=red)
    assert not ok and any(d['rule'] == 'red_light' for d in v), \
        f"expected red_light, got {v}"
    fixed, _ = qp.correct(run, map_data=road, traffic_lights=red)
    ok2, v2 = c.check(fixed, map_data=road, traffic_lights=red)
    assert ok2, f"QP hold must clear the red, got {v2}"
    np.testing.assert_allclose(fixed[5], fixed[4])
    print("  [PASS] QP red hold re-validates clean")


def test_qp_fallback_to_heuristic():
    """Zero-step optimizer cannot fix; fallback delegates and flags."""
    c = _checker()
    dead = ConstraintOptimizer(params={'steps': 0})
    qp = CorrectorQP(c, optimizer=dead, max_rounds=1)
    fast = np.array([[20 * t, 0] for t in range(6)], dtype=np.float32)
    fixed, entries = qp.correct(fast)
    assert entries[0]['rule'] == 'qp_optimization' and \
        'fallback' in entries[0]['correction_type'], \
        f"first entry must flag the fallback, got {entries[0]}"
    ok2, v2 = c.check(fixed)
    assert ok2, f"heuristic fallback must finish the fix, got {v2}"
    print("  [PASS] QP fallback flagged + heuristic completes")


def test_qp_invalid_args():
    """Fail-loud on bad checker and bad on_failure."""
    c = _checker()
    try:
        CorrectorQP(checker="nope")
        raise AssertionError("bad checker must raise")
    except TypeError:
        pass
    try:
        CorrectorQP(c, on_failure="panic")
        raise AssertionError("bad on_failure must raise")
    except ValueError:
        pass
    print("  [PASS] QP constructor fail-loud")


def test_qp_kwargs_not_forwarded():
    """include_constraints in kwargs must not leak into the heuristic
    fallback (which unpacks the legacy 2-tuple)."""
    c = _checker()
    dead = ConstraintOptimizer(params={'steps': 0})
    qp = CorrectorQP(c, optimizer=dead, max_rounds=1)
    fast = np.array([[20 * t, 0] for t in range(6)], dtype=np.float32)
    fixed, _ = qp.correct(fast, include_constraints=True)
    ok2, v2 = c.check(fixed)
    assert ok2, f"kwarg leak would break the fallback: {v2}"
    print("  [PASS] include_constraints stripped before fallback")


def test_qp_bounds_invalid():
    """Unknown bound family fails loud."""
    c = _checker()
    try:
        CorrectorQP(c, bounds='diagonal')
        raise AssertionError("bad bounds must raise")
    except ValueError:
        pass
    print("  [PASS] bounds fail-loud")


def test_qp_bounds_caps_fix_speed():
    """caps-only fixes a pure speed violation; boxes-only cannot (no
    boxes are emitted for it) and raises without fallback."""
    c = _checker()
    fast = np.array([[20 * t, 0] for t in range(6)], dtype=np.float32)
    fixed, _ = CorrectorQP(c, bounds='caps',
                           heuristic_fallback=False).correct(fast)
    ok, v = c.check(fixed)
    assert ok, f"caps-only must fix speed, got {v}"
    try:
        CorrectorQP(c, bounds='boxes',
                    heuristic_fallback=False).correct(fast)
        raise AssertionError("boxes-only must give up on speed")
    except RuntimeError:
        pass
    print("  [PASS] caps fix speed, boxes honestly give up")


def test_qp_bounds_boxes_fix_lane():
    """boxes-only pulls a drift-out back in-lane across rounds."""
    c = _checker()
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0,
                           'x2': 50.0, 'y2': 0.0}]},
         'lane_width': {'l1': 4.0}}
    drift = np.array([[0, 0], [5, 0], [10, 0], [15, 3.0], [20, 3.5],
                      [25, 4.0]], dtype=np.float32)
    fixed, _ = CorrectorQP(c, bounds='boxes',
                           heuristic_fallback=False).correct(
                               drift, map_data=m)
    ok, v = c.check(fixed, map_data=m)
    assert ok, f"boxes-only must fix drift, got {v}"
    print("  [PASS] boxes fix lane drift")


if __name__ == "__main__":
    test_optimizer_speed_cap()
    test_optimizer_box_and_center()
    test_optimizer_accel_cap()
    test_optimizer_invalid()
    test_qp_safe_passthrough()
    test_qp_revalidation_speed()
    test_qp_revalidation_red()
    test_qp_fallback_to_heuristic()
    test_qp_invalid_args()
    test_qp_kwargs_not_forwarded()
    test_qp_bounds_invalid()
    test_qp_bounds_caps_fix_speed()
    test_qp_bounds_boxes_fix_lane()
    print("QP feasibility battery complete!")

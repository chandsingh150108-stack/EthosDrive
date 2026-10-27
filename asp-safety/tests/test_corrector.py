"""Test TrajectoryCorrector."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from checker import TrajectoryChecker
from corrector import TrajectoryCorrector

_RULES_DIR = str(Path(__file__).parent.parent / "rules")


def test_trajectory_corrector():
    """Test TrajectoryCorrector initialization and correction."""
    print("Testing TrajectoryCorrector...")

    # Initialize components
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker)

    assert corrector.max_iterations == 100
    print("  [PASS] Initialization successful")

    fast_traj = np.array([
        [0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]
    ], dtype=np.float32)

    corrected, corrections = corrector.correct(fast_traj)

    # Verify corrected trajectory is safe
    is_safe, _ = checker.check(corrected)
    assert is_safe, "Corrected trajectory still has violations"

    # Verify corrections were applied
    assert len(corrections) > 0
    assert corrections[0]['rule'] == 'speed_limit'

    print("  [PASS] Speed limit correction successful")
    print(f"    Applied {len(corrections)} corrections")

    safe_traj = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]
    ], dtype=np.float32)

    corrected, corrections = corrector.correct(safe_traj)

    assert len(corrections) == 0
    np.testing.assert_array_equal(corrected, safe_traj)

    print("  [PASS] Safe trajectory returned unchanged")

    # Trajectory that violates both speed and collision
    multi_violation_traj = np.array([
        [0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]
    ], dtype=np.float32)

    other_vehicle = np.array([
        [3, 1], [3.5, 1], [4, 1], [4.5, 1], [5, 1], [5.5, 1]
    ], dtype=np.float32)

    corrected, corrections = corrector.correct(
        multi_violation_traj,
        other_vehicles=[other_vehicle]
    )

    # May need multiple iterations
    assert len(corrections) > 0

    print("  [PASS] Multiple violations handled")
    print("\nAll TrajectoryCorrector tests passed!")


def test_corrector_raises_when_uncorrectable():
    """TG-3: max_iterations=0 on an unsafe input must raise RuntimeError."""
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker, max_iterations=0)
    fast_traj = np.array(
        [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]],
        dtype=np.float32,
    )
    try:
        corrector.correct(fast_traj)
    except RuntimeError:
        print("  [PASS] RuntimeError on uncorrectable input")
    else:
        raise AssertionError("expected RuntimeError with max_iterations=0")


def test_corrector_advisory_passthrough():
    """TG-3: advisory-only findings pass through with no geometry change."""
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker)
    slow = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]],
        dtype=np.float32,
    )
    intent = {
        'maneuver': 'lane_change',
        'maneuver_time': 2,
        'maneuver_direction': 'left',
    }
    ok, v = checker.check(slow, ego_intent=intent)
    # late_signal is a checker-level violation but a corrector-level
    # advisory: it must be the ONLY finding in this scene.
    assert not ok and v, f"expected late_signal finding, got {v}"
    assert all(d['rule'] == 'late_signal' for d in v), f"hard findings: {v}"
    corrected, applied = corrector.correct(slow, ego_intent=intent)
    np.testing.assert_array_equal(corrected, slow)
    assert all(a['correction_type'] == 'advisory_no_change'
               for a in applied), applied
    print("  [PASS] Advisory pass-through leaves trajectory untouched")


def test_corrector_idempotent():
    """TG-3: correcting an already-corrected trajectory is a no-op."""
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker)
    fast_traj = np.array(
        [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]],
        dtype=np.float32,
    )
    corrected, _ = corrector.correct(fast_traj)
    recorrected, applied = corrector.correct(corrected)
    assert applied == [], f"second correction should be empty: {applied}"
    np.testing.assert_array_equal(recorrected, corrected)
    print("  [PASS] Corrector is idempotent")


def test_min_speed_vs_cap_convergence():
    """T-8: min_hwy boost vs speed-cap scale must converge, not oscillate.

    A trajectory violating min_hwy_speed gets boosted 10%; the boost
    must not ping-pong against the speed cap across iterations: the
    fixpoint terminates safe with a monotone applied sequence.
    """
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker)
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 100.0, 'y2': 0.0}]},
         'lane_width': {'l1': 4.0},
         'road_type': {'l1': 'highway'},
         'min_speed_limit': {'l1': 10.0}}
    slow = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = checker.check(slow, map_data=m)
    assert not ok and any(d['rule'] == 'min_hwy_speed' for d in v), \
        f"expected min_hwy_speed, got {v}"
    corrected, applied = corrector.correct(slow, map_data=m)
    ok, v = checker.check(corrected, map_data=m)
    assert ok, f"min-boost must converge safe, got {v}"
    kinds = [a['correction_type'] for a in applied]
    # Boosts and met-target no-ops (stale snapshot visits) only: no
    # scale-down / smoothing ping-pong with other strategies.
    assert kinds and set(kinds) <= {'velocity_boost_10pct',
                                    'min_target_met'}, \
        f"boost must not fight other strategies, got {kinds}"
    print(f"  [PASS] min_hwy boost converges without oscillation "
          f"({len(applied)} steps)")


def test_brake_fallback_holds():
    """P-6: on_failure='brake' holds instead of raising (flag+hold)."""
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker, max_iterations=0,
                                    on_failure='brake')
    fast = np.array(
        [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]],
        dtype=np.float32,
    )
    corrected, applied = corrector.correct(fast)
    kinds = [a['correction_type'] for a in applied]
    assert 'emergency_hold_uncorrectable' in kinds, \
        f"expected flagged hold, got {kinds}"
    # Held from T=1: tail frozen at the start point.
    np.testing.assert_array_equal(corrected[1:], np.zeros((5, 2)))
    print("  [PASS] Brake fallback flags + holds")


def test_group_cap_bounds_churn():
    """P-6: per-group cap stops single-strategy spinning.

    Infeasible band (min 20 > default cap 13.9): boost and scale-down
    ping-pong forever without caps. With group_cap=3 each side applies
    at most 3 times, then the loop honestly raises.
    """
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker, max_iterations=30,
                                    params={'group_cap': 3},
                                    on_failure='brake')
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 200.0, 'y2': 0.0}]},
         'lane_width': {'l1': 4.0},
         'road_type': {'l1': 'highway'},
         'speed_limit': {'l1': 13.4},  # max 13.4 vs min 20: infeasible
         'min_speed_limit': {'l1': 20.0}}
    slow = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    corrected, applied = corrector.correct(slow, map_data=m)
    kinds = [a['correction_type'] for a in applied]
    assert 'emergency_hold_uncorrectable' in kinds, \
        f"infeasible band must end in flagged hold, got {kinds}"
    assert sum(k.startswith('velocity_boost') for k in kinds) <= 3, kinds
    assert sum(k == 'velocity_scaled_to_limit' for k in kinds) <= 3, kinds
    print("  [PASS] Group cap enforced, brake fallback flags hold")


def test_collision_shift_preserves_tail():
    """P-16: lateral shift translates the tail rigidly (no new spikes)."""
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker)
    ego = np.array(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0]], dtype=np.float32)
    other = np.array(
        [[2, 0.5]] * 6, dtype=np.float32)
    corrected, applied = corrector.correct(ego, other_vehicles=[other])
    assert any(a['correction_type'].startswith('lateral_shift')
               for a in applied), f"expected a shift, got {applied}"
    # All downstream segments keep their shape; only the entry changes.
    np.testing.assert_allclose(np.diff(corrected[1:], axis=0),
                               np.diff(ego[1:], axis=0))
    print("  [PASS] Collision shift is tail-rigid")


def test_dual_identity_checker():
    """P-16: checker imported as src.checker is accepted (duck-typed)."""
    import sys as _sys
    repo_root = str(Path(__file__).parent.parent)
    if repo_root not in _sys.path:
        _sys.path.insert(0, repo_root)
    from src.checker import TrajectoryChecker as TC2
    other = TC2(_RULES_DIR)
    TrajectoryCorrector(other)  # must not raise TypeError
    print("  [PASS] Dual-identity checker accepted")


def test_stagnation_breaks_early():
    """PR-1: identical violation sets break to fallback in ~4 checks."""
    checker = TrajectoryChecker(_RULES_DIR)
    corrector = TrajectoryCorrector(checker, max_iterations=100,
                                    params={'group_cap': 0},
                                    on_failure='brake')
    fast = np.array(
        [[0, 0], [20, 0], [40, 0], [60, 0], [80, 0], [100, 0]],
        dtype=np.float32,
    )
    corrected, applied = corrector.correct(fast)
    # Cap 0 = no strategy may apply: sets repeat -> stagnation break
    # after 3 repeats instead of burning 100 iterations.
    assert len(applied) == 1, f"expected single hold entry, got {applied}"
    assert applied[0]['correction_type'] == 'emergency_hold_uncorrectable'
    print("  [PASS] Stagnation breaks early to flagged hold")


if __name__ == "__main__":
    test_trajectory_corrector()
    test_corrector_raises_when_uncorrectable()
    test_corrector_advisory_passthrough()
    test_corrector_idempotent()
    test_min_speed_vs_cap_convergence()
    test_brake_fallback_holds()
    test_group_cap_bounds_churn()
    test_collision_shift_preserves_tail()
    test_dual_identity_checker()
    test_stagnation_breaks_early()

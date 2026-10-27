"""Test MetricsCalculator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from evaluation import MetricsCalculator


def test_collision_rate_full():
    """Collision rate is 100% when a box covers the path."""
    calc = MetricsCalculator()
    traj = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    # Box centered at (1, 0) with width=4, height=4 -> covers x:[-1,3] y:[-2,2]
    box = np.array([[1, 0, 0, 4, 4, 0, 0]], dtype=np.float32)

    rate = calc.collision_rate([traj], [box], [box])

    assert rate == 100.0, f"Expected 100%, got {rate}%"
    print("  [PASS] Collision rate (100%)")


def test_collision_rate_zero():
    """Collision rate is 0% when boxes are far from the path."""
    calc = MetricsCalculator()
    traj = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    box = np.array([[10, 10, 0, 0.5, 0.5, 0, 0]], dtype=np.float32)

    rate = calc.collision_rate([traj], [box], [box])

    assert rate == 0.0, f"Expected 0%, got {rate}%"
    print("  [PASS] Collision rate (0%)")


def test_l2_error():
    """L2 error matches the hand-computed mean displacement."""
    calc = MetricsCalculator()
    pred = np.array([[0, 0], [1, 0], [2, 0]], dtype=np.float32)
    gt = np.array([[0, 0], [1, 1], [2, 0]], dtype=np.float32)

    error = calc.l2_error(pred, gt)

    # mean([0, 1, 0]) = 1/3 = 0.333...
    expected = 1.0 / 3.0
    assert abs(error - expected) < 0.01, f"Expected {expected}, got {error}"
    print("  [PASS] L2 error")


def test_violation_detection_rate():
    """Detection rate is 1.0 when the finding matches exactly."""
    calc = MetricsCalculator()
    true_v = [{'rule': 'speed_limit', 'time': 2, 'vehicle': 'ego'}]
    detected_v = [{'rule': 'speed_limit', 'time': 2, 'vehicle': 'ego'}]

    rate = calc.violation_detection_rate(true_v, detected_v)

    assert rate == 1.0, f"Expected 1.0, got {rate}"
    print("  [PASS] Violation detection rate (100%)")


def test_false_positive_rate():
    """False positive rate is 1.0 when truth is empty but one fired."""
    calc = MetricsCalculator()
    true_v = []
    detected_v = [{'rule': 'speed_limit', 'time': 2, 'vehicle': 'ego'}]

    rate = calc.false_positive_rate(true_v, detected_v)

    assert rate == 1.0, f"Expected 1.0, got {rate}"
    print("  [PASS] False positive rate")


def test_average_inference_time():
    """Average inference time is the plain mean."""
    calc = MetricsCalculator()
    times = [10.0, 20.0, 30.0]

    avg = calc.average_inference_time(times)

    assert avg == 20.0, f"Expected 20.0, got {avg}"
    print("  [PASS] Average inference time")


def test_real_output_metrics():
    """T-10: metrics consume real check() outputs (vehicle2/type keys)."""
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent.parent / "src"))
    from checker import TrajectoryChecker
    calc = MetricsCalculator()
    checker = TrajectoryChecker(str(_Path(__file__).parent.parent / "rules"))

    ego = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    other = np.array([[2, 0.5]] * 6, dtype=np.float32)
    _, real = checker.check(ego, other_vehicles=[other])
    assert real, "overlap must produce real findings"
    assert any('vehicle2' in d for d in real), \
        f"expected arity-4 dicts, got {real}"

    # Identity: detection 1.0, false-positive 0.0 on real shapes.
    assert calc.violation_detection_rate(real, real) == 1.0
    assert calc.false_positive_rate(real, real) == 0.0
    assert calc.violation_detection_rate([], real) == 0.0

    # Collision rate against a real planner box on the path.
    from neural_planner import SparseDrivePlanner
    planner = SparseDrivePlanner("nonexistent_weights.pth", device="cpu")
    out = planner.predict(
        np.zeros((6, 3, 256, 704), dtype=np.float32),
        np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0),
        np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0),
        np.eye(4, dtype=np.float32), 0.0)
    hit = np.array([[0, 0], [5, 0], [10, 0], [15, 0], [20, 0], [25, 0]],
                   dtype=np.float32)
    rate = calc.collision_rate([hit], [out['boxes']], [out['boxes']])
    assert rate == 100.0, f"box at (20,0) must be hit, got {rate}"
    print("  [PASS] Metrics consume real checker/planner outputs (T-10)")


def test_yawed_box():
    """P-17: footprint rotates with yaw (90 deg swaps extents)."""
    calc = MetricsCalculator()
    traj = np.array([[1.5, 0.0]], dtype=np.float32)
    # l=4 (x), w=2 (y): (1.5, 0) inside axis-aligned...
    box0 = np.array([[0, 0, 0, 2, 4, 0, 0.0]], dtype=np.float32)
    assert calc.collision_rate([traj], [box0], [box0]) == 100.0
    # ...but outside after a 90-degree yaw (x-extent becomes 1).
    import math
    box90 = np.array([[0, 0, 0, 2, 4, 0, math.pi / 2]], dtype=np.float32)
    assert calc.collision_rate([traj], [box90], [box90]) == 0.0
    print("  [PASS] Yaw-aware footprints")


def test_batch_mismatch_raises():
    """P-17: mismatched batch lengths raise instead of truncating."""
    calc = MetricsCalculator()
    traj = np.array([[0, 0]], dtype=np.float32)
    box = np.array([[0, 0, 0, 2, 2, 0, 0]], dtype=np.float32)
    try:
        calc.collision_rate([traj, traj], [box], [box])
    except ValueError:
        print("  [PASS] Batch mismatch raises")
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_l2_validation():
    """P-17: l2_error rejects bad shapes and NaN/inf."""
    calc = MetricsCalculator()
    good = np.array([[0, 0], [1, 0]], dtype=np.float32)
    for bad in [np.zeros((2, 3), dtype=np.float32),
                np.zeros((2,), dtype=np.float32)]:
        try:
            calc.l2_error(bad, good)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {bad.shape}")
    nan = np.array([[np.nan, 0.0]], dtype=np.float32)
    try:
        calc.l2_error(nan, good)
    except ValueError:
        print("  [PASS] l2_error validates shape + finiteness")
        return
    raise AssertionError("expected ValueError on NaN")


def test_matching_keys_and_fdr():
    """P-17: vehicle-aware matching + false_discovery_rate name."""
    calc = MetricsCalculator()
    assert calc.false_discovery_rate([], [{'rule': 'x', 'time': 0}]) == 1.0
    # Same rule+time, different vehicle2: no match.
    tv = [{'rule': 'collision_avoidance', 'vehicle': 'ego',
           'vehicle2': 'vehicle_0', 'time': 2}]
    dv = [{'rule': 'collision_avoidance', 'vehicle': 'ego',
           'vehicle2': 'vehicle_1', 'time': 2}]
    assert calc.violation_detection_rate(tv, dv) == 0.0
    assert calc.false_discovery_rate(tv, dv) == 1.0
    # Legacy dicts without vehicle keys still match by rule+time.
    assert calc.violation_detection_rate(
        [{'rule': 'speed_limit', 'time': 2}],
        [{'rule': 'speed_limit', 'time': 2, 'vehicle': 'ego'}]) == 1.0
    # Old name still works.
    assert calc.false_positive_rate(tv, dv) == 1.0
    print("  [PASS] Key-aware matching + FDR name")


if __name__ == "__main__":
    test_collision_rate_full()
    test_collision_rate_zero()
    test_l2_error()
    test_violation_detection_rate()
    test_false_positive_rate()
    test_average_inference_time()
    test_real_output_metrics()
    test_yawed_box()
    test_batch_mismatch_raises()
    test_l2_validation()
    test_matching_keys_and_fdr()
    print("\nAll MetricsCalculator tests passed!")

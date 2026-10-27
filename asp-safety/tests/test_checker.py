"""Unit tests for ASP trajectory checker."""

import sys
import os
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from checker import TrajectoryChecker


def test_trajectory_checker():
    """Test basic trajectory checking functionality."""
    print("Testing TrajectoryChecker...")
    
    # Initialize checker with rules directory
    rules_dir = str(Path(__file__).parent.parent / "rules")
    checker = TrajectoryChecker(rules_dir)
    
    # Create a simple test trajectory (straight line)
    trajectory = np.array([
        [0.0, 0.0],   # t=0
        [1.0, 0.0],   # t=1
        [2.0, 0.0],   # t=2
        [3.0, 0.0],   # t=3
        [4.0, 0.0],   # t=4
        [5.0, 0.0],   # t=5
    ])
    
    # Check trajectory (without map data - should be safe)
    is_safe, violations = checker.check(trajectory)
    
    print(f"  Trajectory shape: {trajectory.shape}")
    print(f"  Is safe: {is_safe}")
    print(f"  Violations: {violations}")

    # T-1: slow straight line must be safe (fail loudly, not just print).
    assert is_safe is True, f"slow trajectory flagged unsafe: {violations}"
    assert violations == [], f"expected no violations, got {violations}"
    
    # Test with a violation scenario (speed limit)
    # Create a trajectory that exceeds speed limit
    fast_trajectory = np.array([
        [0.0, 0.0],
        [20.0, 0.0],  # Very fast
        [40.0, 0.0],  # Very fast
        [60.0, 0.0],  # Very fast
        [80.0, 0.0],  # Very fast
        [100.0, 0.0], # Very fast
    ])
    
    is_safe_fast, violations_fast = checker.check(fast_trajectory)
    
    print(f"\n  Fast trajectory test:")
    print(f"  Is safe: {is_safe_fast}")
    print(f"  Violations found: {len(violations_fast)}")

    # T-1: fast trajectory must be unsafe with a speed finding.
    assert is_safe_fast is False, "fast trajectory passed as safe"
    assert any(v["rule"] == "speed_limit" for v in violations_fast), \
        f"expected speed_limit finding, got {violations_fast}"
    
    print("\n✓ TrajectoryChecker tests completed!")


def test_checker_is_not_vacuous():
    """T-4 regression guard: the checker must fire on known-bad inputs.

    Catches the silent-safe incident where everything (overlaps, red
    lights, speeding) reported safe. Fails loudly under pytest.
    """
    rules_dir = str(Path(__file__).parent.parent / "rules")
    checker = TrajectoryChecker(rules_dir)

    ego = np.array(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0],
         [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        dtype=np.float32,
    )

    # 1. Speeding must yield a speed finding.
    fast = np.array(
        [[0.0, 0.0], [20.0, 0.0], [40.0, 0.0],
         [60.0, 0.0], [80.0, 0.0], [100.0, 0.0]],
        dtype=np.float32,
    )
    ok, v = checker.check(fast)
    assert ok is False and len(v) >= 1, \
        f"vacuous checker: fast traj safe, violations={v}"

    # 2. Overlap must yield a collision finding.
    other = np.array([[2.0, 0.0]] * 6, dtype=np.float32)
    ok, v = checker.check(ego, other_vehicles=[other])
    assert ok is False, f"vacuous checker: overlap safe, violations={v}"
    assert any(d["rule"] == "collision_avoidance" for d in v), \
        f"expected collision_avoidance, got {v}"

    # 3. Running a red must yield red_light.
    ok, v = checker.check(
        ego,
        traffic_lights={
            "i1": {"color": "red", "times": [2],
                   "x": 2.0, "y": 0.0, "radius": 10.0}
        },
    )
    assert ok is False, f"vacuous checker: red run safe, violations={v}"
    assert any(d["rule"] == "red_light" for d in v), \
        f"expected red_light, got {v}"

    print("✓ checker-is-not-vacuous guard passed!")


def test_correction():
    """Test trajectory correction functionality."""
    print("\nTesting trajectory correction...")
    
    rules_dir = str(Path(__file__).parent.parent / "rules")
    checker = TrajectoryChecker(rules_dir)
    
    # Create a trajectory that violates speed limit
    trajectory = np.array([
        [0.0, 0.0],
        [20.0, 0.0],
        [40.0, 0.0],
        [60.0, 0.0],
        [80.0, 0.0],
        [100.0, 0.0],
    ])
    
    # Get corrected trajectory
    corrected = checker.get_safe_trajectory(trajectory)
    
    print(f"  Original trajectory end point: {trajectory[-1]}")
    print(f"  Corrected trajectory end point: {corrected[-1]}")

    # CR-4: delegation path must yield a genuinely safe trajectory.
    ok, remaining = checker.check(corrected)
    assert ok is True, f"corrected trajectory still unsafe: {remaining}"
    
    print("\n✓ Trajectory correction tests completed!")


if __name__ == "__main__":
    print("=" * 50)
    print("ASP Safety Layer - Unit Tests")
    print("=" * 50)
    
    try:
        test_trajectory_checker()
        test_checker_is_not_vacuous()
        test_correction()
        
        print("\n" + "=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

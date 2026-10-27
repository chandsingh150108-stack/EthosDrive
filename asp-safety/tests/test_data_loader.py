"""Test DataLoader."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from data_loader import NuScenesLoader


def test_placeholder_init():
    """Placeholder mode activates on a nonexistent path."""
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    assert loader.use_placeholder is True
    print("  [PASS] Initialization with placeholder successful")


def test_scene_sample():
    """Scene sample carries the exact shapes the planner validates."""
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    sample = loader.get_scene_sample(0)

    assert 'images' in sample
    assert 'intrinsics' in sample
    assert 'extrinsics' in sample
    assert 'ego2global' in sample
    assert 'timestamp' in sample

    assert sample['images'].shape == (6, 3, 256, 704)
    assert sample['intrinsics'].shape == (6, 3, 3)
    assert sample['extrinsics'].shape == (6, 4, 4)
    assert sample['ego2global'].shape == (4, 4)
    assert isinstance(sample['timestamp'], float)

    print("  [PASS] Scene sample loaded correctly")
    print(f"    Images shape: {sample['images'].shape}")
    print(f"    Timestamp: {sample['timestamp']}")


def test_trajectory():
    """Placeholder trajectory has shape (<=6, 2) and float32 dtype."""
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    trajectory = loader.get_trajectory(0)

    assert trajectory.shape[1] == 2
    assert trajectory.shape[0] <= 6
    assert trajectory.dtype == np.float32

    print("  [PASS] Trajectory loaded correctly")
    print(f"    Shape: {trajectory.shape}")


def test_map_data():
    """Placeholder map data carries lanes and intersections."""
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    map_data = loader.get_map_data(0)

    assert 'lanes' in map_data
    assert 'intersections' in map_data

    print("  [PASS] Map data loaded correctly")


def test_traffic_lights():
    """Placeholder traffic lights carry the int_0 color state."""
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    traffic_lights = loader.get_traffic_lights(0)

    assert 'int_0' in traffic_lights
    assert 'color' in traffic_lights['int_0']

    print("  [PASS] Traffic lights loaded correctly")


def test_invalid_scene_index():
    """Out-of-range scene index raises IndexError."""
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    try:
        loader.get_scene_sample(100)
    except IndexError:
        print("  [PASS] IndexError raised for invalid scene_idx")
        return
    raise AssertionError("expected IndexError for scene_idx=100")


def test_placeholder_scene_determinism():
    """T-12: per-scene placeholder seeds give identical samples/trajectories."""
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")

    s1 = loader.get_scene_sample(0)
    s2 = loader.get_scene_sample(0)
    np.testing.assert_array_equal(s1['images'], s2['images'])
    np.testing.assert_array_equal(s1['intrinsics'], s2['intrinsics'])
    assert s1['timestamp'] == s2['timestamp']

    t1 = loader.get_trajectory(0)
    t2 = loader.get_trajectory(0)
    np.testing.assert_array_equal(t1, t2)
    print("  [PASS] Placeholder scene determinism (T-12)")


def test_reachable_red():
    """T-9: placeholder red intersection is reachable and fixable."""
    from checker import TrajectoryChecker
    from corrector import TrajectoryCorrector
    from neural_planner import SparseDrivePlanner
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    planner = SparseDrivePlanner("nonexistent_weights.pth", device="cpu")
    images = np.zeros((6, 3, 256, 704), dtype=np.float32)
    eye3 = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    eye4 = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    traj = planner.predict(images, eye3, eye4, eye4[0], 0.0)['trajectory']

    checker = TrajectoryChecker(
        str(Path(__file__).parent.parent / "rules"))
    ok, v = checker.check(
        traj, map_data=loader.get_map_data(0),
        traffic_lights=loader.get_traffic_lights(0))
    assert not ok and any(d['rule'] == 'red_light' for d in v), \
        f"placeholder path must run the reachable red, got {v}"
    corrector = TrajectoryCorrector(checker)
    fixed, _ = corrector.correct(
        traj, map_data=loader.get_map_data(0),
        traffic_lights=loader.get_traffic_lights(0))
    ok, v = checker.check(
        fixed, map_data=loader.get_map_data(0),
        traffic_lights=loader.get_traffic_lights(0))
    assert ok, f"red hold must verify clean, got {v}"
    print("  [PASS] Reachable red fires + corrects (T-9)")


def test_hostile_scene_9():
    """T-9/P-14: scene 9 carries an obstacle on the path (fixable)."""
    from checker import TrajectoryChecker
    from corrector import TrajectoryCorrector
    from neural_planner import SparseDrivePlanner
    loader = NuScenesLoader("nonexistent_path", version="v1.0-mini")
    m9 = loader.get_map_data(9)
    assert 'fixed_obstacles' in m9, "scene 9 must plant an obstacle"
    planner = SparseDrivePlanner("nonexistent_weights.pth", device="cpu")
    images = np.zeros((6, 3, 256, 704), dtype=np.float32)
    eye3 = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    eye4 = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    traj = planner.predict(images, eye3, eye4, eye4[0], 0.0)['trajectory']

    checker = TrajectoryChecker(
        str(Path(__file__).parent.parent / "rules"))
    ok, v = checker.check(
        traj, map_data=m9,
        traffic_lights=loader.get_traffic_lights(9))
    rules = set(d['rule'] for d in v)
    assert not ok and 'fixed_obstacle_contact' in rules, \
        f"scene 9 must hit the planted obstacle, got {v}"
    corrector = TrajectoryCorrector(checker)
    fixed, _ = corrector.correct(
        traj, map_data=m9,
        traffic_lights=loader.get_traffic_lights(9))
    ok, v = checker.check(
        fixed, map_data=m9,
        traffic_lights=loader.get_traffic_lights(9))
    assert ok, f"hostile scene must correct clean, got {v}"
    print("  [PASS] Hostile scene 9 fires + corrects (T-9/P-14)")


if __name__ == "__main__":
    test_placeholder_init()
    test_scene_sample()
    test_trajectory()
    test_map_data()
    test_traffic_lights()
    test_invalid_scene_index()
    test_placeholder_scene_determinism()
    test_reachable_red()
    test_hostile_scene_9()
    print("\nAll DataLoader tests passed!")

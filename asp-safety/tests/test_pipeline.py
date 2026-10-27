"""Test SafetyLayerPipeline."""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_TESTS_DIR.parent / "src"))

import numpy as np
from pipeline import SafetyLayerPipeline
from data_loader import NuScenesLoader

_ASP_SAFETY_DIR = _TESTS_DIR.parent
_PLANNER_PATH = str(_ASP_SAFETY_DIR / "SparseDrive/ckpt/sparsedrive_stage2.pth")
_RULES_DIR = str(_ASP_SAFETY_DIR / "rules")


def _synthetic_inputs():
    images = np.random.randn(6, 3, 256, 704).astype(np.float32)
    intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    ego2global = np.eye(4, dtype=np.float32)
    return images, intrinsics, extrinsics, ego2global, 1234567890.0


def test_pipeline_run():
    """Single-frame run returns the full result dict with valid shapes."""
    pipeline = SafetyLayerPipeline(
        planner_path=_PLANNER_PATH,
        rules_dir=_RULES_DIR,
        device="cpu"
    )
    print("  [PASS] Initialization successful")

    result = pipeline.run(*_synthetic_inputs())

    # Verify result structure
    assert 'original_trajectory' in result
    assert 'safe_trajectory' in result
    assert 'horizon_s' in result
    assert 'violations' in result
    assert 'is_safe' in result
    assert 'inference_time_ms' in result
    assert 'boxes' in result
    assert 'scores' in result

    # Verify shapes: native (6,2) proposal, resampled (4,2) checked grid
    assert result['original_trajectory'].shape == (6, 2)
    assert result['safe_trajectory'].shape == (4, 2)
    assert result['horizon_s'] == 3
    assert isinstance(result['is_safe'], bool)
    assert result['inference_time_ms'] > 0

    # P-18 per-stage timing accounts for the total (within 1 ms).
    stages = result['timing_ms']
    assert set(stages) == {'planner', 'checker', 'corrector'}
    assert all(v >= 0 for v in stages.values())
    assert abs(sum(stages.values()) - result['pipeline_time_ms']) < 1.0

    print("  [PASS] Pipeline run successful")
    print(f"    Original trajectory shape: {result['original_trajectory'].shape}")
    print(f"    Safe trajectory shape: {result['safe_trajectory'].shape}")
    print(f"    Is safe: {result['is_safe']}")
    print(f"    Inference time: {result['inference_time_ms']:.2f} ms")


def test_pipeline_batch():
    """Batch run over 3 placeholder scenes returns tagged results."""
    pipeline = SafetyLayerPipeline(
        planner_path=_PLANNER_PATH,
        rules_dir=_RULES_DIR,
        device="cpu"
    )
    loader = NuScenesLoader("nonexistent_path")
    results = pipeline.run_batch(loader, [0, 1, 2])

    assert len(results) == 3
    assert all('scene_idx' in r for r in results)

    print("  [PASS] Batch run successful")

def test_pipeline_red_light_e2e():
    """GAP-12: planted red light forces unsafe + verified-clean repair.

    Geometry-agnostic: the lane is built from the planner's own CHECKED
    waypoints (T=0 anchor + true +1/+2/+3 s, so in_lane holds under
    real or placeholder inference), with the red intersection planted
    exactly on the T=3 waypoint.
    """
    from checker import TrajectoryChecker
    pipeline = SafetyLayerPipeline(
        planner_path=_PLANNER_PATH,
        rules_dir=_RULES_DIR,
        device="cpu"
    )
    images, intrinsics, extrinsics, ego2global, timestamp = _synthetic_inputs()
    probe = pipeline.run(images, intrinsics, extrinsics, ego2global, timestamp)
    assert probe['safe_trajectory'].shape == (4, 2)
    waypoints = probe['safe_trajectory'].astype(float)

    # Lane segments joint-to-joint along the checked path, with short
    # extensions past both ends so every waypoint sits on the centerline.
    segments = []
    for i in range(len(waypoints) - 1):
        a, b = waypoints[i], waypoints[i + 1]
        if np.linalg.norm(b - a) > 1e-6:
            segments.append({'x1': float(a[0]), 'y1': float(a[1]),
                             'x2': float(b[0]), 'y2': float(b[1])})
    assert segments, "probe trajectory is degenerate"
    d0 = waypoints[1] - waypoints[0]
    d0 = d0 / np.linalg.norm(d0) if np.linalg.norm(d0) > 1e-6 else np.array([1.0, 0.0])
    d1 = waypoints[-1] - waypoints[-2]
    d1 = d1 / np.linalg.norm(d1) if np.linalg.norm(d1) > 1e-6 else np.array([1.0, 0.0])
    segments.insert(0, {'x1': float(waypoints[0][0] - 4 * d0[0]),
                        'y1': float(waypoints[0][1] - 4 * d0[1]),
                        'x2': float(waypoints[0][0]), 'y2': float(waypoints[0][1])})
    segments.append({'x1': float(waypoints[-1][0]), 'y1': float(waypoints[-1][1]),
                     'x2': float(waypoints[-1][0] + 4 * d1[0]),
                     'y2': float(waypoints[-1][1] + 4 * d1[1])})

    wx, wy = float(waypoints[3][0]), float(waypoints[3][1])
    lights = {
        'i1': {'color': 'red', 'times': list(range(6)),
               'x': wx, 'y': wy, 'radius': 0.5}
    }
    map_data = {
        'lanes': {'l1': segments},
        'lane_width': {'l1': 4.0},
        'controls_approach': {'i1': 'l1'},
    }
    result = pipeline.run(images, intrinsics, extrinsics, ego2global,
                          timestamp, map_data=map_data,
                          traffic_lights=lights)
    assert result['is_safe'] is False, "red-light run must be unsafe"
    assert any(v['rule'] == 'red_light' for v in result['violations']), \
        f"expected red_light, got {result['violations']}"
    checker = TrajectoryChecker(_RULES_DIR)
    ok, v = checker.check(result['safe_trajectory'], map_data=map_data,
                          traffic_lights=lights)
    assert ok, f"pipeline safe_trajectory must verify clean, got {v}"
    print("  [PASS] E2E red-light violation corrected by pipeline")


def test_detections_to_trajectories():
    """P-18: score filter + cap + CV projection, no model needed."""
    from pipeline import SafetyLayerPipeline as P
    boxes = np.array([[10.0, 0.0, 0.5, 2.0, 4.5, 1.5, 0.0],
                      [20.0, 0.0, 0.5, 2.0, 4.5, 1.5, 0.0],
                      [30.0, 0.0, 0.5, 2.0, 4.5, 1.5, 0.0]], dtype=np.float32)
    vels = np.array([[2.0, 0.0, 0.0],
                     [0.0, 0.0, 0.0],
                     [1.0, 1.0, 0.0]], dtype=np.float32)
    scores = np.array([0.9, 0.1, 0.8], dtype=np.float32)

    trajs = P._detections_to_trajectories(boxes, vels, scores, 6)
    # Score filter drops box 1; order irrelevant here.
    assert len(trajs) == 2, f"expected 2 kept, got {len(trajs)}"
    # CV projection: box 0 moves +2 m/step in x.
    np.testing.assert_allclose(trajs[0][:, 0], [10, 12, 14, 16, 18, 20])
    np.testing.assert_allclose(trajs[0][:, 1], np.zeros(6))
    # dt=0.5 halves per-step displacement (native 0.5 s grid).
    half = P._detections_to_trajectories(boxes, vels, scores, 6, dt=0.5)
    np.testing.assert_allclose(half[0][:, 0], [10, 11, 12, 13, 14, 15])

    capped = P._detections_to_trajectories(
        boxes, vels, scores, 6, max_agents=1)
    assert len(capped) == 1

    assert P._detections_to_trajectories(None, None, None, 6) == []
    assert P._detections_to_trajectories(
        np.zeros((0, 7), dtype=np.float32),
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((0,), dtype=np.float32), 6) == []
    print("  [PASS] Detections project, filter, and cap correctly")


def test_run_batch_error_schema():
    """T-11: per-scene failures become stable error dicts, not raises."""
    from pipeline import SafetyLayerPipeline as P
    pipeline = P(planner_path="nonexistent_weights.pth",
                 rules_dir=_RULES_DIR, device="cpu")

    def _boom(*args, **kwargs):
        raise ValueError("synthetic planner failure")

    pipeline.planner.predict = _boom
    loader = NuScenesLoader("nonexistent_path")
    results = pipeline.run_batch(loader, [0])
    assert len(results) == 1
    err = results[0]
    assert err['scene_idx'] == 0
    assert err['error'] == "synthetic planner failure"
    assert err['is_safe'] is False
    assert err['violations'] == []
    assert err['original_trajectory'] is None
    assert err['safe_trajectory'] is None
    assert err['timing_ms'] is None
    assert err['boxes'] is None and err['scores'] is None
    print("  [PASS] Batch error schema stable (T-11)")


def test_corrector_raise_propagates():
    """T-11: corrector RuntimeError propagates through run() (pinned)."""
    from pipeline import SafetyLayerPipeline as P
    from corrector import TrajectoryCorrector
    pipeline = P(planner_path="nonexistent_weights.pth",
                 rules_dir=_RULES_DIR, device="cpu")
    pipeline.corrector = TrajectoryCorrector(pipeline.checker,
                                             max_iterations=0)
    loader = NuScenesLoader("nonexistent_path")
    sample = loader.get_scene_sample(9)  # hostile scene: must correct
    try:
        pipeline.run(sample['images'].astype(np.float32),
                     sample['intrinsics'], sample['extrinsics'],
                     sample['ego2global'], sample['timestamp'],
                     map_data=loader.get_map_data(9),
                     traffic_lights=loader.get_traffic_lights(9))
    except RuntimeError:
        print("  [PASS] Corrector RuntimeError propagates (T-11)")
        return
    raise AssertionError("expected RuntimeError with max_iterations=0")


def test_pipeline_corrector_modes():
    """corrector_mode selects the engine and stamps the result."""
    from corrector import TrajectoryCorrector, CorrectorQP
    heur = SafetyLayerPipeline(
        planner_path=_PLANNER_PATH, rules_dir=_RULES_DIR, device="cpu")
    assert heur.corrector_mode == 'heuristic'
    assert isinstance(heur.corrector, TrajectoryCorrector)
    qp = SafetyLayerPipeline(
        planner_path=_PLANNER_PATH, rules_dir=_RULES_DIR, device="cpu",
        corrector_mode='qp')
    assert isinstance(qp.corrector, CorrectorQP)
    for pipe, mode in ((heur, 'heuristic'), (qp, 'qp')):
        result = pipe.run(*_synthetic_inputs())
        assert result['corrector_mode'] == mode, \
            f"expected stamp {mode}, got {result.get('corrector_mode')}"
        assert result['safe_trajectory'].shape == (4, 2)
    try:
        SafetyLayerPipeline(
            planner_path=_PLANNER_PATH, rules_dir=_RULES_DIR,
            device="cpu", corrector_mode='mysterious')
        raise AssertionError("bad corrector_mode must raise")
    except ValueError:
        pass
    print("  [PASS] corrector_mode selects engine + stamps result")


if __name__ == "__main__":
    test_detections_to_trajectories()
    test_pipeline_run()
    test_pipeline_batch()
    test_pipeline_red_light_e2e()
    test_run_batch_error_schema()
    test_corrector_raise_propagates()
    test_pipeline_corrector_modes()
    print("\nAll Pipeline tests passed!")

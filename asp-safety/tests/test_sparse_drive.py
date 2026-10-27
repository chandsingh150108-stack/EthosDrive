"""Test SparseDrive integration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neural_planner import SparseDrivePlanner
import numpy as np

_MODEL_PATH = str(Path(__file__).parent.parent / "SparseDrive/ckpt/sparsedrive_stage2.pth")

def test_sparse_drive_planner():
    """Test SparseDrivePlanner initialization and prediction."""
    print("Testing SparseDrivePlanner...")

    planner = SparseDrivePlanner(
        model_path=_MODEL_PATH,
        device="cpu"  # Use CPU for testing
    )
    print("  [PASS] Initialization successful")

    images = np.random.randn(6, 3, 256, 704).astype(np.float32)
    intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    ego2global = np.eye(4, dtype=np.float32)
    timestamp = 1234567890.0

    result = planner.predict(images, intrinsics, extrinsics, ego2global, timestamp)

    assert 'trajectory' in result
    assert 'boxes' in result
    assert 'scores' in result

    assert result['trajectory'].shape == (6, 2)
    assert result['boxes'].shape[1] == 7
    assert result['scores'].ndim == 1

    print("  [PASS] Prediction successful")
    print(f"    Trajectory shape: {result['trajectory'].shape}")
    print(f"    Boxes shape: {result['boxes'].shape}")
    print(f"    Scores shape: {result['scores'].shape}")

    wrong_images = np.random.randn(3, 3, 256, 704).astype(np.float32)
    try:
        planner.predict(wrong_images, intrinsics, extrinsics, ego2global, timestamp)
    except ValueError:
        print("  [PASS] Correctly raised ValueError for wrong shape")
        print("\nAll SparseDrive tests passed!")
        return
    raise AssertionError("expected ValueError for wrong image shape")


def test_build_projection_mat():
    """RQ-2: K*[R|t] embedded in 4x4; rejects bad shapes."""
    K = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    K[:, 0, 0] = 400.0
    K[:, 1, 1] = 400.0
    E = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    E[:, 0, 3] = 1.5  # camera 1.5 m ahead of lidar origin

    proj = SparseDrivePlanner._build_projection_mat(K, E)
    assert proj.shape == (6, 4, 4), proj.shape
    # K @ [R|t]: top-left 2x2 carries focal*rotation, translation column
    # carries K @ t = [400*1.5, 0, 1.5].
    np.testing.assert_allclose(proj[0, :3, :4], K[0] @ E[0][:3, :], rtol=1e-5)
    np.testing.assert_allclose(proj[0, 3, :], [0, 0, 0, 1])
    assert proj[0, 0, 3] == 400.0 * 1.5

    for bad_K, bad_E in [
        (np.zeros((6, 3), dtype=np.float32), E),
        (K, np.zeros((6, 3, 3), dtype=np.float32)),
    ]:
        try:
            SparseDrivePlanner._build_projection_mat(bad_K, bad_E)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for bad calib shapes")
    print("  [PASS] Projection matrix construction (RQ-2)")


def test_postprocess_golden():
    """T-5: real _postprocess path on mocked model output (no torch).

    Golden values pin the contract the GPU run must reproduce; any
    drift in the model output schema fails loudly here first.
    """
    planner = SparseDrivePlanner(model_path=_MODEL_PATH, device="cpu")

    golden_traj = np.array(
        [[2.5, 0.0], [5.0, 0.1], [7.5, 0.15],
         [10.0, 0.1], [12.5, 0.1], [15.0, 0.15]],
        dtype=np.float64,
    )
    golden_boxes = np.array(
        [[20.0, 0.0, 0.5, 2.0, 4.5, 1.5, 0.0, 0.5, 0.0, 0.1]],
        dtype=np.float64,
    )
    golden_scores = np.array([0.95], dtype=np.float64)
    raw = [{'img_bbox': {
        'final_planning': golden_traj,
        'boxes_3d': golden_boxes,
        'scores_3d': golden_scores,
    }}]

    out = planner._postprocess(raw)
    np.testing.assert_allclose(out['trajectory'], golden_traj.astype(np.float32))
    # Decoded 10-dim boxes are sliced to the 7-dim pipeline contract.
    np.testing.assert_allclose(
        out['boxes'], golden_boxes[:, :7].astype(np.float32))
    np.testing.assert_allclose(out['scores'], golden_scores.astype(np.float32))
    # Planar velocities ride along separately (P-18 agent trajectories).
    np.testing.assert_allclose(
        out['velocities'], golden_boxes[:, 7:10].astype(np.float32))
    assert out['trajectory'].dtype == np.float32
    assert out['boxes'].shape == (1, 7)
    print("  [PASS] Postprocess golden values (T-5)")


def test_postprocess_fail_loud():
    """P-13: missing/misshapen model outputs raise, never placeholder."""
    planner = SparseDrivePlanner(model_path=_MODEL_PATH, device="cpu")

    for bad_raw, exc in [
        ([], KeyError),                                   # empty output
        ([{}], KeyError),                                 # no img_bbox
        ([{'img_bbox': {}}], KeyError),                   # no planning
        ([{'img_bbox': {'final_planning': np.zeros((5, 2))}}],
         ValueError),                                     # wrong traj shape
        ([{'img_bbox': {
            'final_planning': np.zeros((6, 2)),
            'boxes_3d': np.zeros((2, 7)),
            'scores_3d': np.zeros((3,))}}], ValueError),  # score mismatch
    ]:
        try:
            planner._postprocess(bad_raw)
        except exc:
            continue
        raise AssertionError(f"expected {exc} for {bad_raw}")

    # Empty detections stay legal.
    out = planner._postprocess([{'img_bbox': {
        'final_planning': np.zeros((6, 2))}}])
    assert out['boxes'].shape == (0, 7) and out['scores'].shape == (0,)
    print("  [PASS] Postprocess fail-loud (P-13)")


def test_step_dt_contract():
    """P-1: pin the canonical timestep every threshold is derived for.

    STEP_DT=1.0 s. The 50 km/h default cap in speed_regulation.lp
    (1929321 = round(13.8889*100)^2) and the R3.1 2 s headway
    (D^2 < 4*S2) both assume 1 s steps. If the real 0.5 s model output
    ever flows in un-resampled, this test forces the rescaling work
    instead of silently doubling every limit.
    """
    from checker import STEP_DT, SCALE_FACTOR
    assert STEP_DT == 1.0, f"STEP_DT changed to {STEP_DT}: re-derive thresholds"

    cap_ms = 50.0 / 3.6
    cap_sq = round(cap_ms * SCALE_FACTOR) ** 2
    assert cap_sq == 1929321, (
        f"50 km/h squared cap drifted to {cap_sq}: "
        "speed_regulation.lp default is stale"
    )

    # Placeholder waypoints are 2.5 m apart: at STEP_DT that is a
    # 2.5 m/s cruise, comfortably under the 12.5 m/s corrector target.
    planner = SparseDrivePlanner(model_path=_MODEL_PATH, device="cpu")
    out = planner._generate_placeholder()
    steps = np.linalg.norm(np.diff(out['trajectory'], axis=0), axis=1)
    assert np.all(steps < 12.5 * STEP_DT), (
        f"placeholder implies >12.5 m/s at STEP_DT: {steps}"
    )
    print("  [PASS] STEP_DT contract pinned (P-1)")


def test_placeholder_determinism():
    """T-12: seed-42 placeholder predict is bit-identical across calls."""
    planner = SparseDrivePlanner(model_path="nonexistent_weights.pth", device="cpu")
    assert planner.use_placeholder, "bogus path should force placeholder"

    def _inputs():
        return (
            np.random.randn(6, 3, 256, 704).astype(np.float32),
            np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0),
            np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0),
            np.eye(4, dtype=np.float32),
            1234567890.0,
        )

    out1 = planner.predict(*_inputs())
    out2 = planner.predict(*_inputs())
    np.testing.assert_array_equal(out1['trajectory'], out2['trajectory'])
    np.testing.assert_array_equal(out1['boxes'], out2['boxes'])
    np.testing.assert_array_equal(out1['scores'], out2['scores'])
    print("  [PASS] Placeholder determinism (T-12)")


def test_temporal_api_placeholder():
    """N-2: sequence API works without a model (placeholder path)."""
    planner = SparseDrivePlanner(model_path="nonexistent_weights.pth", device="cpu")
    planner.begin_sequence()
    images = np.random.randn(6, 3, 256, 704).astype(np.float32)
    intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
    ego2global = np.eye(4, dtype=np.float32)
    for _ in range(3):
        out = planner.predict_frame(images, intrinsics, extrinsics,
                                     ego2global, 1234567890.0)
        assert out['trajectory'].shape == (6, 2)
    planner.end_sequence()
    print("  [PASS] Temporal sequence API (N-2)")


def test_estimate_ego_status():
    """N-2: pose-derived status puts planar velocity at indices 6,7."""
    from data_loader import estimate_ego_status
    prev = np.eye(4, dtype=np.float32)
    cur = np.eye(4, dtype=np.float32)
    cur[0, 3] = 5.0  # 5 m forward
    cur[1, 3] = 1.0  # 1 m left
    status = estimate_ego_status(prev, cur, dt=1.0)
    assert status.shape == (18,)
    assert abs(status[6] - 5.0) < 1e-5, status[6]
    assert abs(status[7] - 1.0) < 1e-5, status[7]
    assert np.all(status[[0, 1, 2, 3, 4, 5, 8, 9]] == 0.0)
    # Degenerate dt yields zeros, never NaN.
    assert np.all(estimate_ego_status(prev, cur, dt=0.0) == 0.0)
    print("  [PASS] estimate_ego_status layout (N-2)")


if __name__ == "__main__":
    test_sparse_drive_planner()
    test_placeholder_determinism()
    test_temporal_api_placeholder()
    test_estimate_ego_status()
    test_build_projection_mat()
    test_postprocess_golden()
    test_postprocess_fail_loud()
    test_step_dt_contract()

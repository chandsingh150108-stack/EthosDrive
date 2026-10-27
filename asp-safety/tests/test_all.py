"""Comprehensive test suite for ASP Safety Layer."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("ASP Safety Layer - Comprehensive Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Test 1: SparseDrive
    print("\n[1/7] Testing SparseDrive...")
    try:
        from neural_planner import SparseDrivePlanner
        planner = SparseDrivePlanner(
            str(ROOT / "SparseDrive/ckpt/sparsedrive_stage2.pth"),
            device="cpu"
        )
        images = np.random.randn(6, 3, 256, 704).astype(np.float32)
        intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        ego2global = np.eye(4, dtype=np.float32)
        result = planner.predict(images, intrinsics, extrinsics, ego2global, 0.0)
        assert result['trajectory'].shape == (6, 2)
        results['SparseDrive'] = 'PASS'
    except Exception as e:
        results['SparseDrive'] = f'FAIL: {e}'
    
    # Test 2: TrajectoryChecker
    print("\n[2/7] Testing TrajectoryChecker...")
    try:
        from checker import TrajectoryChecker
        checker = TrajectoryChecker(str(ROOT / "rules"))
        safe_traj = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
        is_safe, violations = checker.check(safe_traj)
        assert is_safe == True
        results['TrajectoryChecker'] = 'PASS'
    except Exception as e:
        results['TrajectoryChecker'] = f'FAIL: {e}'
    
    # Test 3: TrajectoryCorrector
    print("\n[3/7] Testing TrajectoryCorrector...")
    try:
        from corrector import TrajectoryCorrector
        corrector = TrajectoryCorrector(checker)
        fast_traj = np.array([[0,0],[20,0],[40,0],[60,0],[80,0],[100,0]], dtype=np.float32)
        corrected, _ = corrector.correct(fast_traj)
        is_safe, _ = checker.check(corrected)
        assert is_safe == True
        results['TrajectoryCorrector'] = 'PASS'
    except Exception as e:
        results['TrajectoryCorrector'] = f'FAIL: {e}'
    
    # Test 4: DataLoader
    print("\n[4/7] Testing DataLoader...")
    try:
        from data_loader import NuScenesLoader
        loader = NuScenesLoader("nonexistent")
        sample = loader.get_scene_sample(0)
        assert sample['images'].shape == (6, 3, 256, 704)
        results['DataLoader'] = 'PASS'
    except Exception as e:
        results['DataLoader'] = f'FAIL: {e}'
    
    # Test 5: MetricsCalculator
    print("\n[5/7] Testing MetricsCalculator...")
    try:
        from evaluation import MetricsCalculator
        calc = MetricsCalculator()
        pred = np.array([[0,0],[1,0],[2,0]], dtype=np.float32)
        gt = np.array([[0,0],[1,1],[2,0]], dtype=np.float32)
        error = calc.l2_error(pred, gt)
        expected = 1.0 / 3.0
        assert abs(error - expected) < 0.01, f"Expected {expected}, got {error}"
        results['MetricsCalculator'] = 'PASS'
    except Exception as e:
        results['MetricsCalculator'] = f'FAIL: {e}'
    
    # Test 6: Pipeline
    print("\n[6/7] Testing Pipeline...")
    try:
        from pipeline import SafetyLayerPipeline
        pipeline = SafetyLayerPipeline(
            str(ROOT / "SparseDrive/ckpt/sparsedrive_stage2.pth"),
            str(ROOT / "rules"),
            device="cpu"
        )
        images = np.random.randn(6, 3, 256, 704).astype(np.float32)
        intrinsics = np.eye(3, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        extrinsics = np.eye(4, dtype=np.float32)[np.newaxis].repeat(6, axis=0)
        ego2global = np.eye(4, dtype=np.float32)
        result = pipeline.run(images, intrinsics, extrinsics, ego2global, 0.0)
        assert 'safe_trajectory' in result
        results['Pipeline'] = 'PASS'
    except Exception as e:
        results['Pipeline'] = f'FAIL: {e}'
    
    # Test 7: ASP Rules (full 63-rule catalog via basic.lp)
    print("\n[7/7] Testing ASP Rules...")
    try:
        import clingo
        rules_dir = ROOT / "rules"
        assert (rules_dir / 'basic.lp').exists(), "basic.lp missing"
        assert (rules_dir / 'helpers.lp').exists(), "helpers.lp missing"
        catalog = ['speed_regulation', 'lane_discipline', 'following_behavior',
                   'collision_avoidance', 'traffic_control', 'right_of_way',
                   'turning_rules', 'overtaking', 'vru_safety',
                   'intersection_behavior', 'emergency', 'comfort_quality',
                   'predictive_safety', 'environmental', 'complex_interactions']
        for rule in catalog:
            assert (rules_dir / f'{rule}.lp').exists(), f"{rule}.lp missing"
        ctrl = clingo.Control(['--warn=none'])
        ctrl.add('base', [], f'#include "{rules_dir.as_posix()}/basic.lp".')
        ctrl.add('base', [], 'vehicle(ego). pos(ego, 0, 0, 0).')
        ctrl.ground([('base', [])])
        results['ASP Rules'] = 'PASS'
    except Exception as e:
        results['ASP Rules'] = f'FAIL: {e}'
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    
    all_pass = True
    for test_name, result in results.items():
        status = "[PASS]" if result == 'PASS' else "[FAIL]"
        print(f"  {status} {test_name}")
        if result != 'PASS':
            print(f"        {result}")
            all_pass = False
    
    print("=" * 60)
    
    if all_pass:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


def test_all_components():
    """Pytest entry point (RQ-6): full smoke suite in one collected test."""
    assert run_all_tests() == 0


if __name__ == "__main__":
    sys.exit(run_all_tests())

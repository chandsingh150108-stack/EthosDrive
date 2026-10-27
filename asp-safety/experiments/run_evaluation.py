"""Placeholder batch demo for ASP Safety Layer (P-20: honest scope).

Runs the full pipeline (SparseDrive placeholder -> ASP Checker ->
Corrector) over synthetic scenes and reports safe-rate, violation
breakdown, and average/p95 latency. This exercises orchestration
only: collision-rate and L2 improvement on real data are measured
by run_calibration.py, not here. Placeholder scenes are
hostile-by-design (scene 9 plants an obstacle; the red at (12, 0)
is reachable), so expect violations on every scene -- the point is
the veto->fix loop, not a clean sheet.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pipeline import SafetyLayerPipeline
from data_loader import NuScenesLoader
from evaluation import MetricsCalculator


def run_evaluation(num_scenes=10, rules_dir=None, model_path=None):
    """Run full evaluation on placeholder or real data.
    
    Args:
        num_scenes: Number of scenes to evaluate
        rules_dir: Path to ASP rule files
        model_path: Path to SparseDrive weights
    """
    print("=" * 60)
    print("ASP Safety Layer - Evaluation")
    print("=" * 60)
    
    base_dir = Path(__file__).parent.parent
    if rules_dir is None:
        rules_dir = str(base_dir / "rules")
    if model_path is None:
        model_path = str(base_dir / "SparseDrive" / "ckpt" / "sparsedrive_stage2.pth")
    
    # Initialize components
    print("\nInitializing pipeline...")
    pipeline = SafetyLayerPipeline(
        planner_path=model_path,
        rules_dir=rules_dir,
        device="cpu"
    )
    loader = NuScenesLoader("nonexistent")
    calculator = MetricsCalculator()
    
    # Evaluate
    print(f"\nEvaluating {num_scenes} scenes...")
    results = pipeline.run_batch(loader, list(range(num_scenes)))
    
    # Collect metrics
    all_violations = []
    all_detection_times = []
    safe_count = 0
    total_count = 0
    
    for result in results:
        if "error" in result:
            print(f"  Scene {result['scene_idx']}: ERROR - {result['error']}")
            continue
        
        total_count += 1
        is_safe = result["is_safe"]
        violations = result["violations"]
        inf_time = result["inference_time_ms"]
        
        if is_safe:
            safe_count += 1
        
        all_violations.extend(violations)
        all_detection_times.append(inf_time)
        
        status = "SAFE" if is_safe else f"VIOLATION({len(violations)})"
        print(f"  Scene {result['scene_idx']}: {status} "
              f"({inf_time:.1f}ms, {len(result.get('scores', []))} detections)")
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    
    if total_count > 0:
        safety_rate = safe_count / total_count * 100
        avg_time = np.mean(all_detection_times) if all_detection_times else 0
        p95_time = np.percentile(all_detection_times, 95) if all_detection_times else 0
        
        violation_types = {}
        for v in all_violations:
            rule = v.get("rule", "unknown")
            violation_types[rule] = violation_types.get(rule, 0) + 1
        
        print(f"  Total scenes evaluated: {total_count}")
        print(f"  Safe trajectories: {safe_count} ({safety_rate:.1f}%)")
        print(f"  Unsafe trajectories: {total_count - safe_count}")
        print(f"  Total violations: {len(all_violations)}")
        if violation_types:
            print(f"  Violation breakdown:")
            for rule, count in sorted(violation_types.items()):
                print(f"    {rule}: {count}")
        print(f"  Average inference time: {avg_time:.2f} ms")
        print(f"  P95 inference time: {p95_time:.2f} ms")
        print(f"  Safety check latency: {calculator.average_inference_time(all_detection_times):.2f} ms")
    else:
        print("  No scenes evaluated successfully.")
    
    print("=" * 60)
    
    return {
        "total": total_count,
        "safe": safe_count,
        "violations": all_violations,
        "times": all_detection_times
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate ASP Safety Layer")
    parser.add_argument("--num-scenes", type=int, default=10,
                       help="Number of scenes to evaluate")
    parser.add_argument("--rules-dir", type=str, default=None,
                       help="Path to ASP rule files")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to SparseDrive weights")
    
    args = parser.parse_args()
    run_evaluation(
        num_scenes=args.num_scenes,
        rules_dir=args.rules_dir,
        model_path=args.model_path
    )

#!/usr/bin/env python3
"""Run multi-seed CARLA closed-loop matrix.

Usage: python experiments/run_matrix.py [--seeds 1,2,3] [--scenarios S1,S2,S3,S4,S5] [--engines off,heuristic,qp] [--plan-engine lane]
"""
import sys
import json
import time
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = r"C:\Users\chand\AppData\Local\Microsoft\WindowsApps\python.exe"  # fallback
# Use the actual python from the env
import sys
PY = sys.executable

SCENARIOS = ['S1', 'S2', 'S3', 'S4', 'S5']
ENGINES = ['off', 'heuristic', 'qp']
SEEDS = [1, 2, 3]

def run_one(scenario, seed, engine, plan_engine='lane', max_distance=600, max_minutes=6, log_dir=None):
    """Run a single closed-loop experiment via subprocess."""
    cmd = [
        sys.executable,
        str(ROOT / "run_closed_loop.py"),
        "--scenario", scenario,
        "--seed", str(seed),
        "--engine", engine,
        "--plan-engine", plan_engine,
        "--max-distance", str(max_distance),
        "--max-minutes", str(max_minutes),
    ]
    if log_dir:
        cmd.extend(["--log-dir", str(log_dir)])
    
    print(f"  RUN: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=max_minutes * 60 + 120)
    dt = time.time() - t0
    
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode}): {result.stderr[:200]}")
        return None
    
    # Parse the last line for the output path
    for line in result.stdout.strip().split('\n'):
        if line.startswith('[run_closed]'):
            parts = line.split()
            # Extract the JSON filename
            for part in parts:
                if part.endswith('.json'):
                    return part
    return None

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', default='1,2,3')
    parser.add_argument('--scenarios', default='S1,S2,S3,S4,S5')
    parser.add_argument('--engines', default='off,heuristic,qp')
    parser.add_argument('--plan-engine', default='lane')
    parser.add_argument('--max-distance', type=float, default=600)
    parser.add_argument('--max-minutes', type=float, default=6)
    parser.add_argument('--log-dir', default=None)
    args = parser.parse_args()
    
    seeds = [int(s) for s in args.seeds.split(',')]
    scenarios = args.scenarios.split(',')
    engines = args.engines.split(',')
    
    log_dir = Path(args.log_dir) if args.log_dir else ROOT / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    total = len(seeds) * len(scenarios) * len(engines)
    print(f"Running {total} experiments: seeds={seeds}, scenarios={scenarios}, engines={engines}")
    print(f"Log dir: {log_dir}")
    
    results = []
    for i, seed in enumerate(seeds):
        for scenario in scenarios:
            for engine in engines:
                print(f"\n[{len(results)+1}/{total}] {scenario} seed={seed} engine={engine}")
                try:
                    out_file = run_one(scenario, seed, engine, args.plan_engine,
                                      args.max_distance, args.max_minutes, log_dir)
                    if out_file:
                        results.append({
                            'scenario': scenario, 'seed': seed, 'engine': engine,
                            'output': out_file, 'status': 'completed'
                        })
                        print(f"  -> {out_file}")
                    else:
                        results.append({
                            'scenario': scenario, 'seed': seed, 'engine': engine,
                            'status': 'failed'
                        })
                        print(f"  -> FAILED")
                except subprocess.TimeoutExpired:
                    results.append({
                        'scenario': scenario, 'seed': seed, 'engine': engine,
                        'status': 'timeout'
                    })
                    print(f"  -> TIMEOUT")
                except Exception as e:
                    results.append({
                        'scenario': scenario, 'seed': seed, 'engine': engine,
                        'status': f'error: {e}'
                    })
                    print(f"  -> ERROR: {e}")
    
    # Save summary
    summary_file = log_dir / f"matrix_summary_{int(time.time())}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'seeds': seeds, 'scenarios': scenarios, 'engines': engines,
            'results': results, 'timestamp': time.time()
        }, f, indent=2)
    print(f"\nSummary saved to {summary_file}")
    
    # Print summary table
    print("\nSUMMARY:")
    for r in results:
        print(f"  {r['scenario']} seed={r['seed']} engine={r['engine']}: {r['status']}")

if __name__ == '__main__':
    main()
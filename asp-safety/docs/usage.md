# ASP Safety Layer - Usage Guide

## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
./setup_env.sh

# Activate the environment
conda activate asp-safety
```

### 2. Run Basic Evaluation

```bash
cd asp-safety
python experiments/run_evaluation.py
```

### 3. Run Unit Tests

```bash
cd asp-safety
python tests/test_checker.py
```

## Using the Checker

### Basic Usage

```python
import numpy as np
from src.checker import TrajectoryChecker

# Initialize checker
checker = TrajectoryChecker(rules_dir="rules")

# Create a trajectory
trajectory = np.array([
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
    [3.0, 0.0],
    [4.0, 0.0],
    [5.0, 0.0],
])

# Check trajectory
is_safe, violations = checker.check(trajectory)

print(f"Is safe: {is_safe}")
print(f"Violations: {violations}")
```

### With Map Data

```python
# Define map data (lane boundaries)
map_data = {
    'lanes': {
        'lane_0': [
            {'x1': -10, 'y1': -2, 'x2': 100, 'y2': -2},
            {'x1': -10, 'y1': 2, 'x2': 100, 'y2': 2}
        ]
    }
}

# Check trajectory with map data
is_safe, violations = checker.check(trajectory, map_data=map_data)
```

### Trajectory Correction

```python
# Get corrected trajectory
corrected = checker.get_safe_trajectory(trajectory)

print(f"Original: {trajectory[-1]}")
print(f"Corrected: {corrected[-1]}")
```

## Project Structure

```
asp-safety/
├── rules/                    # ASP traffic rules (63-rule catalog, R1-R15)
│   ├── helpers.lp           # shared predicates (in_lane, speed, arrival, ...)
│   ├── speed_regulation.lp  # R1.1-R1.5
│   ├── lane_discipline.lp   # R2.1-R2.5
│   ├── following_behavior.lp# R3.1-R3.4
│   ├── collision_avoidance.lp # R4.1-R4.5
│   ├── traffic_control.lp   # R5.1-R5.5
│   ├── right_of_way.lp      # R6.1-R6.5
│   ├── turning_rules.lp     # R7.1-R7.5
│   ├── overtaking.lp        # R8.1-R8.4
│   ├── vru_safety.lp        # R9.1-R9.5
│   ├── intersection_behavior.lp # R10.1-R10.4
│   ├── emergency.lp         # R11.1-R11.4
│   ├── comfort_quality.lp   # R12.1-R12.4
│   ├── predictive_safety.lp # R13.1-R13.4
│   ├── environmental.lp     # R14.1-R14.4
│   ├── complex_interactions.lp # R15.1-R15.4
│   └── basic.lp            # Aggregates all rules
├── src/                      # Source code
│   ├── checker.py           # ASP safety checker
│   ├── corrector.py         # Trajectory fixpoint engine
│   ├── correction_strategies.py # rule->strategy dispatch table
│   ├── neural_planner.py    # SparseDrive wrapper (+temporal API)
│   ├── data_loader.py       # Data loading + real map extraction
│   ├── evaluation.py        # Metrics calculation
│   ├── pipeline.py          # End-to-end pipeline
│   └── utils.py            # Helper functions
├── experiments/              # Harnesses (calibration, temporal, sweeps)
│   ├── run_evaluation.py    # placeholder batch demo
│   ├── run_calibration.py   # real-data baseline + trajectory dumps
│   ├── run_temporal.py      # warm-vs-cold harness
│   ├── cmd_sweep.py         # nav-command semantics probe
│   └── sweep_thresholds.py  # GPU-free threshold sweeps
├── tests/                    # Unit tests (92 green)
│   ├── test_all.py
│   ├── test_checker.py
│   ├── test_checker_edges.py
│   ├── test_corrector.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_performance.py
│   ├── test_pipeline.py
│   ├── test_rules.py
│   ├── test_sparse_drive.py
│   └── test_t2_coverage.py
└── configs/                  # Configuration files
    ├── default.yaml
    └── correction_strategies.yaml
```

## Adding New Rules

### 1. Extend a category file

New rules go in the matching `rules/<category>.lp`, not in new
files (one file per category keeps the blob assembly stable).
Integer-cm conventions are mandatory (`docs/conventions.md`):
no floats, no `sqrt()`, squared distances, 32-bit-safe products.

```asp
% Violation condition (positions are cm integers, speeds squared)
violation(my_new_rule, ego, T) :-
    pos(ego, T, X, Y), time(T),
    my_condition(X, Y).
```

No `#minimize` directives: findings are collected by the checker,
and `override_active/2` (comfort_quality.lp) is the only
cross-rule arbitration point besides `threat_level/3`.

### 2. Register the finding

Add the head name to `src/correction_strategies.py` (routing group)
and assert it in `tests/test_t2_coverage.py` — `test_rule_registry`
fails the suite for unasserted heads (P-9 policy).

## Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Collision Rate | % of trajectories with collisions | <0.05% |
| L2 Error | Average displacement error | <0.7m |
| Inference Time | Time per trajectory check | <50ms |
| Violation Detection Rate | % of true violations detected | >90% |

## Troubleshooting

### Clingo Import Error

If you get an import error for clingo:

```bash
pip install clingo==5.7.1
```

### CUDA Out of Memory

If you get CUDA out of memory errors, construct the planner on CPU
(`TrajectoryChecker` itself has no `device` parameter — it runs Clingo
on CPU always):

```python
from src.pipeline import SafetyLayerPipeline

pipeline = SafetyLayerPipeline(
    planner_path="SparseDrive/ckpt/sparsedrive_stage2.pth",
    rules_dir="rules",
    device="cpu",  # planner device; checker is CPU-only
)
```

### Rule Grounding Errors

If you get grounding errors in ASP rules:

1. Check syntax of `.lp` files
2. Ensure all variables are bound
3. Check for circular dependencies

## Next Steps

1. Closed-loop bridge on the installed CARLA server (0.9.15 at
   `C:\CARLA`, smoke-verified S12), then scripted scenarios
2. Re-sweep thresholds on tracked-agent + warm data
3. Real-scene perf headroom (corrector churn under agents)

---

*Last Updated: September 26, 2026*

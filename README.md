# ASP Safety Layer for Autonomous Vehicles

An ASP-based safety layer that checks and corrects neural planner trajectories against formal traffic rules, reducing collision rate with minimal latency overhead.

## Project Overview

This project implements a safety layer for autonomous driving that:
1. Takes trajectories from a neural planner (SparseDrive)
2. Checks them against formal traffic rules using ASP (Answer Set Programming)
3. Corrects violations if found
4. Returns safe trajectories


## Quick Start

### 1. Setup Environment

```bash
# Run the setup script
./setup_env.sh

# Activate the environment
conda activate asp-safety
```

### 2. Clone SparseDrive

```bash
cd asp-safety
git clone https://github.com/swc-17/SparseDrive.git
cd SparseDrive

# Download pre-trained weights
mkdir -p ckpt
wget https://github.com/swc-17/SparseDrive/releases/download/v1.0/sparsedrive_stage2.pth \
    -O ckpt/sparsedrive_stage2.pth
```

### 3. Download nuScenes Dataset

1. Go to https://www.nuscenes.org/download
2. Create an account and agree to terms
3. Download the mini split for development
4. Extract to `C:/data/sets/nuscenes` (Windows; override the
   `data_dir` default on Linux). The map-expansion zip extracts as
   `expansion/*.json` + `basemap/*.png`; copy both under `maps/`
   (devkit layout — see `docs/environment.md`).

## Project Structure

```
asp-safety/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup_env.sh             # Environment setup script
├── pyproject.toml           # Project metadata
├── .github/workflows/       # CI (Clingo-only fast suites)
├── configs/                 # Configuration files
│   ├── default.yaml
│   └── correction_strategies.yaml  # corrector tuning params
├── rules/                   # ASP traffic rules (63-rule catalog, R1-R15)
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
├── src/                     # Source code
│   ├── __init__.py
│   ├── checker.py           # ASP safety checker (+constraint emission)
│   ├── corrector.py         # Trajectory fixpoint engine (+CorrectorQP)
│   ├── constraint_optimizer.py # projected-gradient QP on ASP bounds
│   ├── correction_strategies.py # rule->strategy dispatch table
│   ├── neural_planner.py    # SparseDrive wrapper (+temporal API)
│   ├── data_loader.py       # nuScenes loading + real map extraction
│   ├── evaluation.py        # Metrics calculation
│   ├── pipeline.py          # End-to-end pipeline
│   └── utils.py             # Utility functions
├── experiments/             # Harnesses (calibration, replay, ablations)
│   ├── run_evaluation.py    # placeholder batch demo
│   ├── run_calibration.py   # real-data baseline + trajectory dumps
│   ├── run_temporal.py      # warm-vs-cold harness
│   ├── replay_eval.py       # tracked-annotation replay baseline
│   ├── cmd_sweep.py         # nav-command semantics probe
│   ├── sweep_thresholds.py  # GPU-free threshold sweeps
│   ├── ablate_corrector.py  # corrector/QP bound-family ablation
│   ├── sweep_qp_params.py   # QP optimizer 12-combo sweep
│   ├── cap_ab.py            # map-tile cap A/B probe
│   ├── carla_launch.py      # CARLA server lifecycle (S12)
│   ├── carla_probe.py       # 6-cam rig + projection sanity (P25-A)
│   ├── carla_context.py     # CARLA facts -> checker context (P25-B)
│   ├── carla_bridge.py      # closed-loop shield bridge (P25-B/D)
│   ├── carla_scenarios.py   # S1-S5 scripted scenarios (P25-D)
│   ├── run_closed_loop.py   # scenario runner CLI (P25-D)
│   ├── carla_sign_test.py   # steer-sign calibration probe
│   ├── carla_metrics.py     # frozen-matrix aggregator (P25-C)
│   ├── sign_test.py         # steering sign harness
│   └── coverage_probe.py    # registry tripwire helper
├── tests/                   # Unit tests (120 green)
│   ├── test_all.py
│   ├── test_checker.py
│   ├── test_checker_edges.py
│   ├── test_constraint_emission.py
│   ├── test_corrector.py
│   ├── test_data_loader.py
│   ├── test_evaluation.py
│   ├── test_performance.py
│   ├── test_pipeline.py
│   ├── test_qp_feasibility.py
│   ├── test_rules.py
│   ├── test_sparse_drive.py
│   └── test_t2_coverage.py
├── docs/                    # Documentation
│   ├── usage.md
│   ├── api_reference.md
│   ├── conventions.md       # units/contracts reference
│   ├── rule_divergences.md  # spec-vs-impl log
│   ├── data_contracts.md
│   ├── sparsedrive_integration.md
│   ├── testdoc.md
│   ├── test_coverage.md     # suite-by-suite coverage map
│   ├── environment.md       # canonical env record
│   ├── carla_harness.md     # CARLA closed-loop deviations ledger
│   └── replay_eval_design.md
└── SparseDrive/             # SparseDrive model (cloned)
```

## Status

CARLA closed-loop pilot complete (2026-10-22, P-25, S19-S22):
6-camera bridge, S1-S5 scripted scenarios, frozen 15-run matrix
(5 scenarios x off/heuristic/qp) with
`experiments/carla_metrics.json` — the two adversarial unshielded
arms collide (13.3% collision rate) while every shielded arm
survives (0 collisions, 0 give-ups); deviations ledger in
`docs/carla_harness.md`. ASP-constrained correction track complete
(2026-10-03, S13-S17): constraint emission in all 15 rule files,
`check(include_constraints=True)`, pure-NumPy QP corrector
(`CorrectorQP`), pipeline `corrector_mode`, bound-family +
param-sweep ablations. 120 tests green, 63-rule catalog with 73/73
firing coverage, warm calibration baseline (L2 10.18 m vs 11.54 m
cold; GT 0.3 / pred 0.2 / fixed 0.5). Details: `whatodo.md`,
`session.md`, `futuretodoist.md`. Targets below are design goals,
not claims — measured values live in `experiments/` verdicts and
JSONs.

## Resources

- **SparseDrive**: https://github.com/swc-17/SparseDrive
- **nuScenes**: https://www.nuscenes.org/
- **Clingo (ASP Solver)**: https://potassco.org/clingo/
- **CARLA Simulator**: https://carla.org/ (optional)

## License

This project is for educational purposes (course project).

---

*Last Updated: October 22, 2026*

# EDI / ASP Safety Layer — Comprehensive Repository Analysis & Documentation Deep-Dive

## Executive Summary

The **ASP Safety Layer for Autonomous Vehicles** is a neuro-symbolic safety shield designed to supervise and correct end-to-end neural trajectory planners (specifically **SparseDrive**, ICRA 2025 / ECCV 2026). The system takes real-world multimodal sensor observations, predicts trajectories via a neural network, translates candidate trajectories and environmental contexts into declarative Answer Set Programming (ASP) facts, evaluates them against a formal **63-rule traffic catalog** in **Clingo**, and applies either a heuristic fixpoint or continuous Quadratic Program (QP) / projected-gradient correction to eliminate rule violations before trajectory execution.

---

## 1. Complete Repository Architecture & File Taxonomy

```
EDI/
├── AGENTS.md                                   # Master specification blueprint (Subdivisions 1-11)
├── asp_traffic_rules.md                        # Full 63-rule ASP catalog specifications across 15 families
├── INSTALL.md                                  # OpenCode control plane configuration guide
├── opencode.json                               # Tool permissions & agent configurations
├── setup_env.sh                                # Root conda setup script
├── requirements.txt                            # Root dependencies
├── skills-lock.json                            # Registered agent skills
├── nuScenes-map-expansion-v1.3.zip             # HD Map expansion data
├── v1.0-mini.tgz                               # NuScenes mini dataset archive
│
└── asp-safety/                                 # Core Implementation Package
    ├── pyproject.toml                          # Package metadata & build configuration
    ├── requirements.txt                        # Strict dependency pins (Python 3.10, PyTorch 2.1.0+cu118)
    ├── setup_env.sh                            # Canonical environment setup
    ├── README.md                               # Project overview, quickstart, roadmap
    ├── SETUP_GUIDE.md                          # 11-step hardware & GPU setup guide (RTX 4050 / CARLA)
    ├── whatodo.md                              # Living task board and execution progress log
    ├── futuretodoist.md                        # Comprehensive itemized backlog (T, P, N, PR, RQ tracks)
    ├── session.md                              # Historical chronological development log (Sessions S0-S17)
    │
    ├── .github/workflows/
    │   └── fast-tests.yml                      # CI workflow for Clingo-only test suites
    │
    ├── configs/
    │   ├── default.yaml                        # Master system configuration parameters
    │   └── correction_strategies.yaml          # Trajectory correction tuning knobs & caps
    │
    ├── docs/                                   # Architectural Contracts & Technical References
    │   ├── api_reference.md                    # Public API reference for Planner, Checker, Corrector, Pipeline
    │   ├── conventions.md                      # Coordinate frames, integer-cm scaling, squared speed units
    │   ├── data_contracts.md                   # Inter-module data flow shapes, types, and schema matrix
    │   ├── environment.md                      # Canonical pinned environment record (CUDA, MMCV, MMDetection)
    │   ├── evaluation_design_asp_correction.md # Rigorous research plan: heuristic vs QP vs ablations
    │   ├── IMPLEMENTATION_PLAN_ASP_CONSTRAINTS.md # Architecture for ASP constraint emission & QP solving
    │   ├── replay_eval_design.md               # Annotation replay harness and CARLA closed-loop design
    │   ├── rule_divergences.md                 # 52 logged divergences between theoretical spec and implementation
    │   ├── sparsedrive_integration.md          # Neural planner boundary, tensors, anchors, temporal queues
    │   ├── testdoc.md                          # Detailed per-suite testing guide & execution instructions
    │   ├── test_coverage.md                    # Rule firing registry verification (73/73 catalog heads)
    │   ├── usage.md                            # End-user developer guide and rule creation tutorial
    │   └── repository_deep_dive.md             # Complete codebase analysis and documentation inventory
    │
    ├── paper/
    │   ├── literature.md                       # 12 annotated peer-reviewed foundational literature reviews
    │   └── report.tex                          # Course report & workshop paper LaTeX manuscript skeleton
    │
    ├── rules/                                  # 63 ASP Traffic Rules (Integer-cm Arithmetic)
    │   ├── helpers.lp                          # Shared predicates, spatial conversions, #show declarations
    │   ├── basic.lp                            # Catalog aggregator and total violation counter
    │   ├── speed_regulation.lp                 # R1.1 - R1.5 (Urban, highway, zone, minimum, VRU speeds)
    │   ├── lane_discipline.lp                  # R2.1 - R2.5 (Lane keeping, transitions, illegal maneuvers)
    │   ├── following_behavior.lp               # R3.1 - R3.4 (Time headway, gap, oscillation damping)
    │   ├── collision_avoidance.lp              # R4.1 - R4.5 (Lateral, longitudinal, safety bubble, cut-in)
    │   ├── traffic_control.lp                  # R5.1 - R5.5 (Red lights, yellow dilemma, green obstacles)
    │   ├── right_of_way.lp                     # R6.1 - R6.5 (Uncontrolled intersections, roundabouts, yields)
    │   ├── turning_rules.lp                    # R7.1 - R7.5 (RTOR, left turns across oncoming, U-turns)
    │   ├── overtaking.lp                       # R8.1 - R8.4 (Passing zones, oncoming traffic, sightlines)
    │   ├── vru_safety.lp                       # R9.1 - R9.5 (Crosswalks, bike lanes, school zones, sidewalks)
    │   ├── intersection_behavior.lp            # R10.1 - R10.4 (Gridlock box blocking, queue integrity)
    │   ├── emergency.lp                        # R11.1 - R11.4 (Emergency braking feasibility, work zones)
    │   ├── comfort_quality.lp                  # R12.1 - R12.4 (Acceleration/deceleration caps, jerk, reversal)
    │   ├── predictive_safety.lp                # R13.1 - R13.4 (TTC bands, multi-horizon collision forecasts)
    │   ├── environmental.lp                    # R14.1 - R14.4 (Wet roads, night overdriving, low visibility)
    │   └── complex_interactions.lp             # R15.1 - R15.4 (Platooning, zipper merges, multiway priority)
    │
    ├── src/                                    # Core Python Architecture
    │   ├── __init__.py                         # Package exports
    │   ├── checker.py                          # TrajectoryChecker: Clingo grounding, solving, atom parsing
    │   ├── constraint_optimizer.py             # ConstraintOptimizer: Projected-gradient continuous solver
    │   ├── correction_strategies.py            # Heuristic strategy dispatch table and parameter registry
    │   ├── corrector.py                        # TrajectoryCorrector (fixpoint) & CorrectorQP (QP solver)
    │   ├── data_loader.py                      # NuScenesLoader: Sensor data, HD map extraction, tracked agents
    │   ├── neural_planner.py                   # SparseDrivePlanner: Model wrapper with warm temporal sequence API
    │   ├── pipeline.py                         # SafetyLayerPipeline: End-to-end integration orchestrator
    │   ├── evaluation.py                       # MetricsCalculator: L2 error, yaw-aware collisions, detection rates
    │   └── utils.py                            # Vector math, kinematics, and coordinate transforms
    │
    ├── tests/                                  # Comprehensive Test Suite (119 Green Tests)
    │   ├── test_all.py                         # Fast 7-component system smoke test
    │   ├── test_checker.py                     # Safety discrimination & non-vacuity assertions
    │   ├── test_checker_edges.py               # Cull radius, temporal guards, warning isolation
    │   ├── test_constraint_emission.py         # Verification that ASP rules emit continuous bounds
    │   ├── test_corrector.py                   # Heuristic fixpoint convergence & edge assertions
    │   ├── test_data_loader.py                 # NuScenes placeholder & real loader contracts
    │   ├── test_evaluation.py                  # Evaluation metric correctness
    │   ├── test_performance.py                 # Latency budgets (bare <50ms, 20-agent <100ms)
    │   ├── test_pipeline.py                    # Proposal -> Veto -> Repair -> Verification loop
    │   ├── test_qp_feasibility.py              # Projected gradient & CorrectorQP feasibility
    │   ├── test_rules.py                       # Per-family safe/unsafe pair tests
    │   ├── test_sparse_drive.py                # SparseDrive shapes, projections, temporal state
    │   ├── test_t2_coverage.py                 # Complete 73/73 catalog rule registry verification
    │   └── __init__.py
    │
    ├── experiments/                            # Evaluation Harnesses, Benchmarks & Frozen Data
    │   ├── run_calibration.py                  # NuScenes real-data calibration harness (GPU/CPU)
    │   ├── run_evaluation.py                   # Offline batch pipeline validation
    │   ├── run_temporal.py                     # Multi-frame warm vs cold inference benchmark
    │   ├── replay_eval.py                      # 184 tracked-agent annotation replay harness
    │   ├── sweep_thresholds.py                 # GPU-free threshold sensitivity sweep
    │   ├── sweep_qp_params.py                  # QP optimizer hyperparameter grid search
    │   ├── ablate_corrector.py                 # 4-engine corrector ablation (Heuristic vs QP variants)
    │   ├── cmd_probe.py / cmd_sweep.py         # SparseDrive navigation command conditioning probes
    │   ├── coverage_probe.py                   # Registry coverage audit tripwire
    │   ├── cap_ab.py                           # Map tile querying and culling optimization probe
    │   ├── carla_launch.py / carla_smoke.py    # CARLA 0.9.15 simulator lifecycle & camera smoke tests
    │   ├── calibration_baseline.json           # Measured baseline (Warm: GT 0.3 / Pred 0.2 / Fixed 0.5)
    │   ├── calibration_extrapolated.json       # Evidence showing CV trajectory extrapolation failed
    │   ├── qp_ablation.json                    # Benchmark comparing Full vs Caps vs Boxes vs Heuristic
    │   ├── qp_sweep.json                       # 12-combination QP optimizer parameter grid results
    │   ├── replay_baseline.json                # Replay metrics on 184 tracked agents
    │   ├── sweep_results.json                  # Detection threshold sweep data
    │   ├── trajs.json                          # Frozen 10-scene dataset dump for GPU-free experimentation
    │   ├── cmd_sweep_results.md                # Analysis of planner navigation head conditioning
    │   ├── extrapolation_verdict.md            # Empirical verdict dropping CV trajectory extrapolation
    │   ├── sweep_verdict.md                    # Empirical verdict setting detection score knee to 0.5
    │   └── temporal_results.md                 # 10/10 scene improvement from warm temporal queues
    │
    └── SparseDrive/                            # Neural Planner Submodule (ICRA 2025)
        ├── README.md                           # Upstream model architecture overview & benchmarks
        ├── requirement.txt                     # Upstream dependency declarations
        ├── ckpt/sparsedrive_stage2.pth         # Stage 2 pre-trained model weights (986 MB)
        ├── projects/configs/                   # Model architectures (small stage 1 & stage 2)
        ├── projects/mmdet3d_plugin/            # Decoders, heads, attention modules, instance bank
        └── tools/ & scripts/                   # Data converters, k-means anchors, visualization
```

---

## 2. In-Depth Analysis of Key Markdown Documents

### A. Core Blueprints & Rules
- [AGENTS.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/AGENTS.md):
  The definitive project roadmap outlining 11 subdivisions from environment setup to paper submission. Note: Subdivision 3 (initial float-based prototype) was explicitly superseded by Subdivision 11, `asp_traffic_rules.md`, and `docs/conventions.md`.
- [asp_traffic_rules.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp_traffic_rules.md):
  A 1,307-line formal catalog detailing 63 traffic rules across 15 distinct legal/safety categories: Speed (R1), Lanes (R2), Following (R3), Collisions (R4), Signals (R5), Priority (R6), Turning (R7), Overtaking (R8), VRUs (R9), Intersections (R10), Emergencies (R11), Comfort (R12), Prediction (R13), Environment (R14), and Complex Multi-Agent interactions (R15).

### B. Engineering Backlog & Session Logs
- [whatodo.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/whatodo.md):
  The live engineering workboard tracking day-to-day delivery. Documents the completion of Month 1 (Weeks 1 to 4) and the October 2026 ASP constraint emission and continuous QP optimization phases.
- [session.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/session.md):
  A granular 768-line chronological log documenting sessions S0 through S17. Detailed milestones include pure-PyTorch fallbacks for non-MSVC Windows environments, CUDA 2.4s inference on RTX 4050, nuScenes map extraction, warm temporal queue caching, CARLA 0.9.15 installation, and QP bound-family ablations.
- [futuretodoist.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/futuretodoist.md):
  The unified historical backlog cross-referencing task IDs (T-2 to T-13, P-1 to P-20, N-1 to N-4, PR-1 to PR-3) with their completion status and technical artifacts.

### C. Technical Contracts & Reference Documents (`docs/`)
- [conventions.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/docs/conventions.md):
  Enforces coordinate system rules: ego-frame, positions in meters in Python, **integer centimeters (x100)** in ASP. Strict integer arithmetic (no floats, no `sqrt`); speeds in ASP are **squared speeds** in $(\text{cm}/\text{step})^2$.
- [data_contracts.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/docs/data_contracts.md):
  Master producer-consumer contract table detailing tensors, shapes, types, and schemas at every stage of the pipeline (from camera images `(6, 3, 256, 704)` to verified trajectory points `(4, 2)`).
- [sparsedrive_integration.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/docs/sparsedrive_integration.md):
  Details the exact boundary between the neural model and ASP shield: projection matrices ($K \cdot [R|t]$), tensor normalization, 10-dimensional bounding box slicing, synthetic k-means anchors, and temporal state isolation.
- [rule_divergences.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/docs/rule_divergences.md):
  Itemizes 52 specific points of divergence between theoretical law specifications and real-world ASP implementations (e.g., int32 overflow protection in R3.4/R15.4, one-sided hold-back boxes for following vehicles, sign-split absolute values).
- [evaluation_design_asp_correction.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/docs/evaluation_design_asp_correction.md):
  The experimental evaluation plan comparing SparseDrive (B0), the heuristic dispatch corrector (B1), and the ASP-constrained QP corrector (B2) across ablations A1–A7 on nuScenes tracked annotations and CARLA.
- [replay_eval_design.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/docs/replay_eval_design.md):
  Architectural design for annotation replay over 184 tracked agents, isolating parked vs live vehicles through trajectory variance without requiring perception semantic classification.
- [test_coverage.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/docs/test_coverage.md):
  Verifies that all 73 distinct violation, hazard, and warning heads across the 63-rule catalog fire on demand in unit tests.

### D. Literature & Research Benchmarking
- [literature.md](file:///c:/Users/chand/OneDrive/Desktop/EDI/asp-safety/paper/literature.md):
  Exhaustive review of 12 core academic foundations: SparseDriveV2 (ECCV 2026), MomAD (CVPR 2025), NAVSIM (NeurIPS 2024), Bench2Drive (NeurIPS 2024), nuScenes (CVPR 2020), Althoff et al. (Proc. IEEE 2025), SMT trajectory repair (Lin et al. 2025), neuro-symbolic LLM+ASP planning (Wei & AbdAlmageed 2026), conformal perception bounds (NeuS 2025), Road2Code (NeuS 2025), and Clingo multi-shot semantics (Gebser et al., TPLP 2019).

---

## 3. Measured Experimental Baselines & Key Findings

| Experiment Track | Key Finding / Metric | Empirical Evidence Source |
|---|---|---|
| **Temporal Inference (N-2)** | Warm multi-frame queues beat cold inference in 10/10 scenes: L2 displacement drops from **11.54 m to 10.18 m** | `experiments/temporal_results.md` |
| **Trajectory Extrapolation (N-3b)** | Extrapolating 3s plans to 6s with constant velocity adds +20 false violations and increases corrector latency 6x; **DROPPED** in favor of honest 3s horizon | `experiments/extrapolation_verdict.md` |
| **Detection Score Sweep (P-5)** | Detection confidence knee established at **0.5**; halves false collisions without losing true safety signals | `experiments/sweep_verdict.md` |
| **Nav-Command Conditioning (P-13)** | Neural planner exhibits weak steer-command conditioning under single-frame evaluation | `experiments/cmd_sweep_results.md` |
| **Parked vs Live Vehicles (S10)** | Tracking agent series over time explains **~60% of collision violations** as parked vehicles; solved without neural box classes | `experiments/replay_baseline.json` |
| **ASP-Constrained QP Optimizer (S14-S17)** | Pure QP solves lane violations **10x faster** (228ms vs 2872ms); joint bound solving (`full` 0.38) outperforms isolated bound subsets (`caps` 0.00, `boxes` 0.12) | `experiments/qp_ablation.json` |
| **Official Benchmark Baseline** | Ground-Truth Safe: **0.30** / Raw Pred Safe: **0.20** / Post-Shield Fixed: **0.50** / L2 Error: **10.18 m** | `experiments/calibration_baseline.json` |

---

## 4. Current System State & Recommended Next Actions

1. **Test Suite Status**: **119 passing unit & integration tests** covering Clingo ASP reasoning, spatial math, QP optimization, and pipeline data contracts.
2. **Simulation Status**: CARLA 0.9.15 server installed and smoke-tested on Town10HD_Opt (`C:\CARLA`); closed-loop reactive bridge remains future work.
3. **Paper Manuscript**: `paper/report.tex` is currently a 5-rule skeleton; ready to be upgraded to reflect the full 63-rule catalog, warm temporal inference, and QP constraint optimization results.


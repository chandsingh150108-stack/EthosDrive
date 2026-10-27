# Literature Review — ASP Safety Layer for Autonomous Driving

> Scope: 12 papers supporting the project architecture — neural planner proposes,
> formal traffic rules specify, ASP solver arbitrates, closed-loop benchmarks evaluate.
> Every author list, year, and venue below was triple-checked against arXiv,
> publisher, and proceedings pages.

---

## A. End-to-end Neural Planning

### [1] SparseDriveV2: Scoring is All You Need for End-to-End Autonomous Driving

- **Authors:** Wenchao Sun, Xuewu Lin, Keyu Chen, Zixiang Pei, Xiang Li, Yining Shi, Sifa Zheng
- **Year:** 2026
- **Published in:** arXiv preprint arXiv:2603.29163 (accepted to the European Conference on Computer Vision, ECCV 2026)
- **Relevance:** Current state of the art in scoring-based end-to-end planning. Introduces a factorized trajectory vocabulary (geometric paths × velocity profiles) enabling dense action-space coverage, achieving 92.0 PDMS on NAVSIM and 89.15 Driving Score on Bench2Drive with a lightweight ResNet-34 backbone.
- **Use in this project:** Successor to the SparseDrive baseline the safety layer wraps; motivates treating the neural planner as a scorer/proposer whose output requires an independent symbolic veto.
- **Limitation:** Scoring alone provides no named rule guarantees; dense vocabularies increase scoring cost.

### [2] Don't Shake the Wheel: Momentum-Aware Planning in End-to-End Autonomous Driving

- **Authors:** Ziying Song, Caiyan Jia, Lin Liu, Hongyu Pan, Yongchang Zhang, Junming Wang, Xingyu Zhang, Shaoqing Xu, Lei Yang, Yadan Luo
- **Year:** 2025
- **Published in:** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2025), pp. 22432–22441
- **Relevance:** Momentum-Aware Driving (MomAD) framework with Topological Trajectory Matching and Momentum Planning Interactor; introduces the Trajectory Prediction Consistency (TPC) metric. Reduces collision rate by 26% versus SparseDrive on the curated Turning-nuScenes set.
- **Use in this project:** Direct evidence that single-frame, single-shot planning is temporally unstable — motivates the project's frame-to-frame inconsistency concern and future cross-frame hysteresis work.
- **Limitation:** Improves stability statistically; provides no formal rule-compliance check.

---

## B. Benchmarks and Datasets

### [3] NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking

- **Authors:** Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang, Zetong Yang, Hongyang Li, Igor Gilitschenski, Boris Ivanovic, Marco Pavone, Andreas Geiger, Kashyap Chitta
- **Year:** 2024
- **Published in:** Advances in Neural Information Processing Systems (NeurIPS 2024), Datasets and Benchmarks Track, pp. 28706–28719
- **Relevance:** Middle ground between open-loop displacement metrics and closed-loop simulation; introduces the Planning Driving Metric Score (PDMS) with safety-aware sub-metrics (collision, drivable-area, time-to-collision, progress, comfort).
- **Use in this project:** Reference evaluation design for the project's planned move beyond L2/collision-rate on placeholder data toward safety-aware scoring.
- **Limitation:** Non-reactive agents; cannot capture compounding closed-loop effects.

### [4] Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-To-End Autonomous Driving

- **Authors:** Xiaosong Jia, Zhenjie Yang, Qifeng Li, Zhiyuan Zhang, Junchi Yan
- **Year:** 2024
- **Published in:** Advances in Neural Information Processing Systems (NeurIPS 2024), Datasets and Benchmarks Track
- **Relevance:** First benchmark for multi-ability closed-loop evaluation of end-to-end driving in CARLA (220 routes, 44 interactive scenarios), with Driving Score, Success Rate, and granular skill assessment.
- **Use in this project:** Target closed-loop harness for the project's unevaluated long-term goal (CARLA runs, collision-before/after measurement).
- **Limitation:** Simulation-only (CARLA); sim-to-real gap remains.

### [5] nuScenes: A Multimodal Dataset for Autonomous Driving

- **Authors:** Holger Caesar, Varun Bankiti, Alex H. Lang, Sourabh Vora, Venice Erin Liong, Qiang Xu, Anush Krishnan, Yu Pan, Giancarlo Baldan, Oscar Beijbom
- **Year:** 2020
- **Published in:** Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2020), pp. 11621–11631
- **Relevance:** Foundational full-sensor-suite dataset (6 cameras, 5 radars, 1 lidar; 1000 scenes) underlying the project's data loader, sensor shapes `(6, 3, 256, 704)`, and detection box format.
- **Use in this project:** Dataset the `NuScenesLoader` targets; source of planned open-loop violation/false-positive calibration.
- **Limitation:** Recorded-log evaluation; straight-road heavy; no closed-loop reactivity.

---

## C. Traffic-Rule Formalization and Compliant Repair

### [6] No More Traffic Tickets: A Tutorial to Ensure Traffic-Rule Compliance of Automated Vehicles

- **Authors:** Matthias Althoff, Sebastian Maierhofer, Gerald Wursching, Yuanfei Lin, Florian Lercher, Roland Stolz
- **Year:** 2025
- **Published in:** Proceedings of the IEEE
- **Relevance:** Tutorial framing traffic rules as central, explicit guardrails; covers formalization of rules, fail-safe planning, reachability analysis, trajectory repair, robustness, runtime verification, and temporal logic.
- **Use in this project:** Conceptual backbone for the 63-rule catalog: rules as explicit central specifications plus a repair stage for violating trajectories.
- **Limitation:** Tutorial scope; rule encodings are formalism-agnostic rather than ASP-specific.

### [7] Traffic-Rule-Compliant Trajectory Repair via Satisfiability Modulo Theories and Reachability Analysis

- **Authors:** Yuanfei Lin, Zekun Xing, Xuyuan Han, Matthias Althoff
- **Year:** 2025
- **Published in:** IEEE (arXiv preprint arXiv:2412.15837)
- **Relevance:** Repairs (rather than replans) rule-violating trajectories by coupling Satisfiability Modulo Theories with set-based reachability analysis; demonstrated in high-fidelity simulators and the real world.
- **Use in this project:** The principled SMT-based alternative to the project's heuristic corrector (scale/shift/hold); frames the corrector's missing feasibility and termination guarantees.
- **Limitation:** SMT solving cost; real-time budget unproven at dense-urban scale.

### [8] Predictive Traffic Rule Compliance using Reinforcement Learning

- **Authors:** Yanliang Huang, Sebastian Mair, Zhuoqi Zeng, Matthias Althoff
- **Year:** 2025
- **Published in:** arXiv preprint arXiv:2503.22925 (intended for the IEEE International Conference on Intelligent Transportation Systems, ITSC 2025)
- **Relevance:** Replaces the actor network with a motion-planning module and trains the critic on traffic-rule robustness, using the critic output directly as the planner cost to predict and prevent violations beyond the planning horizon.
- **Use in this project:** The learning-based alternative to symbolic repair; informs the project's rule-priority and cost-design discussion.
- **Limitation:** Reward-shaping fragility; no hard guarantees on rule satisfaction.

---

## D. Neuro-Symbolic Safety

### [9] A Neuro-Symbolic Framework Combining Inductive and Deductive Reasoning for Autonomous Driving Planning

- **Authors:** Hongyan Wei, Wael AbdAlmageed
- **Year:** 2026
- **Published in:** arXiv preprint arXiv:2603.12421
- **Relevance:** Closest methodological prior: a Large Language Model dynamically extracts scene rules, an Answer Set Programming solver performs deterministic logical arbitration into discrete decisions, and a decision-conditioned decoder with a differentiable Kinematic Bicycle Model produces feasible trajectories (L2 0.57 m, collision 0.075% on nuScenes).
- **Use in this project:** Direct precedent for LLM-assisted rule extraction plus ASP arbitration over neural proposals; the KBM feasibility layer is explicitly what this project lacks.
- **Limitation:** Preprint under review; LLM rule-extraction soundness is itself unverified.

### [10] State-Dependent Conformal Perception Bounds for Neuro-Symbolic Verification of Autonomous Systems

- **Authors:** Thomas Waite, Yuang Geng, Trevor Turnquist, Ivan Ruchkin, Radoslav Ivanov
- **Year:** 2025
- **Published in:** Proceedings of Machine Learning Research, Volume 288, 2nd International Conference on Neuro-symbolic Systems (NeuS 2025), pp. 127–143
- **Relevance:** Synthesizes heteroskedastic (state-dependent) conformal perception-error bounds as the optimal interface between neural perception and symbolic control verification, yielding tighter end-to-end safety guarantees than time-uniform bounds.
- **Use in this project:** Formalizes the project's narrow neural↔symbolic interface as a perception-bound problem; justifies the honest claim that guarantees are statistical, not absolute.
- **Limitation:** Demonstrated on mountain-car scale; scaling to full driving stacks is open.

### [11] From Road to Code: Neuro-Symbolic Program Synthesis for Autonomous Driving Scene Translation and Analysis (Road2Code)

- **Authors:** J. Leung, G. Tong, P. S. Duggirala, P. Chakravarthula
- **Year:** 2025
- **Published in:** Proceedings of Machine Learning Research, Volume 288, 2nd International Conference on Neuro-symbolic Systems (NeuS 2025)
- **Relevance:** Translates real-world driving video and lidar into symbolic Scenic programs for the CARLA simulator via Large Language Model program synthesis, enabling scene translation, editing, and post-mortem analysis.
- **Use in this project:** Test-generation path for the ASP checker: real-to-sim symbolic scenarios for violation-regression and certification testing.
- **Limitation:** Synthesis fidelity bounds; generated programs require validation before use as ground truth.

---

## E. ASP Solver

### [12] Multi-shot ASP Solving with clingo

- **Authors:** Martin Gebser, Roland Kaminski, Benjamin Kaufmann, Torsten Schaub
- **Year:** 2019
- **Published in:** Theory and Practice of Logic Programming, Cambridge University Press, Volume 19, Issue 1, pp. 27–82
- **Relevance:** Definitive reference for clingo's grounding-plus-solving paradigm, multi-shot operational semantics, and Python API control capacities — the exact machinery (`Control`, `add`, `ground`, `solve`, `on_model`) the project's `TrajectoryChecker` invokes.
- **Use in this project:** Solver foundation for all 63 rules; explains per-check grounding cost and why the project uses a fresh `Control` with a pre-concatenated program blob.
- **Limitation:** General ASP systems paper; driving-specific encodings (integer-cm, squared speeds, horizon bounds) are the project's own contribution.

---

## Synthesis Table

| # | Paper | Year | Venue | Role in this project |
|---|-------|------|-------|----------------------|
| 1 | SparseDriveV2 | 2026 | arXiv (ECCV 2026) | Neural proposer SOTA |
| 2 | MomAD | 2025 | Proc. IEEE/CVF CVPR | Temporal-stability critique |
| 3 | NAVSIM | 2024 | Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks | Safety-aware open-loop eval |
| 4 | Bench2Drive | 2024 | Adv. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks | Closed-loop eval target |
| 5 | nuScenes | 2020 | Proc. IEEE/CVF CVPR | Dataset |
| 6 | No More Traffic Tickets | 2025 | Proc. IEEE | Rule-compliance framing |
| 7 | Trajectory Repair via SMT | 2025 | IEEE (arXiv) | Principled repair alternative |
| 8 | Predictive Compliance via RL | 2025 | arXiv (IEEE ITSC) | Learning-based alternative |
| 9 | Inductive + Deductive Planning | 2026 | arXiv | Closest ASP method |
| 10 | Conformal Perception Bounds | 2025 | Proc. Mach. Learn. Res. (NeuS) | Statistical guarantee framing |
| 11 | Road2Code | 2025 | Proc. Mach. Learn. Res. (NeuS) | Scenario synthesis for testing |
| 12 | Multi-shot ASP with clingo | 2019 | Theory Pract. Log. Program. (CUP) | Solver foundation |

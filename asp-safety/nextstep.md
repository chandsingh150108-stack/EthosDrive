# nextstep.md — P-25: CARLA Closed-Loop Safety Harness (Final Execution Plan)

**Status**: APPROVED PLAN, NOT STARTED (decisions locked 2026-10-21)
**Goal**: Convert the installed CARLA 0.9.15 server + verified smoke path into a reactive closed-loop harness that measures the ASP safety shield's behavioral safety (collision rate, violation rate/mile, intervention rate, min TTC, replan-fidelity) in scripted scenarios, run paired with the shield off vs on (heuristic AND qp corrector engines).
**Scope decisions (user, 2026-10-21)**: pilot-first matrix (5 scenarios × 1 seed × 3 engine states = 15 runs, extend to 3 seeds only if results justify); both corrector engines in the loop; ~3-week budget with per-phase ledger checkpoints.

---

## 0. Verified grounding (facts already in-hand; no re-probing needed)

### 0.1 Environment
- CARLA 0.9.15 stable at `C:\CARLA\WindowsNoEditor\CarlaUE4.exe` (outside repo), client `carla==0.9.15` in conda env `asp-safety` (python 3.10).
- Ports 2000/2001 free; no admin required; `-quality-level=Low -RenderOffScreen` launch flags proven in `experiments/carla_launch.py` (port-poll loop, 60×10 s).
- Default startup map Town10HD_Opt smoke-verified (`experiments/carla_smoke.py`): connect, spawn `vehicle.tesla.model3`, 6-camera-capable single `sensor.camera.rgb` at 704×256 fov 90, frame read via `np.frombuffer(img.raw_data).reshape(H,W,4)[:,:,:3]`, `SMOKE OK`.
- Server render ≈ 10–20 FPS (6-camera rig); closed-loop wall rate planner-bound ≈ 0.3–0.4 replans/s (CUDA 2.4 s/frame on RTX 4050, CPU 3.9 s/frame fallback).

### 0.2 Planner contract (already implemented; the bridge treats it as a black box)
- `SparseDrivePlanner` (`src/neural_planner.py`):
  - `predict(images, intrinsics, extrinsics, ego2global, timestamp, nav_cmd=None, ego_status=None)` — COLD; calls `reset_temporal()` every invocation; requires `images (6,3,256,704)`, `intrinsics (6,3,3) f32`, `extrinsics (6,4,4) f32` camera-to-vehicle, `ego2global (4,4) f32`, `timestamp float`.
  - WARM N-2 path: `begin_sequence()` → `predict_frame(...)` ×5 with per-frame `ego_status` (18,) via `data_loader.estimate_ego_status(prev_g, cur_g, dt)` → `end_sequence()`; `predict_frame` builds `img_metas` internally; never mix warm state across independent sequences.
  - `nav_cmd`: default `[1,0,0]`; head semantics UNVERIFIED (cmd sweep 2026-09-26) — KEEP DEFAULT in all closed-loop runs.
  - Returns `{'trajectory': (6,2) float32 at t=0.5..3.0 s, ego/lidar frame (x forward, y left); 'boxes': (N,7) [x,y,z,w,l,h,yaw]; 'scores': (N,)}`.

### 0.3 Checker/pipeline ingress (the only seam the bridge must satisfy)
- `TrajectoryChecker.check(traj, other_vehicles, map_data, traffic_lights, zones)`:
  - `traj`, each `other_vehicles[i]`: `(6,2)` float32, ego frame, 1-second surface (T=0 anchor + +1/+2/+3 s), UNITS: integer centimetres (SCALE_FACTOR=100), per `docs/conventions.md` (STEP_DT=1.0, squared distances, no floats in ASP).
  - `map_data` accepted keys (from `_add_map_data_to`, verified 2026-10-21): `lanes {lid: [{'x1','y1','x2','y2'},...]}`, `lane_width {lid: cm}`, `lane_direction {lid: {'dx','dy'}}`, `road_type {lid: 'urban'|'highway'}`, `speed_limit {lid}`/`min_speed_limit {lid}` (cm/step), `intersections {iid: {'x','y','radius'}}`, `lane_boundary [{'l1','l2','type'}]`, `lane_restriction`, `lane_type {lid: 'bicycle'|'protected_left'}`, `crosswalks [{'id','x1'..'y2'}]`, `sidewalks [...]`, `fixed_obstacles [...]`, `no_passing_zones`, `merge_lanes {lid: {'end_point'}}`, `conflict_points [{'id','x','y'}]`, `no_u_turn [lid]`, `no_turn_on_red {iid:[dir]}`, `controls_approach {iid: lid}`, `oncoming_lanes {lid: iid}`, `sightline`.
  - `traffic_lights {iid: {'color': 'red'|'green'|..., 'time': int, 'x','y','radius'}}`.
  - Bare-trajectory-only 50 km/h cap (squared 1,929,321 cm) is automatically gated when lanes are present — matches CARLA's absent posted limits.
- `SafetyLayerPipeline.run(...)` returns `original_trajectory`, `safe_trajectory`, `violations`, `is_safe`, `inference_time_ms`, `corrector_mode` stamp; `corrector_mode` is a constructor arg — for the bridge we do NOT instantiate two pipelines (6 GB VRAM cannot hold two model copies); instead one planner + one checker + two correctors sharing the checker (Phase P25-C detail, §5.1).

### 0.4 Correctors (both engines in the loop)
- `TrajectoryCorrector(checker, max_iterations=100)` — heuristic fixpoint, production default, map-aware, tail-rigid, stagnation break, fail-loud.
- `CorrectorQP(checker)` — pure-NumPy projected gradient, 3 solve/re-validate rounds, `bounds='full'|'caps'|'boxes'`, flagged heuristic fallback on failure (fallback never receives `include_constraints`; 2-tuple contract pinned by a test).
- Both minimize displacement vs the original plan; both stamped `corrector_mode` in JSON summaries.

### 0.5 What does NOT exist yet (the whole build)
- Reactive bridge (CARLA frames → planner → controller → sim), agent/light/map context extraction from CARLA, scripted scenarios, closed-loop metrics, paired-run driver, their docs and the report section. Section B of `docs/replay_eval_design.md` remains "future work (P-25)"; D3 in `docs/evaluation_design_asp_correction.md` is ⬜.

---

## 1. Architecture (one diagram, frozen)

```
CARLA server (sync mode, fixed dt=0.05 s, Low/OffScreen)
  ├─ 6× rgb sensors 704×256 (nuScenes-like mounts) ──┐
  ├─ collision sensor on ego                          │
  └─ actor state each tick (vehicles/walkers/lights)   │
                                                      ▼
                                          carla_bridge.py main loop
     every N ticks (replan, e.g. N=40 → 2 s sim):
       1. assemble planner inputs (image stack, K, C, ego2global, ts)
       2. warm predict_frame → plan (6,2) native 0.5 s grid
       3. carla_context.build(): agents' CV rollouts (×100 cm), light facts, zones
       4. engine dispatch:
            off        → executed = plan
            heuristic  → corrector.correct(plan, ...) → executed = safe_trajectory
            qp         → CorrectorQP.solve → executed
            failure    → revert to plan, flag intervention_failed (mirrors pipeline give-up semantics)
       5. controller.pursue(executed): pure-pursuit + PID → VehicleControl
       6. metrics tick: collision check, min-TTC, intervention bookkeeping
     between replans: reuse step 5 against a time-parameterized copy of the
     last executed plan (CLOCK NOTE: executed plan is consumed in sim time;
     its 0.5 s grid does not match the 2 s replan gap — the controller
     tracks the FAR end of the plan slower/faster per its duration capping,
     see §4.3 cadence rule)
```

**Cadence rule (decided, not open)**: replan every N ticks; at replan time the new plan's 6 waypoints at 0.5 s span 3 s but are consumed over up to N×dt sim seconds. The controller normalizes the plan to ITS consumption window: it parameterizes the plan by arc-time `s(t)` = plan progress advanced at controller-measured speed, i.e. chase the plan's end waypoint with target speed = plan end-segment speed; intermediate waypoints are lookahead references only. This avoids the 0.5 s grid being "played back too fast". Documented divergence from naive waypoint-queuing.

---

## 2. Deliverable file list (new files ONLY; zero changes to `src/`, `rules/`, tests)

| File | Phase | Purpose |
|---|---|---|
| `experiments/carla_probe.py` | P25-A | 6-cam rig + coordinate/projection sanity (one-shot, print + save png/npy) |
| `experiments/carla_launch.py` (EDIT) | P25-A | add `--map` arg + stale-process kill |
| `experiments/carla_context.py` | P25-B | sim actors/lights/zones → checker-ingress dicts (×100 cm), CV rollouts, culling mirror (range+conflict retention copied from `_cull_agents` ratios) |
| `experiments/carla_bridge.py` | P25-B/C | synchronous loop, warm queue, controller, collision sensor, replan dispatch {off,heuristic,qp}, logging |
| `experiments/carla_scenarios.py` | P25-D | S1–S5 scripted setups + actor choreography + seeds |
| `experiments/carla_metrics.py` | P25-C | per-run aggregation: collision rate/mile, violation rate/mile, intervention rate, min TTC, fidelity L2, wall/sim times |
| `experiments/run_closed_loop.py` | P25-B/C/D | CLI driver: one run → one JSON row |
| `experiments/carla_metrics.json` | P25-D | pilot matrix output (frozen artifact) |
| `docs/carla_harness.md` | P25-D | harness design + measured verdicts + caveats ledger |
| Doc edits at closeout | P25-D | `replay_eval_design.md` §B → implemented; `report.tex` D3 subsection; `futuretodoist.md` P-25 CLOSED; `session.md` S19 entries; `whatodo.md` day log |

Rules of engagement (per repo coding notes): no `TODO` comments; plan divergences go in `docs/carla_harness.md` "Deviations" section and the ledger (NOT `docs/rule_divergences.md` — that ledger is for RULE divergences). No lint/type changes; `pytest` must remain exactly 120/120 (no new src tests; experiment code self-verified via its own drivers).

---

## 3. Phase P25-A — Rig + coordinate probe (days 1–2)

### 3.1 Tasks
1. **Extend `carla_launch.py`**: optional `--map <name>` (default: don't load, use startup map), optional `--kill` (scan for `CarlaUE4.exe` processes, `taskkill /F /IM CarlaUE4.exe`).
2. **`carla_probe.py`** steps (run once, print report):
   a. Launch server via `carla_launch.py`; connect, `set_timeout(300)`, print versions/map.
   b. `settings.synchronous_mode=True; fixed_delta_seconds=0.05`.
   c. Spawn ego at spawn point 0; traffic manager NOT used in v1.
   d. Spawn 6 `sensor.camera.rgb` (704×256) at nuScenes-like mounts — FRONT (fov ~52°, +x forward), FRONT_LEFT (+yaw side, fov ~52°), FRONT_RIGHT (−yaw), BACK (yaw 180°, fov ~65°), BACK_LEFT, BACK_RIGHT; mounts ≈ 2.0 m height, ±0.10/±0.60 m lateral. fov values initial guesses; probe prints per-cam fov used.
   e. Derive per-cam intrinsics K from `fov_x` + image size: `fx = (W/2)/tan(fov_x/2)`, `fy = fx` (square pixels), `cx=W/2, cy=H/2`.
   f. Derive extrinsics C_i (camera→vehicle 4×4): compose `ego.get_transform() × cam.get_transform()` then apply the CARLA(UE4)→nuScenes handedness adaptation:
      - CARLA world: x fwd, y RIGHT, z UP (UE4 left-handed); CARLA camera image: +x right, +y down, +z into scene (standard CV).
      - nuScenes convention (what the planner expects): camera-to-ego matrices so CAM_FRONT's camera +x points along ego +x, +z along ego +z (devkit-calibrated analog). For the CARLA rig: build cam→CARLA-vehicle `T_v_c` from the sensor transform, then axis-flip `y→−y` (CARLA right → nuScenes left) to convert CARLA-vehicle into the planner's ego frame.
      - **VERIFICATION (the probe's job)**: spawn a `vehicle.tesla` actor 12 m ahead of ego, 2 m left; get its simulated 3D `bounding_box`, transformed into camera space using our `K, C`; project to CAM_FRONT pixel (u,v) with pinhole `u = fx·x/z + cx`; print computed pixel and confirm it is INSIDE the 704×256 image and plausibly positioned left-of-center; repeat for a right-side actor to confirm the left/right sign. Save the CAM_FRONT image + a `.npz` of (K, C_i×6, frame) for eyeball verification.
   g. Report VRAM: `torch.cuda.mem_get_info()` before/after one warm `predict_frame` vs server textures.

### 3.2 Deliverable/exit criteria
- `carla_probe.py` prints `PROBE OK` with: 6× (shape, dtype) stacks, one successful projection per test actor, VRAM headroom printed. If projection sign is wrong, fix the flip and re-run — sign errors here poison everything downstream.

### 3.3 Checkpoint (ledger)
- `session.md` S19 entry (probe results + flip decision + VRAM numbers); `futuretodoist.md` §5.2 P-25 gains `— P25-A PROBE DONE 2026-10-xx` line; `whatodo.md` day log line.

---

## 4. Phase P25-B — Bridge v1, shield OFF (days 3–5)

### 4.1 `carla_context.py` — sim state → checker-ingress dicts
Functions (signatures frozen here):

```python
def build_agent_context(world, ego_vehicle, ego_plan, horizon_s=3.0, dt=1.0) -> List[np.ndarray]:
    # Returns list of (6,2) float32 agent trajectories in EGO frame at the
    # 1-second ASP surface (T=0..5 capped at horizon 3 s per conventions),
    # ×100 cm, one entry per obstacle actor within 160 m with conflict
    # retention: vehicles/walkers CV-rolled out from their last two sim
    # positions (matches run_calibration's detected-context semantics);
    # parked series serialized as-is (rules handle); culling mirror:
    # range ≤160 m, drop non-conflicting (|Δy|>40 m AND receding); scripts
    # can force-include actor ids relevant to the scenario.

def build_light_context(world, ego_location, cluster_radius=60.0) -> Tuple[Dict, Dict]:
    # Cluster world.get_actors().filter('traffic.traffic_light') by trigger-
    # volume center distance ≤ cluster_radius → one pseudo-intersection per
    # cluster: iid=f"light_{min_token[:8]}", {x,y} mean of trigger centers,
    # radius = max distance from center to affected waypoints +10 m margin.
    # Per iid: state at the CURRENT tick; color map: Green→'green',
    # Yellow→'yellow', Red→'red', Off→skip iid entirely (absent facts = no
    # rule claim). Return (intersections, lights) dicts shaped exactly like
    # checker's map_data['intersections'] and traffic_lights kwargs
    # (already ×100 cm).

def build_map_context(world_map, ego_location) -> Dict:
    # Town10HD_Opt OpenDrive-driven: query waypoints around/ahead of ego;
    # emit current lane centerline segments (3–5 m spacing, ×100 cm),
    # lane_width, lane_direction from the waypoint transform,
    # road_type='urban', fixed_obstacles=NONE (v1).
    # OPTION B (retreat if noisy): ONLY current-lane centerline + width,
    # skip intersections/lights unless scenario 2 requires them.
    # Retreat decision documented in carla_harness.md.

def build_zone_context(scenario_cfg) -> Optional[List[Dict]]:
    # v1: None for all scenarios (no school/construction zones scripted).
```

- Units rule stated once, used everywhere: **all lengths ×100 to cm before export; all times already 1 s integer timesteps; no speed facts in v1**.
- **Self-verification harness (no server needed)**: `python experiments/carla_context.py --selftest` runs synthetic world-state fixtures through `build_agent_context`/`build_light_context` into a real `TrajectoryChecker` instantiation (rules, no GPU) and prints the `.check()` verdict + violations for one canned unsafe fixture. This is the context layer's acceptance test; run at end of P25-B.

### 4.2 `carla_bridge.py` — no-shield loop
Frozen responsibilities:

```python
class CarlaBridge:
    def __init__(self, cfg: BridgeConfig):   # dataclass: scenario, seed, engine,
        ...                                  # replan_ticks=40, max_minutes=6, map, camera fovs
    def setup(self):     # launch (or reuse) server, sync settings, spawn rig, sensors, scenario
    def run(self) -> RunLog:                 # the loop below
    def teardown(self):  # destroy sensors/actors, reset settings, optional keep-server
```

Main loop details (frozen):
1. `world.tick()` with `fixed_delta_seconds=0.05`, synchronous.
2. Every `replan_ticks` (default 40 → 2 s sim):
   - image stack: per-camera frame queues from `sensor.listen`; pull latest frames matching the same `frame` id; if any camera lags, wait ≤0.5 s or drop frame and run inference on the previous synchronized set — log drops.
   - `planner.begin_sequence()` ONCE per RUN (ego sequence continuous; NOT per replan), then `predict_frame(...)` with `ego_status = estimate_ego_status(prev_ego2global, cur_ego2global, dt)` on nuScenes-convention ego matrices built from `ego.get_transform()` (converted by the probe's flip).
   - engine dispatch per §5; `executed = safe_trajectory if engine != 'off' and safe else original, flagged`.
3. Controller between ticks (§4.3).
4. Collision sensor (`sensor.other.collision` attach ego): callback appends `(tick, other_actor.type, impulse_strength)`; per-tick manual overlap check on script actors (physics fallback).
5. `RunLog` rows per replan: `ts_sim, plan_raw, executed, violations, checks_ms, corrects_ms, map_iids, agents_count, speeds, collision_events, fix_flags`.
6. Run end: `end_sequence()`; `RunLog` → `experiments/runs/{scenario}_{seed}_{engine}.json` (run-id dedupe by timestamp).

### 4.3 Controller (pure pursuit + PID, ~120 lines, no external deps)
```python
def pursue(plan_world, speed_now, sim_dt):
  # Convert executed plan (ego frame) → world path via current ego2world.
  # End-anchored consumption (§1 cadence rule):
  #   target = plan[-1]; consumption window D = min(2 s, replan window)
  #   lookahead L = clamp( speed_now*0.6, 2.0, 6.0 m )
  #   alpha = yaw error to target point; steer = clamp( 2*wheelbase*sin(alpha)/L , -1, 1 )
  # speed: v_target = plan end-segment speed (|p5-p4|/0.5 s), clamped [0, 13.9 m/s]
  #        PID throttle = clamp( 1.2*(v_t - v_now)/max(v_t,4), 0, 1 ), brake if v_t<0.2
  # anti-chatter: steer deadband 0.02, throttle smoothing 0.2
```
- Wheelbase 2.85 m (model3). Constants in `BridgeConfig`; tune once on S1.

### 4.4 Verification & exit criteria (shield off)
- S1 free-drive run completes ≥600 m without exception; collision-free (S1 sanity).
- Replan cadence stable: mean replan wall ≤3.5 s; dropped-frame rate ≤5% logged.
- Controller tracks plan; fidelity metric (§6.5) computed and non-degenerate.
- `selftest` of `carla_context.py` green.
- Checkpoint ledger line (same pattern as P25-A).

---

## 5. Phase P25-C — Shield in the loop (days 6–7)

### 5.1 Dual-engine wiring (no second model load)
- Bridge holds ONE planner, ONE checker, and a dispatch table:

```python
correctors = {
  'heuristic': TrajectoryCorrector(checker),
  'qp':        CorrectorQP(checker),        # bounds='full'
}
executed = correctors[engine].correct(plan_1s_surface, other_vehicles=agents,
                                     map_data=map_facts, traffic_lights=light_facts,
                                     zones=zone_facts) if engine != 'off' else plan
# corrector kwargs semantics per src/corrector.py:321 (map_data/zones in kwargs);
# on Exception/give-up (mirror of pipeline semantics):
#   executed = original plan; fix_flags.append('revert_to_plan')
```

- NO change to `pipeline.py`; the pipeline's role is subsumed by direct corrector use inside the bridge (the bridge is the pipeline's closed-loop analogue, not a second pipeline; documented in `carla_harness.md`).
- Map facts recomputed each replan from current sim state (lights per tick, agent rollouts).

### 5.2 Intervention logging
Per replan row: `engine`, `is_safe_raw`, `violations_top`, `corrections_applied_count`, `corrector_ms`, `checker_ms`, `fix_flags`, `safe_vs_plan_delta_L2`.

### 5.3 Warm semantics inside one run (decided)
- `begin_sequence()` once at run start; `predict_frame` on successive replans (2 s sim apart). Divergence note (logged): warm queue sees coarser frame cadence (2 s) than nuScenes (0.5 s); accepted since `img_metas` carry real timestamps.

### 5.4 Exit criteria
- One full shielded run logs telemetry end-to-end: `original_trajectory`, `safe_trajectory`, `violations`, `fix_flags`, `corrector_ms` non-zero.
- Both engines produce a log (heuristic first; QP second — QP may flag fallback many times; expected, recorded).
- Checkpoint ledger line.

---

## 6. `carla_metrics.py` — metric definitions (frozen before P25-D so scenario runs aren't re-interpreted)

| Metric | Definition (exact) |
|---|---|
| collision_rate | (# runs with ≥1 collision sensor event) / (# runs) per engine group; per-mile = Σ collisions / Σ sim km of the group |
| violation_rate_per_mile | Σ over replans of #violation-entries surfaced at that replan / Σ sim km |
| intervention_rate | (#replans where is_safe_raw=False AND corrections applied without revert) / #replans; plus give-up rate = #replans with `revert_to_plan` / #replans |
| min_TTC | at each replan pre-shield: min over agents, over t ∈ {1,2,3 s} of (‖agent_t − plan_t‖₂ / closing speed), in meters (ASP handles only the ×100 cm surface); reported in sim seconds |
| fidelity_L2 | RMS between plan waypoints (0.5 s grid) and sim-ego actual positions at those relative times (m), averaged per run |
| sim_km | integrated ego arc length (m) / 1000 |
| wall vs sim | both recorded; cadence factor = replan wall time / replan sim time |

- Collisions with scripted scenario actors count with full weight; ambient actors (none in v1; scripts dispense traffic) count and are tagged when present.

---

## 7. Phase P25-D — Scenarios + pilot matrix (week 3)

### 7.1 Scenario specs (`carla_scenarios.py`)
| ID | Name | Setup | Scripted behavior | Rule families exercised |
|---|---|---|---|---|
| S1 | free drive | Town10HD_Opt, chosen spawn point, no ambient traffic | none | baseline; R1/R2/R12 incidental |
| S2 | red-light run | ego teleported 80 m before a signalized intersection approach; controlling light actor resolved by script and SET to red ~2 s before ego's stop line | light scripted red | R5.1 (R5.2/R5.3), R6.x |
| S3 | cut-in | NPC spawned 2 lanes left, 25 m ahead, same direction, slower | NPC changes into ego's lane at t=3 s | R4.1, R4.3, R3.x |
| S4 | jaywalker | pedestrian spawned to cross ego's path OUTSIDE a marked crossing, 15 m ahead | walker crosses at 1.2 m/s | R9.1, R11.1 |
| S5 | oncoming left-turn | oncoming NPC waits in opposing lane, starts left turn across ego's path at t=2.5 s | NPC turns across | R6.3, R7.2, R4.x |

- Each scenario: fixed coordinates (parsed once from real `carla.Map.get_waypoint` calls and recorded as literals), seed injection, own teardown. Determinism: synchronous fixed dt + scripted choreography → identical sim trajectory per seed; planner GPU nondeterminism remains the only stochastic element (flagged in metrics doc).

### 7.2 Pilot matrix (Pilot-first decision)
- 5 scenarios × 1 seed × 3 engine states {off, heuristic, qp} = **15 runs**.
- Run order: per scenario, [off, heuristic, qp] back-to-back (deterministic sim common across the triple).
- Est. wall: 3 min sim/run → 90 replans × ~2.4 s ≈ **3.6 min/run** ≈ **55 min total**; with startup/margins → **≤2 h**.
- Seed extension: only for scenarios showing an off-vs-on discrepancy → seeds 2,3 (adds ≤9 runs).

### 7.3 Exit criteria for P25-D
- `experiments/carla_metrics.json` frozen with 15 (or 15+9) rows + summary; per-run logs under `experiments/runs/`.
- ≥1 scenario where the shielded run measurably differs from shield-off (either direction is publishable evidence; the caution is symmetrical).
- Checkpoint entries + closeout docs.

---

## 8. Closeout (documentation + ledgers)

1. `docs/carla_harness.md`: architecture, cadence rule, fact-mapping rules, retreat decisions, Deviation log, frozen metric definitions, per-scenario verdicts, caveats (replan-gap honesty, domain shift, mini-level evidence only, scenario-actor curation).
2. `replay_eval_design.md` §B: header updated to "implemented (`experiments/carla_bridge.py` + scenarios); measured verdicts in §C pointer".
3. `report.tex`: new Experiments subsection "Closed-loop (D3)" with the metrics matrix + caveats; Limitations entry replaced by the measured-in-sim status where applicable.
4. `futuretodoist.md`: P-25 CLOSED with pointer to the metrics artifact; §5 Open line updated (P-23 remains the sole open long-tail item).
5. `session.md`: S19 entries per checkpoint (P25-A/B/C/D) + closeout day summary; `whatodo.md` day log mirroring.
6. `README.md`: experiments list gains the three new harness files.
7. Verification gates: `pytest tests/ -q` → exactly `120 passed` (run twice to confirm no drift); `python tests/test_all.py` → `7/7 PASS`; outputs recorded in the closeout checkpoints.

---

## 9. Risks & declared retreats (decision rules, not just warnings)

| Risk | Trigger | Retreat (in order) |
|---|---|---|
| VRAM squeeze (server 6-cam + planner, 6 GB total) | probe VRAM headroom <1.5 GB or OOM | 1) drop BACK cams → 3-cam rig (documented divergence); 2) `image_size 384×160` with recalculated K; 3) planner on CPU (3.9 s/frame; cadence ~0.26/s; saturating but evidence remains) |
| CARLA-imagery planner degradation (domain shift) | fidelity_L2 degenerates vs nuScenes baseline | ORACLE-CONTEXT arm: sim-GT agent trajectories in the shield regardless of planner detections (flagged arm, still shield-honest); planner-quality arm reported separately |
| Light→intersection clustering too noisy | S2 impossible without facts | restrict S2 to ONE hand-picked controlling light; emit ONLY that iid (documented divergence) |
| Corrector give-ups in-loop (scene-8 whack-a-mole class) | `revert_to_plan` rate >60% on any scenario | report as verdict (the known P-23 gap); do NOT raise corrector iterations; retreat = switch that scenario's shield engine to the OTHER corrector and log |
| Controller budget vs plan window | sim km <200 m on S1 | raise `replan_ticks` to 60/80 (3–4 s sim gap) and re-test; wall cost unchanged |
| Server drift across identical seeds | identical-seed runs diverge | synchronous dt enforcement + seed re-assert + agent churn check each replan in `RunLog` |
| 2 s replan gap means shield isn't causal between replans (lag-to-act gap) | — | DOCUMENTED LIMITATION, not fixed in v1: corrections were issued against the OLD plan's future; scenarios scripted with ≥2 s lead so the shield's window covers the scripted event's onset; this is the runbook's honesty paragraph |

---

## 10. Order of execution (final; sequential)

```
P25-A probe (day 1-2)      → checkpoint (blocked if flips found)
P25-B bridge v1 OFF (3-5)  → exit S1 lap + selftest          → checkpoint
P25-C shield wiring (6-7)  → both engines tested on S1       → checkpoint
P25-D scenarios (8-10)     → S1..S5 scripted, verified one-by-one
pilot matrix (day 10-11)   → 15 runs, frozen artifact
docs + ledgers (day 12)    → closeout gates + final verification
```

Budget: ~2 weeks of calendar at 3–4 h/day inside the approved 3-week envelope, including seed-extension slack and the two hazards most likely to bite (light-clustering accuracy, VRAM).

---

*Plan authored and approved 2026-10-21 (session S18 closure follow-up); execution owner: P-25 workstream. This file is the frozen contract; deviations go to `docs/carla_harness.md` Deviations + ledger, not here.*

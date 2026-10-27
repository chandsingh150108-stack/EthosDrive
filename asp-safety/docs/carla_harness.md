# CARLA Harness — Deviations

Deviations from plan contracts for the P-25 CARLA closed-loop workstream.
This ledger carries harness/experiment divergences; rule divergences stay
in `docs/rule_divergences.md`. Nothing here changes `src/`, `rules/`, or
`tests/`; frozen files keep their contract (nextstep.md 8).

## Architecture + cadence rule (P25-B/C/D)

`experiments/carla_bridge.py` runs the closed loop in synchronous mode:
6-camera rig + collision sensor on a Tesla model3 ego -> planner
(SparseDrive neural plan, or the lane plan engine `--plan-engine lane`
used by the frozen matrix because the neural plan is OOD on CARLA
imagery) -> ASP checker -> corrector (engine `off`/`heuristic`/`qp`)
-> pure-pursuit controller -> `world.tick()`. Scenarios
(`experiments/carla_scenarios.py`) script enemies via
`resolve_setup/tick/cleanup` hooks; `run_closed_loop.py` is the CLI;
`carla_metrics.py` aggregates run JSONs.

- World settings: `synchronous_mode=True`, `fixed_delta_seconds=0.05`.
- **Cadence rule**: replan every `replan_ticks=40` ticks = 2 s sim
  (`nextstep.md` 4.3). Between replans the last (possibly corrected)
  plan is executed by the controller; the shield is therefore NOT
  causal inside the 2 s gap — a documented limitation, mitigated by
  scripting every hazard with >=2 s lead so the shield's window covers
  the event onset.
- Setup sweeps leftover actors (id >= 1000, not `traffic.*`/`spectator`);
  cleanup restores async mode and resets all map lights.

## Fact mapping (P25-B/D)

- **Agents**: CARLA actors (vehicles/walkers) -> checker `other_actors`
  with CARLA-vehicle-frame positions (x forward, y left via `F`), 1.0 s
  timestep.
- **Map**: `world.get_map().get_waypoint` -> lane facts (`in_lane`,
  `lane_center`, speed limits); route ahead of ego only.
- **Lights**: only signalized approaches contribute — a light is kept
  when its **own trigger center** is within 30 m of a stop-line
  waypoint ahead of ego (else the CARLA light actor's stop lines are
  map-wide and pollute facts). Cluster radius = min member stop ring
  + 10 m; red facts span the horizon (`times: [0..5]`). Details in
  "Traffic-light facts" below.
- **Ego plan**: columns are (forward, lateral) for checker facts;
  SparseDrive's native (lateral, forward) is swapped in `_replan`.
- Images flipped vertically (CARLA BGRA top-down vs planner BGR
  bottom-up); intrinsics K from per-camera FOV, extrinsics composed
  `C_i = S @ T_v_c @ F` (verified by projection sanity, P25-A).

## Frozen metric definitions (P25-C, `experiments/carla_metrics.py`)

- `collision_rate_pct` — share of runs whose ego collided.
- `violations_per_mile` — ASP findings / driven miles, mean per run.
- `intervention_rate` — share of replans where the shield materially
  changed the plan (`safe_vs_plan_l2 > 0` or `revert_to_plan` flag).
- `replan_wall_ms_mean` — per-replan wall time (planner + checker +
  corrector); includes sync-mode decision pauses (see "Replan wall
  semantics").
- `frame_drop_rate` — dropped sensor synchros / replans.
- Aggregate is a mean over completed runs; frozen artifact:
  `experiments/carla_metrics.json` (15 runs, 2026-10-22).

## Per-scenario verdicts (frozen 15-run matrix, 2026-10-22)

Seed 1, Town10HD_Opt, plan engine `lane`, replan 40 ticks, dt 0.05 s.
Aggregate: collision 13.3% (2/15, both off arms), intervention 0.518,
replan wall mean 1395 ms, frame drops 0, give-up 0 on every arm.

| Run | Engine | Status | Collided | sim km | Notes |
|---|---|---|---|---|---|
| S1 off 161659 | off | distance_reached | no | 0.60 | clean free drive |
| S1 heur 175238 | heuristic | distance_reached | no | 0.60 | 31/91 replans corrected |
| S1 qp 175631 | qp | distance_reached | no | 0.60 | 43/103 corrected |
| S2 off 153537 | off | distance_reached | no | 0.60 | runs the pinned-red junction (95 findings) |
| S2 heur 155403 | heuristic | wall_cap | no | 0.003 | holds at the red box edge |
| S2 qp 160155 | qp | wall_cap | no | 0.003 | holds at the red box edge |
| S3 off 163204 | off | stalled | **yes** | 0.034 | rear-ends the cut-in NPC |
| S3 heur 164903 | heuristic | wall_cap | no | 0.025 | follows/holds behind the NPC |
| S3 qp 165554 | qp | wall_cap | no | 0.025 | survives |
| S4 off 171926 | off | distance_reached | no | 0.60 | kinematic walker: no collision event (limitation) |
| S4 heur 172553 | heuristic | wall_cap | no | 0.469 | waits out the jaywalker |
| S4 qp 173218 | qp | wall_cap | no | 0.518 | waits out the jaywalker |
| S5 off 173559 | off | stalled | **yes** | 0.028 | collides in the turn conflict |
| S5 heur 174229 | heuristic | wall_cap | no | 0.016 | holds |
| S5 qp 174851 | qp | wall_cap | no | 0.016 | holds |

## Caveats (evidence honesty)

- **Single seed**: the frozen matrix is n=15 at seed 1; collision-rate
  confidence intervals are wide. Seed extension (seeds 2-3) is planned.
- **Kinematic scripted actors**: VRUs and NPCs are kinematic, so the S4
  off arm drives through the walker with no collision event; S4's
  evidence is shield-hold behavior, not collision rate. S3/S5 NPCs do
  collide (vehicles are physics-impulsed).
- **Lane plan engine drives the ego**: the neural planner is OOD on
  CARLA imagery (see "Neural planner state"); neural plans are logged
  unshielded in the run JSONs.
- **Replan-gap non-causality**: corrections target the old plan's
  future inside the 2 s gap (documented in nextstep.md 9, echoed in
  "Replan wall semantics").
- **Mini-level + in-sim evidence only**: no deployment claim follows.
- Wall caps on hazard scenarios (S2/S3/S4/S5 shielded arms) mean
  "survived the scripted hazard", not "drove the route".
- Demo/visible-window runs live in `experiments/runs/_view_demo/`,
  outside the frozen glob; the Epic window is infeasible on this 6 GB
  rig (backlog P-26).

## Axis conventions (P25-B, S18 closure)

- **Plan columns are (lateral, forward), swapped in `_replan`**
  (`plan_native[1::2][:3, ::-1]`): SparseDrive returns (lateral, forward);
  the bridge reorders to (forward, lateral) for the lane plan path but
  keeps checker plan facts in (forward, lateral).
- **Camera images flipped vertically** (`arr[:, :, 2::-1]`): CARLA BGRA
  buffer rows come top-to-bottom; the planner expects BGR bottom-up.
- **CAM_CONFIGS are 6-tuples** (name, yaw, fov, dx, dy, dz) — the plan
  showed 5-tuples.
- **Steer sign calibrated**: `steer_sign = -1.0` (verified in
  `experiments/carla_sign_test.py`, port 2001 test pattern).

## Controller constants (measured)

wheelbase 2.85 m; lookahead 0.6 * v clamped 2-6 m; steer_gain 2.0;
steer_deadband 0.02; throttle_gain 1.2; throttle_smooth 0.2;
v_target_cap 13.9 m/s.

## Neural planner state (P25-B evidence)

- SparseDrive inference meets the latency budget on closed-loop runs
  (exec_ms mean ~2629 < 3500) but is out-of-distribution on CARLA
  imagery: plans curve left-laterally (1.06 -> 7.65 m over 3 s) and the
  ego crashed into a static building at 11.05 s (run
  `S1_1_off_20261022_140825.json`), then stalled at 47 m.
- Decision: the lane engine (`--plan-engine lane`) drives the closed-loop
  scenarios; neural plans remain logged when requested. Documented as a
  deliberate deviation from the plan-level assumption that the neural
  plan is the default planner upstream of the shield.

## Junction overrun (P25-B root cause)

Track-junction waypoint chain walks were verified clean manually; the
crash at (41.2, 72.1) came from the controller outrunning the plan:
junction-speed plan horizon (10.5 m at 3.5 m/s) is consumed between 2 s
replans (5.8 m per 2 s), look-ahead latched past plan end, target flipped
backward (alpha -126 deg), vehicle steered into vegetation. Fixes:
(a) `_control_tick` extends the target beyond plan end along the final
segment direction; (b) junction speed cap raised 3.5 -> 5.0 m/s in
`_lane_plan`.

## Traffic-light facts (P25-D)

- CARLA 0.9.15 stress: `TrafficLight` exposes the `trigger_volume`
  property but NOT a `get_trigger_volume()` accessor; `carla_context`
  fell back to that accessor and silently returned zero light facts.
  Fixed to read the property.
- Cluster radius: CARLA light actors reference stop waypoints across the
  map (tl 9 stop lines 74+ m from its own trigger center). Max-over-all
  stop distances inflated the red box to ~130 m, painting "inside the
  red junction" findings from far out and making hold-position
  corrections unfixable (100-iteration give-ups on S2). Radius now =
  min over cluster members of that member's own stop-line ring +10
  (observed 41.7 m at the Town10HD cluster).
- Light-state facts span the prediction horizon (`times: [0..5]`), not
  just T=0, so a red reading constrains the whole plan slice.
- S2 spawn selection requires a stop waypoint within 30 m of the light's
  own trigger center (a genuinely signalized approach), then freezes the
  whole junction cluster red: per-tick `set_state(Red)` PLUS a one-time
  `set_red_time(1e6)` + `reset_group()` pin, because CARLA group timers
  otherwise leave red after ~4 min and drop yellow/green flickers.
- Every scenario cleanup resets all map lights (`set_green_time(20)`,
  `set_yellow_time(3)`, `set_red_time(20)`, `reset_group()`) to keep
  cross-run light state deterministic.
- S1/S3/S4/S5 spawn picking skips any candidate whose 130 m route enters
  a signalized cluster (45 m radius around trigger centers), so scripted
  events dominate the evidence instead of light phase timing.
- Consequence: S2's red duration is realistic (holds through the full
  run); the off arm drives through red (81 red_light findings), both
  shield arms hold at the box edge with give_up_rate 0.0.

## Corrector behavior (P25-D, experiments-side only)

- `longitudinal_clearance` remapped from the collision family (lateral
  tail shift) to the follow family (slow down) at `carla_bridge` import
  time: a lateral shift cannot fix rear-end gaps and ping-ponged to
  100-iteration give-ups on cut-in runs.
- Both correctors constructed with `on_failure='brake'`: fixpoint
  exhaustion now returns an emergency-hold plan instead of raising and
  being caught as `revert_to_plan` (raw plan execution).
- Heuristic corrector gets `lateral_shift_m=-2.0` (cut-ins come from the
  left lane: the +2.0 default shifts INTO the intruder).
- S1 heuristic give-ups observed earlier (12/62 revert rows) came from
  the old revert path plus a leaked-NPC collision; after the fixes the
  heuristic arm passes S1 with give_up_rate 0.0.

## Harness robustness (P25-D)

- Setup sweeps leftover actors (leaked NPCs/walkers from aborted probe
  scripts sat on the shared spawn routes and poisoned later runs with
  false stalls/collisions).
- Stall detector skips windows that include planned holds (`hold_ticks`
  gate): resuming after a red-light / jaywalker wait would false-positive
  the zero-progress check at the resume boundary.
- Known limitation: kinematic VRUs (the S4 walker and S3/S5 NPC vehicles)
  do not generate physical collision events against the physics ego; the
  off arm for S4 drives through the walker without a collision event.
  S4's evidence is therefore shield-hold behavior rather than
  collision-rate.

## Choreography (P25-D)

- S3 cut-in re-parameterized: NPC spawns 30 m ahead in the adjacent
  lane, alternates its own lane ~1 s, then swings into ego's lane
  touching down at ego-arc 26 m at t ~ 2.4 s (~1 s before the 7 m/s ego
  arrives). The off arm rear-ends it (~34 m); the shield arms survive
  (full-tracked follow / emergency hold).
- S5 oncoming left-turn NPC: spawns 15 m before the turn start, begins
  the left turn at t=2.5 s, speed 6 m/s.
- S4 walker: start offset 5.5 m from the path center point at arc 15,
  crosses the full road at 1.2 m/s.

## Replan wall semantics

Sync mode freezes simulation time during replans: a 5.3 s corrector
fixpoint (S3/S5 adversarial geometry) charges wall cost but zero sim
time; the reported `replan_wall_ms_mean` includes those decision pauses.

## Visible window (S23)

- `experiments/carla_launch.py` gained `--quality {Low,Epic}`,
  `--windowed` (visible window 1280x720, replaces `-RenderOffScreen`)
  and `--stream-pool N` (`-ExecCmds="r.Streaming.PoolSize N"`).
  Defaults stay Low/offscreen so every harness caller is unchanged.
- **Epic is infeasible on this rig**: four boot crashes during map load
  before the driver update (Epic windowed x2, Epic windowed + pool
  trim, Epic offscreen — last one lived to ~110 s). Retried 2026-10-23
  with NVIDIA driver 617.42: the early boot crash is fixed (Epic
  windowed now survives ~2-3 min) but it dies during full map load
  with free VRAM down to 65 MiB — the complete Epic texture/shadow set
  needs more than 6 GB. Epic + `r.Streaming.PoolSize 1200` died within
  100 s (trim too late in the load order). 0.9.15 has no intermediate
  tier, so the ladder resolves to **windowed Low** (backlog P-26).
- `experiments/carla_vram_probe.py` gates the shared GPU: server +
  window + 6-cam rig -> planner load -> warm `predict_frame` vs
  >=1.5 GB free. Windowed Low: PASS, 4.03 GB post-predict.
- Ego-follow spectator: `BridgeConfig.view_ego` (`--view` on
  `run_closed_loop.py`) repositions the spectator 12 m behind / 7 m
  above the ego, pitch -25 deg, each tick. Rendering convenience only
  — no physics or metric effect.
- Live demo pair written to `experiments/runs/_view_demo/` (outside the
  frozen matrix glob): S3 off collided at sim 10.15 s; S3 heuristic
  wall_cap at 0.025 km, collided=False, give_up 0.0.

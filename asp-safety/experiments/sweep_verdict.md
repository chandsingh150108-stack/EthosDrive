# P-5 threshold sweep verdict (2026-09-26)

GPU-free sweep over dumped 10-scene trajs (`sweep_results.json`).
Agent context (detections->trajectories) included; without it every
.lp threshold row is flat (no agents in the harness).

## Decision: min_score 0.3 -> 0.5, everything else unchanged

| min_score | GT-FP scenes | pred-hit scenes | GT collision | pred collision |
|---|---|---|---|---|
| 0.1 | 9 | 9 | 32 | 25 |
| 0.3 (old) | 8 | 9 | 27 | 16 |
| 0.5 (new) | 5 | 9 | 17 | 8 |
| 0.7 | 4 | 9 | 4 | 4 (signal loss) |

0.5 is the knee: GT-FP 8->5 scenes, pred hits steady (9 scenes,
collision counts roughly halved but present). 0.7 keeps dropping GT
but eats true pred signal.

max_agents: flat 10/20/40 (keep 20). All .lp thresholds flat
(bubble 3-6 m, long gap, lat clear, default cap, wrong-way min,
worker clearance): no knees on this data -- colliding boxes sit
well inside 3 m (parked cars lining the log path), so bubble size
is not the discriminator. Keep all defaults.

## Residual GT-FP at 0.5 (5 scenes, understood)

- lane_keeping 5: geometry residuals (off-map starts, edge
  departures; see N-1 ledger).
- unsafe_lane_change_gap 5 / uncontrolled_priority 3-4: genuine log
  behaviors (lane changes into tight gaps, shared intersections)
  that conservative rules flag by design.
- collision_avoidance ~17: proximity < 5 m to roadside boxes, mostly
  parked cars. The bubble cannot distinguish parked from live
  traffic (boxes carry no class in our contract) -- logged as future
  work, not a threshold problem.

## Follow-ups (not this track)

- Parked-vs-live distinction needs box classes in the contract.
- Fix-rate under agent context is 0.50 with 12 s/scene corrector
  churn (P-6 fallback policy).
- Re-sweep after N-2 warm predictions change the pred distribution.

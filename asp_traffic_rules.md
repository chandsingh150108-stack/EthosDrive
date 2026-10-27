# ASP Traffic Rule Catalog for Autonomous Vehicle Safety Layer

## Conventions Used Throughout This Document

- **Coordinate system:** ego-frame, integer, scaled ×100 (1 unit = 1 cm). All arithmetic is integer-only; squared distances replace Euclidean distance so no `sqrt()` is required.
- **Timesteps:** `T` ranges `0..5` (6 total), matching the planner's prediction horizon.
- **Core facts** (given): `pos(V,T,X,Y)`, `lane(L,X1,Y1,X2,Y2)`, `light_state(I,Color,T)`, `intersection(I,X,Y,R)`, `speed(V,T,S)` (S = cm/timestep, integer).
- **Extended facts** (introduced per-rule as needed, since the base schema can't cover pedestrians, weather, signage, etc. on its own): these are called out explicitly in each rule's "Input Facts Required" so the catalog is self-contained. Naming is kept consistent across rules (e.g. `pedestrian(P,T,X,Y)` is reused everywhere a pedestrian position is needed).
- **`ego`** denotes the planned vehicle; other `V` values denote tracked agents from perception.
- **Helper predicates** referenced across multiple rules are defined once here and reused rather than re-derived:
  - `dist2(V1,V2,T,D) :- pos(V1,T,X1,Y1), pos(V2,T,X2,Y2), V1!=V2, D = (X1-X2)*(X1-X2)+(Y1-Y2)*(Y1-Y2).`
  - `heading(V,T,DX,DY) :- pos(V,T,X,Y), pos(V,T+1,X2,Y2), DX=X2-X, DY=Y2-Y.`
  - `on_segment(X,Y,X1,Y1,X2,Y2) :- (X2-X1)*(Y-Y1) - (Y2-Y1)*(X-X1) == 0` (collinearity, used as the core of in-lane tests, combined with bounding-box checks).
- **Priority levels** follow: *Critical* (imminent collision/legal red line) > *High* (clear violation, corrected every cycle) > *Medium* (soft constraint, corrected when it doesn't conflict with higher priority) > *Low* (comfort/quality, advisory).
- **ASP Complexity** and **Performance Impact** are estimates for grounding under Clingo with a perception horizon of ~20 tracked agents and 6 timesteps.

Rules are numbered `R<category>.<index>` for cross-referencing in "Dependency" fields.

---

## 1. Speed Regulation

### R1.1 — Urban Speed Limit Compliance
- **Description:** Ensures the planned trajectory never exceeds the posted speed limit on urban road segments.
- **Traffic Law Basis:** Uniform Vehicle Code §11-801 / local urban speed ordinances (typically 25–35 mph / 40–56 km/h).
- **Input Facts Required:** `pos(V,T,X,Y)`, `speed(V,T,S)`, `lane(L,X1,Y1,X2,Y2)`, `road_type(L,urban)`, `speed_limit(L,Limit)` (cm/timestep).
- **Violation Condition:**
  ```
  violation(urban_speed, ego, T) :- speed(ego,T,S), in_lane(ego,T,L), road_type(L,urban), speed_limit(L,Limit), S > Limit.
  ```
- **Detection Logic:** For each timestep, resolve which lane segment `ego` occupies via `in_lane/3` (nearest-segment + perpendicular-distance test), fetch that lane's limit, compare integer speed directly — no unit conversion needed since both are pre-scaled to cm/timestep.
- **Correction Strategy:** Rescale the velocity vector between `T` and `T+1` so resulting `S` equals `Limit`, holding heading direction fixed; propagate the reduction smoothly backward one timestep to avoid a discontinuous braking spike (see R12.1).
- **Priority Level:** High
- **Example Scenario:** Planner predicts ego accelerating to 45 mph through a 30 mph urban corridor after a light turns green; rule caps the predicted speed profile before execution.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R12.1 (smooth deceleration), R1.3 (variable zones can lower `Limit` further)
- **Test Case:** `lane(l1,0,0,0,5000). road_type(l1,urban). speed_limit(l1,1340).` (30 mph≈1340 cm/timestep at 10Hz) with `pos(ego,3,0,1500,X,Y)` implying `speed(ego,3,1800)`.

### R1.2 — Highway Maximum Speed Limit
- **Description:** Caps predicted speed on highway-classified segments at the posted maximum.
- **Traffic Law Basis:** State highway speed statutes (e.g. 65–80 mph limits on interstates).
- **Input Facts Required:** `pos(V,T,X,Y)`, `speed(V,T,S)`, `lane(L,...)`, `road_type(L,highway)`, `speed_limit(L,Limit)`.
- **Violation Condition:**
  ```
  violation(hwy_speed, ego, T) :- speed(ego,T,S), in_lane(ego,T,L), road_type(L,highway), speed_limit(L,Limit), S > Limit.
  ```
- **Detection Logic:** Identical structure to R1.1, differentiated only by `road_type`; kept as a separate rule because highway corrections must preserve merge/overtake feasibility (larger safety margins, see R8) rather than braking hard.
- **Correction Strategy:** Cap the terminal velocity at `Limit`; if the delta is large, spread the reduction across all 6 timesteps rather than the last one, respecting max deceleration (R12.1).
- **Priority Level:** High
- **Example Scenario:** Predicted trajectory shows ego at 90 mph merging onto an interstate with a posted 70 mph limit.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R1.4 (minimum highway speed — the two must not produce an infeasible band), R12.1
- **Test Case:** `road_type(l2,highway). speed_limit(l2,3130).` with predicted `speed(ego,5,4020)`.

### R1.3 — Variable Speed Zones (School / Construction / Weather-Triggered)
- **Description:** Applies a temporarily reduced limit when the ego trajectory overlaps a school zone, active construction zone, or a weather-flagged segment.
- **Traffic Law Basis:** School zone statutes (typically 15–20 mph during posted hours), work-zone speed reduction laws.
- **Input Facts Required:** `pos(V,T,X,Y)`, `speed(V,T,S)`, `zone(Z,X1,Y1,X2,Y2,Type)`, `zone_active(Z,T)` (time-of-day/flagger-derived), `zone_speed_limit(Z,Limit)`.
- **Violation Condition:**
  ```
  violation(variable_zone_speed, ego, T) :- speed(ego,T,S), pos(ego,T,X,Y), zone(Z,X1,Y1,X2,Y2,_), inside_zone(X,Y,X1,Y1,X2,Y2), zone_active(Z,T), zone_speed_limit(Z,Limit), S > Limit.
  ```
- **Detection Logic:** `inside_zone/6` is an axis-aligned (or rotated-rect via cross-product) bounding test against the zone polygon; `zone_active` gates on schedule facts (school hours) or a perception flag (cones/flaggers detected) so the rule only fires when the reduction actually applies.
- **Correction Strategy:** Overwrite the effective `Limit` used in R1.1/R1.2 with `zone_speed_limit(Z,Limit)` for any timestep inside the zone (min of the two if zones overlap), then reapply the R1.1 correction procedure.
- **Priority Level:** Critical (school zones with detected pedestrians escalate to Critical; otherwise High)
- **Example Scenario:** Trajectory passes through a marked school zone at 3:15pm dismissal time at 28 mph against a posted 15 mph zone limit.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium (zone polygon tests each timestep)
- **Dependency:** R1.1, R9.2 (school zone pedestrian behavior), R14.3 (visibility)
- **Test Case:** `zone(z1,1000,0,3000,2000,school). zone_active(z1,3). zone_speed_limit(z1,670).` with `pos(ego,3,2000,1000)`, `speed(ego,3,1250)`.

### R1.4 — Minimum Speed on Highways
- **Description:** Flags predicted speeds below the statutory minimum on highway segments where slow driving itself is unsafe.
- **Traffic Law Basis:** UVC §11-802 minimum speed regulation — impeding normal traffic flow is itself a violation absent an emergency.
- **Input Facts Required:** `speed(V,T,S)`, `in_lane(V,T,L)`, `road_type(L,highway)`, `min_speed_limit(L,MinLimit)`, `emergency_flag(V,T)`.
- **Violation Condition:**
  ```
  violation(min_hwy_speed, ego, T) :- speed(ego,T,S), in_lane(ego,T,L), road_type(L,highway), min_speed_limit(L,MinLimit), S < MinLimit, not emergency_flag(ego,T).
  ```
- **Detection Logic:** Straightforward integer comparison; the `not emergency_flag` guard exempts legitimate slow-downs (breakdown, obstacle in path) from being "corrected" upward into danger.
- **Correction Strategy:** If safe (clear lane ahead per R4 checks), accelerate up to `MinLimit`; if not safe, suppress the correction and instead trigger a lane-change or shoulder recommendation rather than forcing acceleration into a hazard.
- **Priority Level:** Medium
- **Example Scenario:** Planner produces an overly conservative trajectory crawling at 35 mph on a highway with a 45 mph minimum, with clear road ahead.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R1.2 (must stay below max while raised), R4.2 (longitudinal clearance before accelerating), R11.1 (emergency braking overrides this)
- **Test Case:** `road_type(l2,highway). min_speed_limit(l2,2010).` with `speed(ego,2,1560)`, no `emergency_flag(ego,2)`.

### R1.5 — Reduced Speed Near Pedestrians/Cyclists
- **Description:** Requires a speed reduction whenever a pedestrian or cyclist is detected within a lateral proximity band of the ego path, independent of posted limits.
- **Traffic Law Basis:** General duty-of-care / vulnerable-road-user statutes present in most jurisdictions (e.g. "reduce speed when a hazard exists").
- **Input Facts Required:** `pos(ego,T,X,Y)`, `speed(ego,T,S)`, `vru(P,T,X2,Y2,Type)` (Type = pedestrian/cyclist), `vru_speed_cap(Type,Distance,Cap)` lookup table.
- **Violation Condition:**
  ```
  violation(vru_proximity_speed, ego, T) :- speed(ego,T,S), pos(ego,T,X,Y), vru(P,T,X2,Y2,Type), D2 = (X-X2)*(X-X2)+(Y-Y2)*(Y-Y2), D2 <= 250000, vru_speed_cap(Type,close,Cap), S > Cap.
  ```
  (250000 cm² ≈ 5 m radius, threshold table keyed by banded distance rather than a continuous function to stay integer-only.)
- **Detection Logic:** Squared-distance bucket lookup (`close` / `medium` / `far`) against a small static cap table per VRU type, avoiding any sqrt or floating division.
- **Correction Strategy:** Cap speed to the banded value for the closest VRU at each timestep; if multiple VRUs are present, take the minimum cap across all of them.
- **Priority Level:** Critical
- **Example Scenario:** A cyclist is detected 4 m to the side of the planned path at a predicted ego speed of 35 mph; rule forces a reduction to the "close" band cap (e.g. 15 mph).
- **ASP Complexity:** Medium
- **Performance Impact:** Medium (pairwise distance check per VRU per timestep)
- **Dependency:** R9.1 (crosswalk yielding), R9.4 (blind-spot detection), R4.1 (lateral clearance)
- **Test Case:** `vru(p1,2,1200,800,cyclist). vru_speed_cap(cyclist,close,670).` with `pos(ego,2,1000,800)`, `speed(ego,2,1560)`.

## 2. Lane Discipline

### R2.1 — Lane Keeping (Boundary Containment)
- **Description:** Ensures every predicted position stays within the boundaries of a valid lane rather than straddling or exiting it.
- **Traffic Law Basis:** General "drive within a single lane as nearly as practicable" statutes (UVC §11-309).
- **Input Facts Required:** `pos(ego,T,X,Y)`, `lane(L,X1,Y1,X2,Y2)`, `lane_width(L,W)`.
- **Violation Condition:**
  ```
  violation(lane_keeping, ego, T) :- pos(ego,T,X,Y), not in_any_lane(X,Y).
  in_any_lane(X,Y) :- lane(L,X1,Y1,X2,Y2), lane_width(L,W), perp_dist2(X,Y,X1,Y1,X2,Y2,D2), D2 <= (W*W)/4.
  ```
- **Detection Logic:** `perp_dist2/7` computes the squared perpendicular distance from point to the segment line using the standard integer cross-product formula (`((Y2-Y1)*X-(X2-X1)*Y+X2*Y1-Y2*X1)^2 / ((X2-X1)^2+(Y2-Y1)^2)`, kept as a ratio rather than doing division inside Clingo where possible, or precomputed as a lookup for known lane geometry).
- **Correction Strategy:** Project the offending point onto the nearest lane centerline, replacing X,Y with the projected coordinates for that timestep, then re-interpolate adjacent timesteps for continuity.
- **Priority Level:** High
- **Example Scenario:** A predicted point at T=3 drifts 2.1 m from centerline in a 3.5 m lane, exceeding the half-width tolerance.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R2.5 (lane departure warning is the early-warning version of this), R12.3 (correction must remain smooth)
- **Test Case:** `lane(l1,0,0,0,5000). lane_width(l1,350).` with `pos(ego,3,220,2500)`.

### R2.2 — Lane Change Permission & Minimum Gap
- **Description:** Restricts lane-change trajectories to segments where crossing is legally permitted and requires a minimum longitudinal gap in the target lane.
- **Traffic Law Basis:** Solid-line crossing prohibitions; safe lane-change statutes requiring adequate clearance.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `lane_boundary(L1,L2,Type)` (Type = solid/dashed), `pos(V,T,X,Y)` for other vehicles, `speed(V,T,S)`.
- **Violation Condition:**
  ```
  violation(illegal_lane_change, ego, T) :- crosses_boundary(ego,T,L1,L2), lane_boundary(L1,L2,solid).
  violation(unsafe_lane_change_gap, ego, T) :- crosses_boundary(ego,T,L1,L2), target_gap(ego,V,T,Gap), Gap < min_gap(S).
  ```
- **Detection Logic:** `crosses_boundary/4` fires when consecutive positions `T,T+1` fall on opposite sides of a boundary segment (sign change in the cross-product test). `target_gap` computes the longitudinal squared/linear distance to the nearest vehicle already in the target lane; `min_gap/1` is a speed-indexed lookup (larger gap required at higher relative speed).
- **Correction Strategy:** If the boundary is solid, remove the lane-change segment entirely and re-project onto the current lane; if the gap is insufficient, delay the crossing timestep until the gap opens or cancel the maneuver.
- **Priority Level:** Critical (gap violation), High (solid-line violation)
- **Example Scenario:** Ego plans to merge left across a solid line to avoid slow traffic, or merges into a 4 m gap at a 15 mph relative closing speed.
- **ASP Complexity:** Complex
- **Performance Impact:** High (pairwise checks against all lane-occupying vehicles per boundary crossing)
- **Dependency:** R3.1 (following distance), R4.3 (cut-in detection from the other vehicle's perspective)
- **Test Case:** `lane_boundary(l1,l2,solid).` with `pos(ego,2,-10,1000)`, `pos(ego,3,10,1050)` (sign change across x=0 boundary).

### R2.3 — Wrong-Way Driving Detection
- **Description:** Detects a predicted heading that opposes the designated direction of travel for the occupied lane.
- **Traffic Law Basis:** One-way street and divided-highway direction-of-travel statutes.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `lane(L,X1,Y1,X2,Y2)`, `lane_direction(L,DX,DY)`.
- **Violation Condition:**
  ```
  violation(wrong_way, ego, T) :- in_lane(ego,T,L), heading(ego,T,HX,HY), lane_direction(L,DX,DY), Dot = HX*DX + HY*DY, Dot < 0.
  ```
- **Detection Logic:** Dot product of the ego heading vector against the lane's canonical direction vector; a negative dot product means travel opposes the lane's designated direction (>90° divergence).
- **Correction Strategy:** This is treated as non-correctable via trajectory nudging — trigger an immediate stop/hold command and flag for takeover rather than attempting to "fix" a wrong-way trajectory locally.
- **Priority Level:** Critical
- **Example Scenario:** A localization or map-matching error causes the planner to output a trajectory heading into oncoming lane direction after a confusing off-ramp.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R11.1 (emergency braking), overrides all other lane rules when triggered
- **Test Case:** `lane_direction(l1,0,1).` with `heading(ego,2,5,-40)`.

### R2.4 — HOV / Bus Lane Restriction
- **Description:** Prevents the trajectory from using a restricted-access lane (HOV, bus-only, bike-only) unless the vehicle meets the posted eligibility.
- **Traffic Law Basis:** HOV/managed-lane and dedicated-transit-lane statutes.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `lane_restriction(L,Type,ActiveHours)`, `occupancy(ego,N)`, `current_time(Tm)`.
- **Violation Condition:**
  ```
  violation(restricted_lane, ego, T) :- in_lane(ego,T,L), lane_restriction(L,hov,MinOcc,ActiveHours), occupancy(ego,N), N < MinOcc, active(ActiveHours, current_time).
  violation(restricted_lane, ego, T) :- in_lane(ego,T,L), lane_restriction(L,bus_only,_,ActiveHours), active(ActiveHours, current_time).
  ```
- **Detection Logic:** Lane-level restriction facts are matched against static vehicle occupancy count and current time-of-day window (`active/2` checks the timestamp against the restriction's active hours, since many HOV lanes are time-limited).
- **Correction Strategy:** Re-route the affected timesteps into the nearest adjacent unrestricted lane using the same lane-change gap logic as R2.2.
- **Priority Level:** Medium
- **Example Scenario:** Trajectory drifts into a bus-only lane during its active enforcement hours (7–9am) while merging.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R2.2 (re-routing uses lane-change safety checks)
- **Test Case:** `lane_restriction(l3,bus_only,none,"7:00-9:00").` with `current_time("8:15")`, `pos(ego,1,X,Y)` resolving to `l3`.

### R2.5 — Lane Departure Early Warning
- **Description:** Fires a soft advisory before an actual lane-keeping violation (R2.1) occurs, when the trajectory trend shows the vehicle approaching a lane boundary.
- **Traffic Law Basis:** Not a direct statute — implements the safety intent behind lane-keeping law as a predictive early-intervention layer.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `lane(L,X1,Y1,X2,Y2)`, `lane_width(L,W)`.
- **Violation Condition:**
  ```
  warning(lane_departure, ego, T) :- pos(ego,T,X,Y), perp_dist2(X,Y,X1,Y1,X2,Y2,D2), lane_width(L,W), D2 > (W*W)/4 * 64/100, D2 <= (W*W)/4.
  ```
  (Fires in the outer ~20% band of the half-width, i.e. `sqrt(0.64)=0.8` of the boundary, expressed as an integer ratio to avoid sqrt.)
- **Detection Logic:** Same perpendicular-distance computation as R2.1 but bucketed into a warning band short of the hard boundary.
- **Correction Strategy:** Apply a gentle centering bias (small lateral correction spread over 2–3 timesteps) rather than the harder snap-to-center used once R2.1 has actually fired.
- **Priority Level:** Low
- **Example Scenario:** Trajectory drifts to 85% of the lane's half-width at T=2 without yet crossing the boundary.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R2.1 (escalates into it if uncorrected), R12.3 (smooth correction)
- **Test Case:** `lane_width(l1,350).` with a perpendicular offset of 150 cm (half-width 175 cm → 85%).

## 3. Following Behavior

### R3.1 — Safe Following Distance (Time-Based)
- **Description:** Enforces a minimum time-headway gap to the vehicle directly ahead, expressed as a time-to-reach threshold rather than a fixed distance.
- **Traffic Law Basis:** "Three-second rule" / assured-clear-distance statutes common across jurisdictions.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)`, `speed(ego,T,S)`, `lead_vehicle(ego,V,T)`.
- **Violation Condition:**
  ```
  violation(following_time, ego, T) :- lead_vehicle(ego,V,T), gap_dist(ego,V,T,D), speed(ego,T,S), S > 0, D * 100 < S * MinTimeUnits.
  ```
  (`MinTimeUnits` encodes 2 seconds in timestep units, e.g. 20 at 10 Hz; comparison cross-multiplied to avoid division: `D/S < 2` becomes `D*100 < S*20`.)
- **Detection Logic:** `gap_dist/4` is the longitudinal distance to the identified lead vehicle in the same lane; the cross-multiplied inequality keeps everything integer.
- **Correction Strategy:** Reduce planned speed for subsequent timesteps until the projected gap satisfies the minimum time headway, respecting max deceleration (R12.1).
- **Priority Level:** High
- **Example Scenario:** Ego trails a lead vehicle at 45 mph with only a 1-second gap.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R3.2 (speed-dependent distance is a complementary hard floor), R12.1
- **Test Case:** `lead_vehicle(ego,v2,3).` with `pos(ego,3,0,0)`, `pos(v2,3,0,900)`, `speed(ego,3,2010)` (gap 900cm at ~45mph → <2s).

### R3.2 — Safe Following Distance (Speed-Dependent Absolute Floor)
- **Description:** Enforces a minimum absolute gap that scales with ego speed, acting as a hard floor independent of the time-based rule (covers low-speed edge cases where time-headway alone under-constrains the gap).
- **Traffic Law Basis:** State-specific "one car length per 10 mph" style guidance codified in some driver-safety statutes.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)`, `speed(ego,T,S)`, `lead_vehicle(ego,V,T)`.
- **Violation Condition:**
  ```
  violation(following_min_gap, ego, T) :- lead_vehicle(ego,V,T), gap_dist(ego,V,T,D), speed(ego,T,S), min_gap_table(S,MinD), D < MinD.
  ```
- **Detection Logic:** `min_gap_table/2` is a monotonic step lookup (e.g. banded by 10 mph increments) mapping speed to a minimum absolute gap in cm, avoiding a continuous formula.
- **Correction Strategy:** Same as R3.1 — reduce speed / delay approach until `D >= MinD`.
- **Priority Level:** High
- **Example Scenario:** At very low speed (5 mph, e.g. stop-and-go traffic) the time-based rule permits a near-zero gap; this floor keeps at least ~2 car lengths.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R3.1 (max of the two gap requirements governs), R3.3
- **Test Case:** `min_gap_table(220,600).` with `speed(ego,1,220)`, `gap_dist(ego,v2,1,400)`.

### R3.3 — Tailgating Pattern Detection
- **Description:** Detects sustained close-following behavior across multiple consecutive timesteps rather than a single-frame gap dip, distinguishing tailgating from a momentary, transient closing event.
- **Traffic Law Basis:** Aggressive-driving / following-too-closely statutes that consider sustained proximity, not instantaneous distance alone.
- **Input Facts Required:** `violation(following_time, ego, T)` (from R3.1), across the horizon.
- **Violation Condition:**
  ```
  violation(tailgating, ego) :- N = #count{ T : violation(following_time, ego, T) }, N >= 4.
  ```
- **Detection Logic:** Aggregate count of R3.1 violations across the 6-timestep horizon; four or more flagged timesteps out of six indicates a sustained pattern rather than a single transient dip.
- **Correction Strategy:** Escalate the R3.1 correction magnitude (larger deceleration budget allotted) and suppress any planned acceleration for the remainder of the horizon.
- **Priority Level:** High
- **Example Scenario:** Ego's trajectory shows a persistently sub-1-second gap across T=1 through T=5.
- **ASP Complexity:** Medium (aggregate/count)
- **Performance Impact:** Low (post-hoc aggregation over already-computed facts)
- **Dependency:** R3.1
- **Test Case:** `violation(following_time,ego,1..4).` (four consecutive flagged steps).

### R3.4 — Caravan / Queue Behavior
- **Description:** When ego is part of a multi-vehicle queue (e.g. stopped or slow-moving traffic), enforces consistent gap-holding relative to the vehicle ahead rather than independently optimizing speed, preventing oscillatory accordion behavior.
- **Traffic Law Basis:** Not a direct statute — operationalizes reasonable-and-prudent following behavior in queued traffic to reduce rear-end risk from stop-and-go oscillation.
- **Input Facts Required:** `pos(V,T,X,Y)` for the queue, `speed(V,T,S)`, `lead_vehicle(ego,V,T)`, `queue_member(V)`.
- **Violation Condition:**
  ```
  violation(queue_oscillation, ego) :- lead_vehicle(ego,V,T), lead_vehicle(ego,V,T+1), speed(ego,T,S1), speed(ego,T+1,S2), |S2-S1| > MaxDelta, speed(V,T,SV), SV < LowSpeedThreshold.
  ```
- **Detection Logic:** Flags large speed swings between consecutive timesteps specifically while the lead vehicle's own speed is low (i.e. queued conditions), which is the signature of accordion/traffic-wave-inducing behavior (see also R15.4).
- **Correction Strategy:** Smooth the ego speed profile toward the lead vehicle's rolling-average speed rather than reacting to its instantaneous fluctuations.
- **Priority Level:** Medium
- **Example Scenario:** In stop-and-go traffic, planner alternates between hard braking and hard acceleration to chase a lead vehicle that is itself oscillating around 8 mph.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R15.4 (traffic wave dampening is the multi-vehicle generalization of this), R12.1
- **Test Case:** `speed(v2,2,130). speed(ego,2,300). speed(ego,3,50).` (delta 250 while lead is slow).

## 4. Collision Avoidance

### R4.1 — Minimum Lateral Clearance
- **Description:** Requires a minimum side-to-side gap between ego and any adjacent vehicle or object occupying a neighboring lane or shoulder.
- **Traffic Law Basis:** "Safe passing distance" statutes (some jurisdictions codify a specific minimum, e.g. 3 feet for cyclists — see R9).
- **Input Facts Required:** `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)`, `lateral_offset(ego,V,T,DY)`.
- **Violation Condition:**
  ```
  violation(lateral_clearance, ego, V, T) :- pos(ego,T,X,Y), pos(V,T,X2,Y2), V != ego, lateral_offset(ego,V,T,DY), abs(DY) < MinLateral, longitudinal_overlap(ego,V,T).
  ```
- **Detection Logic:** `lateral_offset` is the perpendicular component of separation relative to ego's heading; `longitudinal_overlap` confirms the two vehicles' extents overlap along the direction of travel (otherwise lateral offset alone is irrelevant, e.g. a car far ahead in an adjacent lane).
- **Correction Strategy:** Shift the ego path laterally away from the encroaching object within the current lane's available width, or delay a planned lane change until clearance is restored.
- **Priority Level:** Critical
- **Example Scenario:** A truck in the adjacent lane drifts to within 40 cm of the ego's side while both are traveling alongside each other.
- **ASP Complexity:** Medium
- **Performance Impact:** High (pairwise against every tracked object per timestep)
- **Dependency:** R2.1 (lane keeping), R9.3 (bike-lane-specific clearance is a stricter variant)
- **Test Case:** `pos(ego,3,0,1000). pos(v3,3,60,1005).` (60 cm lateral offset vs 100 cm minimum).

### R4.2 — Minimum Longitudinal Clearance
- **Description:** Requires a minimum front/rear gap to any vehicle sharing the same lane, independent of the time-based following rules, as a hard geometric floor.
- **Traffic Law Basis:** Assured-clear-distance-ahead doctrine.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)`, `in_lane(ego,T,L)`, `in_lane(V,T,L)`.
- **Violation Condition:**
  ```
  violation(longitudinal_clearance, ego, V, T) :- in_lane(ego,T,L), in_lane(V,T,L), longitudinal_gap(ego,V,T,D), D < MinLongGap.
  ```
- **Detection Logic:** `longitudinal_gap` projects both vehicle positions onto the lane's direction vector and takes the difference; compared against a fixed geometric minimum (distinct from R3's speed-scaled gap, this is the absolute floor that applies even at very low relative speed).
- **Correction Strategy:** Immediately reduce speed to restore the geometric minimum, taking priority over comfort constraints (R12).
- **Priority Level:** Critical
- **Example Scenario:** Predicted trajectory closes to within 1.5 m of a stopped vehicle ahead in the same lane.
- **ASP Complexity:** Simple
- **Performance Impact:** Medium
- **Dependency:** R3.1, R3.2, R11.1 (emergency braking feasibility check)
- **Test Case:** `in_lane(ego,4,l1). in_lane(v4,4,l1).` with `pos(ego,4,0,1000)`, `pos(v4,4,0,1150)` (150 cm gap).

### R4.3 — Cut-In Detection
- **Description:** Detects another vehicle entering ego's lane ahead at an unsafe gap, distinguishing this externally-caused hazard from ego-initiated following violations so the correction targets ego's reaction rather than blaming ego's positioning.
- **Traffic Law Basis:** Right-of-way and reasonable-reaction doctrines; informs the AV's defensive response rather than assigning fault.
- **Input Facts Required:** `pos(V,T,X,Y)`, `in_lane(V,T,L)`, `in_lane(V,T-1,L2)`, `L != L2`, `pos(ego,T,Xe,Ye)`.
- **Violation Condition:**
  ```
  hazard(cut_in, V, T) :- in_lane(V,T,L), in_lane(V,T-1,L2), L != L2, in_lane(ego,T,L), gap_dist(ego,V,T,D), D < SafeReactionGap.
  ```
- **Detection Logic:** Flags a lane-change-into-ego's-path event by comparing an external vehicle's lane membership between consecutive timesteps; `SafeReactionGap` is derived from ego's current speed (reaction-time based, similar table to R3.2).
- **Correction Strategy:** Immediately open the gap by decelerating (bounded by R12.1) rather than attempting a lateral avoidance maneuver, since a cut-in ahead is best handled longitudinally.
- **Priority Level:** Critical
- **Example Scenario:** A vehicle merges into ego's lane from an adjacent lane with only an 8 m gap at 50 mph closing speed.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R3.1, R2.2 (mirrors the lane-change rule from the other vehicle's side), R13.1 (predictive collision check should anticipate this)
- **Test Case:** `in_lane(v5,3,l1). in_lane(v5,2,l2).` with `in_lane(ego,3,l1)`, `gap_dist(ego,v5,3,800)`.

### R4.4 — Rear-End Collision Risk
- **Description:** Flags trajectories where ego's own deceleration profile would create excessive closing risk for a vehicle following ego (as opposed to ego rear-ending someone else, covered by R4.2/R13.2).
- **Traffic Law Basis:** General duty not to brake abruptly without cause where it creates an unreasonable rear-end risk to following traffic (codified in several jurisdictions' "no sudden stop" clauses).
- **Input Facts Required:** `speed(ego,T,S)`, `speed(ego,T+1,S2)`, `pos(V,T,X,Y)` for trailing vehicle, `following_gap(V,ego,T,D)`, `speed(V,T,SV)`.
- **Violation Condition:**
  ```
  hazard(rear_end_risk, ego, T) :- speed(ego,T,S), speed(ego,T+1,S2), Decel = S - S2, Decel > SuddenStopThreshold, following_gap(V,ego,T,D), speed(V,T,SV), reaction_dist(SV,Decel,RD), D < RD.
  ```
- **Detection Logic:** Computes whether a trailing vehicle, at its current speed and typical reaction time, could stop in time given ego's planned sudden deceleration and the current gap; `reaction_dist` is a lookup/formula using integer approximation of braking distance (speed-squared bucketed, not sqrt-based).
- **Correction Strategy:** Where the deceleration isn't safety-critical (i.e. not itself avoiding a collision), soften the braking profile to stay under `SuddenStopThreshold`; where it is safety-critical (e.g. R4.2 firing), this rule is suppressed in favor of the higher-priority stop.
- **Priority Level:** High (Medium when it conflicts with a Critical stop requirement, in which case it is overridden)
- **Example Scenario:** Planner outputs a hard brake to precisely hit a yellow light despite a closely-following vehicle with no compelling safety reason to brake that hard.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R4.2, R12.1, R5.2 (yellow light decision zone is a common trigger of unnecessary hard braking)
- **Test Case:** `speed(ego,2,900). speed(ego,3,300). following_gap(v6,ego,2,400). speed(v6,2,850).`

### R4.5 — Side-Swipe Risk (Converging Trajectories)
- **Description:** Detects predicted trajectories that converge laterally with another vehicle's predicted trajectory even if their current positions are not yet in conflict, catching side-swipe risk before it manifests as a clearance violation.
- **Traffic Law Basis:** General collision-avoidance duty; predictive extension of R4.1.
- **Input Facts Required:** `pos(ego,T,X,Y)` for T and T+1, `predicted_pos(V,T,X2,Y2)` for the tracked object's short-horizon prediction.
- **Violation Condition:**
  ```
  hazard(sideswipe_risk, ego, V, T) :- pos(ego,T,X,Y), pos(ego,T+1,X1,Y1), predicted_pos(V,T,X2,Y2), predicted_pos(V,T+1,X3,Y3), lateral_offset_now(X,Y,X2,Y2,D0), lateral_offset_next(X1,Y1,X3,Y3,D1), D1 < D0, D1 < ConvergenceThreshold.
  ```
- **Detection Logic:** Compares lateral separation at `T` vs `T+1`; a decreasing trend (`D1 < D0`) that crosses below a threshold flags convergence before an actual clearance breach, giving a one-timestep lead time.
- **Correction Strategy:** Apply a small preemptive lateral bias away from the converging trajectory, before the deficit reaches R4.1's hard threshold.
- **Priority Level:** High
- **Example Scenario:** Ego and an adjacent-lane vehicle are both drifting toward the shared boundary; neither is yet in violation of R4.1 but the trend predicts a breach within one timestep.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R4.1 (this is its early-warning counterpart), R13.1
- **Test Case:** `pos(ego,2,0,1000). pos(ego,3,30,1050).` `predicted_pos(v7,2,90,1005). predicted_pos(v7,3,55,1055).` (offset shrinking from 90 to 25).

## 5. Traffic Control Compliance

### R5.1 — Red Light Compliance
- **Description:** Prohibits any predicted position from entering an intersection's controlled zone while the governing signal is red.
- **Traffic Law Basis:** Universal red-light statutes (UVC §11-202).
- **Input Facts Required:** `pos(ego,T,X,Y)`, `intersection(I,Xc,Yc,R)`, `light_state(I,red,T)`, `controls_approach(I,L)`.
- **Violation Condition:**
  ```
  violation(red_light, ego, T) :- pos(ego,T,X,Y), intersection(I,Xc,Yc,R), D2=(X-Xc)*(X-Xc)+(Y-Yc)*(Y-Yc), D2 <= R*R, light_state(I,red,T), controls_approach(I,L), in_lane(ego,T,L).
  ```
- **Detection Logic:** Squared-distance-within-radius test for "inside the intersection box"; gated on the specific approach lane being governed by the red-showing signal head (`controls_approach` disambiguates multi-approach intersections).
- **Correction Strategy:** Truncate the trajectory at the stop line (nearest point on the intersection boundary along the approach lane) and hold; do not permit any timestep past that boundary until the light state input changes.
- **Priority Level:** Critical
- **Example Scenario:** Planner's trajectory carries ego 2 m into the intersection box at T=4 while the light is red for that approach.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R5.2 (yellow decision zone determines whether stopping was feasible), R10.2 (box blocking)
- **Test Case:** `intersection(i1,5000,5000,300). light_state(i1,red,4). controls_approach(i1,l7).` with `pos(ego,4,5000,5000)`.

### R5.2 — Yellow Light Decision Zone
- **Description:** Determines, for a trajectory approaching a yellow light, whether stopping safely is still feasible; if not, permits proceeding through rather than forcing an unsafe hard stop.
- **Traffic Law Basis:** "Dilemma zone" doctrine embedded in most jurisdictions' yellow-light statutes (proceed-if-cannot-safely-stop).
- **Input Facts Required:** `pos(ego,T,X,Y)`, `speed(ego,T,S)`, `light_state(I,yellow,T)`, `intersection(I,Xc,Yc,R)`, `stopping_distance(S,SD)`.
- **Violation Condition:**
  ```
  violation(unsafe_yellow_response, ego, T) :- light_state(I,yellow,T), dist_to_stopline(ego,I,T,D), speed(ego,T,S), stopping_distance(S,SD), D > SD, not decelerating(ego,T).
  ```
- **Detection Logic:** Compares the distance remaining to the stop line against the vehicle's computable stopping distance at current speed (a bucketed integer lookup keyed on speed, avoiding a v²/2a floating formula); if there's room to stop safely but the plan doesn't decelerate, that's the violation — not the act of proceeding.
- **Correction Strategy:** If `D > SD` (can stop safely): insert a deceleration profile targeting a stop at the line. If `D <= SD` (already committed): validate the through-trajectory doesn't clip into red (feeds R5.1) and leave it uncorrected — forcing a stop here would itself be unsafe (see R4.4).
- **Priority Level:** High
- **Example Scenario:** Light turns yellow when ego is 60 m out at 40 mph with ample stopping distance available (~35 m) but the plan continues at speed.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R5.1, R4.4 (avoid unsafe hard braking when already inside the dilemma zone), R12.1
- **Test Case:** `light_state(i1,yellow,2). dist_to_stopline(ego,i1,2,6000). speed(ego,2,1780). stopping_distance(1780,3500).`

### R5.3 — Green Light With Obstacle in Intersection
- **Description:** Prohibits entering an intersection on a green signal if the box is already occupied such that ego could not clear it before conflicting traffic receives right-of-way.
- **Traffic Law Basis:** "Do not enter unless you can proceed without obstructing" — anti-gridlock statutes independent of signal color.
- **Input Facts Required:** `light_state(I,green,T)`, `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)` for any vehicle/obstacle inside the box, `intersection(I,Xc,Yc,R)`.
- **Violation Condition:**
  ```
  violation(green_with_obstacle, ego, T) :- light_state(I,green,T), about_to_enter(ego,I,T), obstacle_in_box(V,I,T), not clear_path(ego,V,I,T).
  ```
- **Detection Logic:** `obstacle_in_box` reuses the same radius test as R5.1 applied to any tracked vehicle; `clear_path` checks whether ego's planned path through the box has a feasible gap around the obstacle's footprint.
- **Correction Strategy:** Hold ego at the stop line despite the green signal until the obstruction clears, functionally identical to the R5.1 correction but triggered by occupancy rather than signal color.
- **Priority Level:** Critical
- **Example Scenario:** Light is green but a stalled vehicle occupies the center of the intersection; ego's plan would enter behind it and potentially block the box.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R5.1, R10.2 (box blocking prevention — this is the same underlying hazard from the "entering" side)
- **Test Case:** `light_state(i1,green,3).` with `pos(v8,3,5000,5000)` inside `intersection(i1,5000,5000,300)`, ego about to enter.

### R5.4 — Flashing Signal Handling
- **Description:** Treats a flashing-red signal as a mandatory stop (equivalent to a stop sign) and a flashing-yellow as a proceed-with-caution reduced-speed zone.
- **Traffic Law Basis:** Flashing-signal statutes present in virtually all US/UK/EU-derived traffic codes.
- **Input Facts Required:** `light_state(I,flashing_red,T)`, `light_state(I,flashing_yellow,T)`, `pos(ego,T,X,Y)`, `intersection(I,Xc,Yc,R)`.
- **Violation Condition:**
  ```
  violation(flashing_red_no_stop, ego, T) :- light_state(I,flashing_red,T), at_stopline(ego,I,T), not has_stopped(ego,I).
  violation(flashing_yellow_speed, ego, T) :- light_state(I,flashing_yellow,T), pos(ego,T,X,Y), intersection(I,Xc,Yc,R), D2=(X-Xc)*(X-Xc)+(Y-Yc)*(Y-Yc), D2 <= (R*2)*(R*2), speed(ego,T,S), S > CautionSpeed.
  ```
- **Detection Logic:** `has_stopped/2` tracks whether a full-stop event (a timestep with `speed=0` at the stop line) exists in ego's recent history for that intersection, distinguishing flashing-red from ordinary red (which requires holding, not a momentary touch-and-go).
- **Correction Strategy:** For flashing-red, insert a full-stop segment at the line before allowing proceed (subject to R6.1 right-of-way once stopped); for flashing-yellow, cap speed similar to R1.3's mechanism.
- **Priority Level:** Critical (flashing-red), Medium (flashing-yellow)
- **Example Scenario:** Signal is in a power-fault flashing-red state (common after outages); planner treats it like a normal green and proceeds without stopping.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R6.1 (uncontrolled/stop-sign-equivalent right-of-way after stopping), R1.3
- **Test Case:** `light_state(i1,flashing_red,2).` with `pos(ego,2,X,Y)` at the stop line and no prior `speed(ego,_,0)`.

### R5.5 — Protected Arrow Signal Compliance
- **Description:** Restricts turning movements to only those permitted by an active protected-arrow phase, and prohibits the corresponding movement when the arrow is not showing (even if the adjacent through-signal is green).
- **Traffic Law Basis:** Protected-turn-phase statutes — a green ball does not authorize a turn that conflicts with opposing traffic when a separate arrow signal governs it.
- **Input Facts Required:** `light_state(I,arrow_green,Direction,T)`, `light_state(I,arrow_red,Direction,T)`, `pos(ego,T,X,Y)`, `planned_turn(ego,Direction)`.
- **Violation Condition:**
  ```
  violation(unprotected_turn, ego, T) :- planned_turn(ego,Direction), about_to_enter(ego,I,T), light_state(I,arrow_red,Direction,T), not light_state(I,green,T).
  ```
- **Detection Logic:** Checks the specific directional arrow state relevant to the planned turn rather than the general through-signal; the `not light_state(I,green,T)` clause allows the turn only when the general green additionally permits an unprotected (yield-on-turn) movement, deferring to R7.2 for the yield check in that case.
- **Correction Strategy:** Hold ego at the line until `arrow_green` is asserted for the relevant direction, or re-route the correction into a yield-and-turn sequence governed by R7.2 if only the general green is active.
- **Priority Level:** Critical
- **Example Scenario:** Ego plans a left turn while the left-turn arrow is red, even though the through-movement signal is green.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R7.2 (left turn across traffic — governs the fallback unprotected case), R5.1
- **Test Case:** `light_state(i1,arrow_red,left,3).` with `planned_turn(ego,left)`, ego `about_to_enter(ego,i1,3)`.

## 6. Right-of-Way

### R6.1 — Uncontrolled Intersection Priority
- **Description:** At an intersection with no signal or with all-way/none-way stop ambiguity, assigns priority based on arrival order and, when tied, the standard right-hand-first rule.
- **Traffic Law Basis:** Uncontrolled-intersection right-of-way statutes (first-to-arrive, then yield-to-right-on-tie).
- **Input Facts Required:** `intersection(I,Xc,Yc,R)`, `arrival_time(V,I,T)` (derived: first timestep a vehicle reaches the stop line), `approach_side(V,I,Side)`.
- **Violation Condition:**
  ```
  violation(uncontrolled_priority, ego, T) :- about_to_enter(ego,I,T), arrival_time(V,I,Tv), arrival_time(ego,I,Te), Tv < Te, not has_cleared(V,I,T).
  violation(uncontrolled_priority, ego, T) :- about_to_enter(ego,I,T), arrival_time(V,I,Te), arrival_time(ego,I,Te), approach_side(V,I,right_of_ego), not has_cleared(V,I,T).
  ```
- **Detection Logic:** First rule: strict arrival-order precedence. Second rule: simultaneous-arrival tiebreak using relative approach geometry (`right_of_ego` computed from the two approach headings' cross product).
- **Correction Strategy:** Hold ego at the stop line until the prioritized vehicle has cleared the box (`has_cleared`), independent of any signal state.
- **Priority Level:** Critical
- **Example Scenario:** Four-way stop; a cross-traffic vehicle arrived one timestep earlier and hasn't yet cleared the box.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R10.1 (queue position at the stop), R5.3
- **Test Case:** `arrival_time(v9,i2,1). arrival_time(ego,i2,2).` with `about_to_enter(ego,i2,3)`, `not has_cleared(v9,i2,3)`.

### R6.2 — Roundabout Entry Priority
- **Description:** Requires ego to yield to all traffic already circulating in the roundabout before entering, regardless of ego's own arrival order.
- **Traffic Law Basis:** Roundabout yield-on-entry statutes (circulating traffic always has priority).
- **Input Facts Required:** `roundabout(R,Xc,Yc,Radius)`, `pos(V,T,X,Y)` for circulating vehicles, `in_circulation(V,R,T)`, `about_to_enter(ego,R,T)`.
- **Violation Condition:**
  ```
  violation(roundabout_entry, ego, T) :- about_to_enter(ego,R,T), in_circulation(V,R,T), approach_conflict(ego,V,R,T).
  ```
- **Detection Logic:** `approach_conflict` checks whether a circulating vehicle would reach ego's entry point within a short time window, using angular position around the roundabout and circulating speed rather than Cartesian distance.
- **Correction Strategy:** Hold at the yield line; re-evaluate every timestep until the circulating gap is clear, then re-plan the entry trajectory into the gap.
- **Priority Level:** Critical
- **Example Scenario:** Ego's plan enters the roundabout while a circulating vehicle is 2 seconds from the entry point.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R2.2 (entry is itself a lane-change-like maneuver), R6.3
- **Test Case:** `in_circulation(v10,r1,2).` with angular approach putting `v10` within the conflict window of `about_to_enter(ego,r1,2)`.

### R6.3 — Yield to Oncoming Traffic During Turns
- **Description:** Requires a left-turning (or right-turn-on-red, jurisdiction-dependent) ego to yield to oncoming through traffic before completing the turn.
- **Traffic Law Basis:** Universal left-turn-yields-to-oncoming-traffic statute.
- **Input Facts Required:** `planned_turn(ego,left)`, `pos(V,T,X,Y)` for oncoming traffic, `oncoming(V,ego,T)`, `speed(V,T,S)`.
- **Violation Condition:**
  ```
  violation(turn_yield, ego, T) :- planned_turn(ego,left), crosses_oncoming_path(ego,T), oncoming(V,ego,T), time_to_conflict_point(V,T,TTC), TTC < SafeTurnWindow.
  ```
- **Detection Logic:** `time_to_conflict_point` estimates, using integer speed and remaining distance to the geometric conflict point, how soon the oncoming vehicle would arrive there; compared against the time ego's turn would occupy that point.
- **Correction Strategy:** Delay the turn's execution timesteps until `TTC >= SafeTurnWindow` for all oncoming vehicles; hold in the turn lane/pocket in the interim.
- **Priority Level:** Critical
- **Example Scenario:** Ego begins a left turn while an oncoming vehicle is 1.5 seconds from the conflict point at 35 mph.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R7.2 (left turn across traffic — shares the same geometry), R13.2 (TTC threshold rule)
- **Test Case:** `oncoming(v11,ego,2).` with `time_to_conflict_point(v11,2,12)` (1.2s at 10Hz) vs `SafeTurnWindow=30`.

### R6.4 — Emergency Vehicle Yielding
- **Description:** Requires ego to yield right-of-way (pull toward the shoulder and/or stop) when an emergency vehicle with active lights/siren is detected approaching from any direction.
- **Traffic Law Basis:** Universal move-over / yield-to-emergency-vehicle statutes.
- **Input Facts Required:** `emergency_vehicle(V,T,X,Y,active)`, `pos(ego,T,X2,Y2)`, `approaching(V,ego,T)`.
- **Violation Condition:**
  ```
  violation(emergency_yield, ego, T) :- emergency_vehicle(V,T,X,Y,active), approaching(V,ego,T), not yielding_maneuver(ego,T).
  ```
- **Detection Logic:** `approaching` fires when the emergency vehicle is on a converging or same-lane trajectory within a detection radius; `yielding_maneuver` checks whether ego's plan for that timestep includes the expected pull-over/stop response.
- **Correction Strategy:** Override the current plan with a lateral shift toward the road edge combined with a deceleration-to-stop profile, holding until the emergency vehicle has passed (`approaching` clears).
- **Priority Level:** Critical
- **Example Scenario:** An ambulance with active lights approaches from behind in the same lane; ego's plan continues straight at cruising speed.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R2.1 (must respect lane/shoulder geometry while pulling over), R11.1
- **Test Case:** `emergency_vehicle(amb1,2,0,-2000,active). approaching(amb1,ego,2).` with no yielding maneuver in the plan.

### R6.5 — Pedestrian Crosswalk Priority
- **Description:** Requires ego to yield right-of-way to any pedestrian lawfully present in a crosswalk that intersects ego's path, at both signalized and unsignalized crossings.
- **Traffic Law Basis:** Universal pedestrian-in-crosswalk right-of-way statutes.
- **Input Facts Required:** `crosswalk(C,X1,Y1,X2,Y2)`, `pedestrian(P,T,X,Y)`, `pos(ego,T,Xe,Ye)`, `path_intersects(ego,C,T)`.
- **Violation Condition:**
  ```
  violation(crosswalk_yield, ego, T) :- pedestrian(P,T,X,Y), in_crosswalk(X,Y,C), path_intersects(ego,C,T), not stopped_before(ego,C,T).
  ```
- **Detection Logic:** `in_crosswalk` bounds the pedestrian's position against the crosswalk polygon; `path_intersects` checks whether ego's predicted footprint overlaps that same polygon within the same or an adjacent timestep window (accounting for pedestrian walking speed).
- **Correction Strategy:** Insert a full stop before the crosswalk boundary for every timestep the pedestrian occupies it, resuming only once `in_crosswalk` is false for that pedestrian.
- **Priority Level:** Critical
- **Example Scenario:** A pedestrian is midway through an unsignalized marked crosswalk that ego's right-turn path crosses.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R9.1 (broader pedestrian safety rule this specializes), R10.4
- **Test Case:** `crosswalk(c1,1000,900,1000,1100).` with `pedestrian(p1,3,1000,1000)`, `path_intersects(ego,c1,3)`.

## 7. Turning Rules

### R7.1 — Right Turn on Red
- **Description:** Permits a right turn on a red signal only where not prohibited by signage and only after a full stop and confirmed clear path.
- **Traffic Law Basis:** Right-turn-on-red-after-stop statutes (jurisdiction-dependent; some intersections post "No Turn on Red").
- **Input Facts Required:** `light_state(I,red,T)`, `no_turn_on_red(I,Direction)`, `planned_turn(ego,right)`, `has_stopped(ego,I)`, `clear_path(ego,I,T)`.
- **Violation Condition:**
  ```
  violation(rtor_no_sign, ego, T) :- planned_turn(ego,right), light_state(I,red,T), no_turn_on_red(I,right), about_to_enter(ego,I,T).
  violation(rtor_no_stop, ego, T) :- planned_turn(ego,right), light_state(I,red,T), about_to_enter(ego,I,T), not has_stopped(ego,I).
  violation(rtor_unclear, ego, T) :- planned_turn(ego,right), light_state(I,red,T), about_to_enter(ego,I,T), not clear_path(ego,I,T).
  ```
- **Detection Logic:** Three independent gates — signage prohibition, prior full-stop event, and current clear-path (no conflicting traffic/pedestrians per R6.5/R6.3-style checks) — any one failing blocks the turn.
- **Correction Strategy:** If prohibited by sign, hold until green (defers to R5.1). If not yet stopped, insert the stop segment first. If path unclear, hold post-stop until clear, then proceed.
- **Priority Level:** Critical (sign violation, pedestrian conflict), High (missing full stop)
- **Example Scenario:** Ego plans a rolling right turn on red without a complete stop at an intersection lacking a "No Turn on Red" sign.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R5.1, R6.5, R6.3
- **Test Case:** `light_state(i3,red,2). planned_turn(ego,right).` with no prior `speed(ego,_,0)` at `i3`.

### R7.2 — Left Turn Across Traffic
- **Description:** Governs the full geometric and timing constraints of completing an unprotected left turn across one or more oncoming lanes.
- **Traffic Law Basis:** Left-turn-yields-to-oncoming statute (see also R6.3, which this rule builds on with the multi-lane crossing geometry).
- **Input Facts Required:** `planned_turn(ego,left)`, `oncoming_lane(L,I)` for each lane crossed, `oncoming(V,ego,T)` per lane, `turn_duration(ego,Est)`.
- **Violation Condition:**
  ```
  violation(left_turn_across, ego, T) :- planned_turn(ego,left), oncoming_lane(L,I), oncoming(V,ego,T), lane_of(V,L), time_to_conflict_point(V,T,TTC), turn_duration(ego,Est), TTC < Est + SafetyMargin.
  ```
- **Detection Logic:** Extends R6.3 to check every individual oncoming lane the turn crosses, not just the nearest one, since a clear near lane doesn't guarantee a clear far lane.
- **Correction Strategy:** Same as R6.3 — hold until all crossed lanes are clear for the full estimated turn duration plus margin.
- **Priority Level:** Critical
- **Example Scenario:** Ego's turn crosses two oncoming lanes; the near lane is clear but a vehicle in the far lane would arrive during the turn.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R6.3, R13.2
- **Test Case:** `oncoming_lane(l4,i3). oncoming_lane(l5,i3).` with `oncoming(v12,ego,2)` in `l5` giving `TTC=15` vs `Est+Margin=25`.

### R7.3 — U-Turn Restrictions
- **Description:** Prohibits U-turn maneuvers at locations where signage, intersection geometry, or sightline conditions make them illegal.
- **Traffic Law Basis:** U-turn-prohibited statutes (posted signage, business-district blanket prohibitions, limited-sightline restrictions).
- **Input Facts Required:** `planned_maneuver(ego,u_turn,T)`, `no_u_turn(L)`, `sightline_distance(L,SD)`, `required_sightline(SpeedLimit,ReqSD)`.
- **Violation Condition:**
  ```
  violation(u_turn_prohibited, ego, T) :- planned_maneuver(ego,u_turn,T), in_lane(ego,T,L), no_u_turn(L).
  violation(u_turn_sightline, ego, T) :- planned_maneuver(ego,u_turn,T), in_lane(ego,T,L), sightline_distance(L,SD), speed_limit(L,Limit), required_sightline(Limit,ReqSD), SD < ReqSD.
  ```
- **Detection Logic:** First clause is a direct signage-fact lookup; second compares the mapped sightline distance for that lane against the minimum required for the posted speed (faster roads require longer sightlines to execute a safe U-turn).
- **Correction Strategy:** Remove the U-turn segment entirely and re-route via the next legal turnaround point (a signaled intersection or dedicated U-turn lane, if mapped).
- **Priority Level:** High
- **Example Scenario:** Planner attempts a U-turn mid-block on a 45 mph arterial with a sightline of only 60 m against a required 120 m.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R6.3 (still applies to whatever oncoming traffic exists during a permitted U-turn)
- **Test Case:** `no_u_turn(l6).` with `planned_maneuver(ego,u_turn,3)`, `in_lane(ego,3,l6)`.

### R7.4 — Turn Signal Timing (Where Detectable)
- **Description:** When the platform can command a turn-signal actuation, requires it to be active for a minimum duration before any lane-change or turn maneuver begins.
- **Traffic Law Basis:** Signal-of-intent-before-turning statutes (typically 100 ft / 3 seconds minimum, jurisdiction-dependent).
- **Input Facts Required:** `planned_maneuver(ego,Type,T)` (Type = turn/lane_change), `signal_active(ego,Direction,T)`.
- **Violation Condition:**
  ```
  violation(late_signal, ego, T) :- planned_maneuver(ego,_,T), maneuver_direction(ego,T,Direction), not signal_active(ego,Direction,T-MinLeadSteps).
  ```
- **Detection Logic:** Checks backward from the maneuver's start timestep by `MinLeadSteps` (the minimum lead time expressed in timestep units) for continuous signal activation.
- **Correction Strategy:** Insert/advance the signal-activation command to begin at `T - MinLeadSteps`, delaying the physical maneuver itself if the full horizon doesn't allow enough lead time within 6 steps.
- **Priority Level:** Medium
- **Example Scenario:** Planner initiates a lane change at T=1 with the turn signal only commanded to activate at T=1 as well, giving zero lead time.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R2.2 (the lane-change maneuver this typically precedes)
- **Test Case:** `planned_maneuver(ego,lane_change,1).` with no `signal_active(ego,left,T)` fact for any `T<1`.

### R7.5 — Turning Radius Compliance
- **Description:** Ensures the planned turn's curvature does not exceed the vehicle's physical turning-radius limit or produce off-tracking that would encroach on adjacent lanes/curbs.
- **Traffic Law Basis:** Not a direct statute but a kinematic-feasibility constraint that supports lane-discipline compliance (R2.1) during turns; also relevant to "wide right turn" statutes for large vehicles.
- **Input Facts Required:** `pos(ego,T,X,Y)` for consecutive T, `min_turning_radius(ego,Rmin)`.
- **Violation Condition:**
  ```
  violation(turn_radius, ego, T) :- pos(ego,T-1,X0,Y0), pos(ego,T,X1,Y1), pos(ego,T+1,X2,Y2), curvature_radius(X0,Y0,X1,Y1,X2,Y2,R), R < Rmin, R != 0.
  ```
- **Detection Logic:** `curvature_radius` fits a circle through three consecutive predicted points (standard integer circumradius formula using squared side lengths) and compares against the vehicle's known minimum turning radius.
- **Correction Strategy:** Re-sample the turn path with a wider arc meeting `Rmin`, extending the maneuver over additional timesteps if the tighter turn was needed to fit within the original horizon.
- **Priority Level:** High
- **Example Scenario:** Planned turn geometry implies a 4 m radius while the vehicle's minimum is 6 m, which would require encroaching into the oncoming lane to complete.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R2.1, R4.1
- **Test Case:** Three consecutive `pos/4` facts fitting a circumradius of 380 cm against `min_turning_radius(ego,550)`.

## 8. Overtaking Rules

### R8.1 — Permitted Overtaking Conditions
- **Description:** Allows an overtaking (passing) maneuver only when the target lane is clear for a sufficient distance and marking/geometry permits crossing.
- **Traffic Law Basis:** General overtaking statutes requiring clear sightline and adequate opposing-lane clearance before passing.
- **Input Facts Required:** `planned_maneuver(ego,overtake,T)`, `lane_boundary(L1,L2,Type)`, `oncoming(V,ego,T)`, `sightline_distance(L1,SD)`.
- **Violation Condition:**
  ```
  violation(unsafe_overtake, ego, T) :- planned_maneuver(ego,overtake,T), lane_boundary(L1,L2,solid), crosses_boundary(ego,T,L1,L2).
  violation(unsafe_overtake, ego, T) :- planned_maneuver(ego,overtake,T), oncoming(V,ego,T), time_to_conflict_point(V,T,TTC), overtake_duration(ego,Est), TTC < Est + SafetyMargin.
  ```
- **Detection Logic:** Reuses the solid-boundary check from R2.2 plus a timing check structurally identical to R7.2's oncoming-conflict logic, applied to the full pass-and-return maneuver duration rather than a single lane change.
- **Correction Strategy:** Abort/delay the overtake until a solid-line-free, sufficiently-clear window exists; fall back to following behind the slower vehicle under R3.1/R3.2 in the interim.
- **Priority Level:** Critical
- **Example Scenario:** Planner attempts to pass a slow vehicle by crossing a solid centerline, or with an oncoming vehicle 2 seconds out against a 4-second overtake duration.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R2.2, R7.2, R8.2
- **Test Case:** `planned_maneuver(ego,overtake,2). lane_boundary(l1,l2,solid).` with `crosses_boundary(ego,2,l1,l2)`.

### R8.2 — No-Passing Zones (Solid Lines, Hills, Curves)
- **Description:** Explicitly prohibits overtaking within mapped no-passing zones, including geometrically-derived zones (limited sightline over a hill crest or through a curve) in addition to marked solid lines.
- **Traffic Law Basis:** No-passing-zone statutes based on both pavement markings and sightline-distance requirements.
- **Input Facts Required:** `no_passing_zone(Z,X1,Y1,X2,Y2)`, `sightline_distance(L,SD)`, `required_sightline(SpeedLimit,ReqSD)`, `pos(ego,T,X,Y)`.
- **Violation Condition:**
  ```
  violation(no_passing_zone, ego, T) :- planned_maneuver(ego,overtake,T), pos(ego,T,X,Y), no_passing_zone(Z,X1,Y1,X2,Y2), inside_zone(X,Y,X1,Y1,X2,Y2).
  violation(no_passing_sightline, ego, T) :- planned_maneuver(ego,overtake,T), in_lane(ego,T,L), sightline_distance(L,SD), speed_limit(L,Limit), required_sightline(Limit,ReqSD), SD < ReqSD.
  ```
- **Detection Logic:** First clause is a direct mapped-zone membership test (marked no-passing zones); second derives an effective no-passing condition from map-encoded sightline data even where no explicit marking fact exists (covers hill crests/curves where sightline is the actual hazard).
- **Correction Strategy:** Suppress the overtake maneuver entirely for any timestep inside the zone or below the sightline threshold; resume evaluation of R8.1 once past the zone.
- **Priority Level:** Critical
- **Example Scenario:** Planner attempts to pass just before a blind curve where mapped sightline drops to 45 m against a 90 m requirement at the posted speed.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R8.1
- **Test Case:** `no_passing_zone(z2,4000,0,5000,0).` with `pos(ego,3,4500,0)`, `planned_maneuver(ego,overtake,3)`.

### R8.3 — Overtaking on the Right (Where Prohibited)
- **Description:** Prohibits passing another vehicle on the right except where explicitly permitted (e.g. multi-lane roads where the vehicle ahead is turning left, or on designated multi-lane freeways).
- **Traffic Law Basis:** Pass-on-the-left-only default rule with enumerated exceptions.
- **Input Facts Required:** `planned_maneuver(ego,overtake,T)`, `overtake_side(ego,T,Side)`, `road_type(L,Type)`, `lead_vehicle_turning(V,left)`.
- **Violation Condition:**
  ```
  violation(right_overtake, ego, T) :- planned_maneuver(ego,overtake,T), overtake_side(ego,T,right), in_lane(ego,T,L), road_type(L,two_lane), not lead_vehicle_turning(V,left).
  ```
- **Detection Logic:** Flags a right-side pass specifically on two-lane roads absent the standard exception (lead vehicle signaling/executing a left turn), leaving multi-lane freeway right-side overtaking (a separate, generally-permitted case) unflagged via the `road_type` guard.
- **Correction Strategy:** Re-route the overtake to the left side following R8.1's clearance logic, or cancel the overtake and resume following.
- **Priority Level:** High
- **Example Scenario:** On a two-lane rural road, planner swings onto the shoulder to pass a slow-moving vehicle on the right with no left-turn exception applicable.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R8.1, R2.1 (shoulder use itself is a lane-discipline concern)
- **Test Case:** `road_type(l1,two_lane).` with `planned_maneuver(ego,overtake,2)`, `overtake_side(ego,2,right)`, no `lead_vehicle_turning(_,left)`.

### R8.4 — Overtaking Clearance Requirements
- **Description:** Requires a minimum lateral clearance be maintained throughout the overtake maneuver, distinct from the general lateral-clearance rule (R4.1) because overtaking clearance requirements are often statutorily larger (e.g. specific minimums for passing cyclists).
- **Traffic Law Basis:** Safe-passing-distance statutes with maneuver-specific minimums (e.g. "3 feet when passing a bicycle").
- **Input Facts Required:** `planned_maneuver(ego,overtake,T)`, `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)`, `overtaken_type(V,Type)`, `overtake_clearance_min(Type,MinLateral)`.
- **Violation Condition:**
  ```
  violation(overtake_clearance, ego, V, T) :- planned_maneuver(ego,overtake,T), overtaken_type(V,Type), overtake_clearance_min(Type,MinLateral), lateral_offset(ego,V,T,DY), abs(DY) < MinLateral.
  ```
- **Detection Logic:** Same lateral-offset computation as R4.1, but with a per-VRU/vehicle-type minimum lookup that is typically stricter than the generic vehicle-to-vehicle minimum (see R9.3 for the bicycle-lane-specific variant).
- **Correction Strategy:** Widen the ego lateral offset during the overtake window to meet `MinLateral`, borrowing width from the opposite lane only within the bounds already validated by R8.1's clear-window check.
- **Priority Level:** Critical (when Type=cyclist/pedestrian), High otherwise
- **Example Scenario:** Ego overtakes a cyclist with only 60 cm of lateral clearance against a 90 cm statutory minimum.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R4.1, R8.1, R9.3
- **Test Case:** `overtaken_type(c1,cyclist). overtake_clearance_min(cyclist,90).` with `lateral_offset(ego,c1,3,60)`.

## 9. Pedestrian & Cyclist Safety

### R9.1 — General Crosswalk Yielding
- **Description:** Broader form of R6.5 covering unmarked crosswalks implied at intersections (most jurisdictions treat every intersection corner-to-corner as a legal crosswalk even without markings).
- **Traffic Law Basis:** Unmarked-crosswalk statutes extending pedestrian right-of-way to intersection corners generally.
- **Input Facts Required:** `intersection(I,Xc,Yc,R)`, `pedestrian(P,T,X,Y)`, `implied_crosswalk(I,Corner1,Corner2)`, `path_intersects(ego,I,T)`.
- **Violation Condition:**
  ```
  violation(implied_crosswalk_yield, ego, T) :- pedestrian(P,T,X,Y), near_intersection_corner(X,Y,I), path_intersects(ego,I,T), not stopped_before(ego,I,T).
  ```
- **Detection Logic:** `near_intersection_corner` extends the crosswalk-membership test of R6.5 to the geometric corner region even absent explicit `crosswalk/5` marking facts.
- **Correction Strategy:** Identical stop-and-hold correction as R6.5.
- **Priority Level:** Critical
- **Example Scenario:** A pedestrian steps off an unmarked corner at a residential intersection ego is turning through.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R6.5
- **Test Case:** `implied_crosswalk(i4,c_ne,c_nw).` with `pedestrian(p2,2,X,Y)` near corner `c_ne`, `path_intersects(ego,i4,2)`.

### R9.2 — School Zone Pedestrian Behavior
- **Description:** Applies heightened stopping/yielding sensitivity around school zones, including stopping for a crossing guard signal independent of pedestrian position.
- **Traffic Law Basis:** School-zone crossing-guard authority statutes (a raised stop sign/guard signal compels a stop regardless of other signals).
- **Input Facts Required:** `zone(Z,X1,Y1,X2,Y2,school)`, `crossing_guard_signal(Z,T,stop)`, `pos(ego,T,X,Y)`.
- **Violation Condition:**
  ```
  violation(school_guard_stop, ego, T) :- crossing_guard_signal(Z,T,stop), inside_zone(X,Y,X1,Y1,X2,Y2), pos(ego,T,X,Y), zone(Z,X1,Y1,X2,Y2,school), not stopped_before(ego,Z,T).
  ```
- **Detection Logic:** Direct fact-driven stop compulsion; overrides any conflicting green-light/right-of-way facts within the zone for that timestep.
- **Correction Strategy:** Insert a full stop before the guard's position and hold until `crossing_guard_signal` changes to `go`.
- **Priority Level:** Critical
- **Example Scenario:** A crossing guard holds up a stop sign at a school zone while ego's plan continues through on a green light.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R1.3, R5.1 (this rule takes precedence over signal state)
- **Test Case:** `crossing_guard_signal(z1,3,stop).` with `pos(ego,3,2000,1000)` inside `zone(z1,1000,0,3000,2000,school)`.

### R9.3 — Bicycle Lane Respect
- **Description:** Prohibits the ego trajectory from entering a marked bicycle lane except for the brief, permitted crossing needed to complete a lawful turn.
- **Traffic Law Basis:** Dedicated-bicycle-lane statutes restricting motor vehicle use except at turn crossings.
- **Input Facts Required:** `lane(L,X1,Y1,X2,Y2)`, `lane_type(L,bicycle)`, `pos(ego,T,X,Y)`, `planned_turn(ego,_)`.
- **Violation Condition:**
  ```
  violation(bike_lane_encroachment, ego, T) :- pos(ego,T,X,Y), in_lane_geom(X,Y,L), lane_type(L,bicycle), not turn_crossing(ego,T,L).
  ```
- **Detection Logic:** Reuses the lane-membership geometry from R2.1 restricted to bicycle-typed lanes; `turn_crossing` exempts only the single/adjacent timestep(s) needed to cross the bike lane while completing a signaled turn, per R9.3-specific clearance (R8.4).
- **Correction Strategy:** Re-project any non-turn-crossing timestep out of the bicycle lane back into the adjacent travel lane.
- **Priority Level:** High
- **Example Scenario:** Planner routes a lane change through a bike lane to reach a right-turn pocket without it being an immediate-turn crossing.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R2.1, R8.4, R9.4
- **Test Case:** `lane_type(l8,bicycle).` with `pos(ego,2,X,Y)` inside `l8` and no `planned_turn(ego,_)` at `T=2`.

### R9.4 — Blind Spot Detection for Vulnerable Road Users
- **Description:** Prevents a lane change or turn when a pedestrian or cyclist occupies a region of the ego vehicle's known perception/mechanical blind spot rather than relying solely on their tracked position (which may be stale or low-confidence in that region).
- **Traffic Law Basis:** General duty-of-care extension; operationalizes "check your blind spot" as a hard constraint rather than an assumption.
- **Input Facts Required:** `blind_spot_zone(ego,Side,X1,Y1,X2,Y2)`, `vru(P,T,X,Y,Type)`, `planned_maneuver(ego,Type2,T)`.
- **Violation Condition:**
  ```
  violation(blind_spot_maneuver, ego, T) :- planned_maneuver(ego,lane_change,T), maneuver_direction(ego,T,Side), blind_spot_zone(ego,Side,X1,Y1,X2,Y2), vru(P,T,X,Y,_), inside_zone(X,Y,X1,Y1,X2,Y2).
  violation(blind_spot_maneuver, ego, T) :- planned_maneuver(ego,lane_change,T), maneuver_direction(ego,T,Side), blind_spot_zone(ego,Side,X1,Y1,X2,Y2), last_confident_detection_age(P,T,Age), Age > StaleThreshold, last_known_pos(P,X,Y), inside_zone(X,Y,X1,Y1,X2,Y2).
  ```
- **Detection Logic:** First clause: a currently-tracked VRU sits inside the mapped blind-spot polygon on the maneuver side. Second clause: a VRU whose last confident detection is stale (perception dropout, common right at the blind-spot boundary) is conservatively assumed still present.
- **Correction Strategy:** Cancel/delay the lane change until the zone is clear or a fresh, confident detection confirms the VRU has moved clear.
- **Priority Level:** Critical
- **Example Scenario:** A cyclist's tracked confidence drops as they enter the ego's right-rear blind spot just before a planned right lane change.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R2.2, R9.3, R4.1
- **Test Case:** `blind_spot_zone(ego,right,50,-800,300,-200).` with `last_confident_detection_age(p3,2,12)`, `last_known_pos(p3,150,-500)`.

### R9.5 — Sidewalk Encroachment Prevention
- **Description:** Prohibits any predicted trajectory point from entering a mapped sidewalk area, including during low-speed maneuvers like parking or pulling to a curb.
- **Traffic Law Basis:** General prohibition on operating a motor vehicle on a sidewalk except for a direct, permitted driveway crossing.
- **Input Facts Required:** `sidewalk(S,X1,Y1,X2,Y2)`, `pos(ego,T,X,Y)`, `driveway_crossing(ego,T)`.
- **Violation Condition:**
  ```
  violation(sidewalk_encroachment, ego, T) :- pos(ego,T,X,Y), inside_zone(X,Y,X1,Y1,X2,Y2), sidewalk(S,X1,Y1,X2,Y2), not driveway_crossing(ego,T).
  ```
- **Detection Logic:** Direct polygon-membership test against mapped sidewalk geometry, exempted only for the specific timestep(s) of a legitimate perpendicular driveway crossing.
- **Correction Strategy:** Re-project the offending point back onto the roadway/shoulder boundary nearest the sidewalk edge.
- **Priority Level:** Critical
- **Example Scenario:** A curb-pullover maneuver overshoots and a predicted point lands 40 cm onto the adjacent sidewalk.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R2.1, R9.1 (pedestrians are most likely to be present exactly where this fires)
- **Test Case:** `sidewalk(s1,0,1200,2000,1400).` with `pos(ego,4,900,1250)`, no `driveway_crossing(ego,4)`.

## 10. Intersection Behavior

### R10.1 — Queue Position at Red Light
- **Description:** Requires ego to stop behind the last vehicle already queued at a red light rather than planning a trajectory that would overlap or cut into the existing queue.
- **Traffic Law Basis:** General duty to maintain a single-file queue and not to leapfrog stopped traffic at a controlled intersection.
- **Input Facts Required:** `light_state(I,red,T)`, `pos(V,T,X,Y)` for queued vehicles, `queue_position(V,I,N)`, `pos(ego,T,Xe,Ye)`.
- **Violation Condition:**
  ```
  violation(queue_cut, ego, T) :- light_state(I,red,T), last_queued_vehicle(V,I,T), longitudinal_gap(ego,V,T,D), D < MinQueueGap, not behind(ego,V,T).
  ```
- **Detection Logic:** Identifies the current last vehicle in queue and checks whether ego's predicted stop point would fall behind it with adequate gap, using the `behind/3` ordering predicate derived from lane-direction projection.
- **Correction Strategy:** Re-target ego's stop point to `MinQueueGap` behind the last queued vehicle's current (or projected) position.
- **Priority Level:** High
- **Example Scenario:** Planner's stop point at a red light would place ego alongside rather than behind the last queued car.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R4.2, R5.1
- **Test Case:** `last_queued_vehicle(v13,i5,2).` with `pos(v13,2,0,3000)`, ego's planned stop at `pos(ego,4,0,3050)` (only 50 cm gap).

### R10.2 — Box Blocking Prevention
- **Description:** Prohibits ego from entering an intersection unless there is sufficient space to fully clear it before the plan would otherwise require stopping inside.
- **Traffic Law Basis:** Anti-gridlock / "don't block the box" statutes.
- **Input Facts Required:** `intersection(I,Xc,Yc,R)`, `pos(ego,T,X,Y)`, `queue_ahead(V,I,T)`, `ego_length(Len)`.
- **Violation Condition:**
  ```
  violation(box_blocking, ego, T) :- about_to_enter(ego,I,T), queue_ahead(V,I,T), space_beyond_box(ego,V,I,T,Space), Space < ego_length(Len), Len > Space.
  ```
- **Detection Logic:** Computes remaining space beyond the far side of the intersection box before the next queued vehicle, comparing against ego's own length to determine whether it could fully exit before traffic would force a stop mid-box.
- **Correction Strategy:** Hold ego at the near stop line rather than entering, re-evaluating each timestep as the downstream queue moves.
- **Priority Level:** Critical
- **Example Scenario:** Traffic beyond the intersection is backed up such that entering now would leave ego stopped inside the box when the cross-signal turns green.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R5.1, R5.3
- **Test Case:** `queue_ahead(v14,i5,3).` with `space_beyond_box(ego,v14,i5,3,350)` vs `ego_length(480)`.

### R10.3 — Protected Left Turn Lane Compliance
- **Description:** Requires ego, once committed to a protected left-turn pocket/lane, to complete the turn only during the corresponding protected phase and not to re-merge into through traffic.
- **Traffic Law Basis:** Dedicated-turn-lane statutes restricting a vehicle in that lane to the corresponding movement.
- **Input Facts Required:** `lane_type(L,protected_left)`, `in_lane(ego,T,L)`, `planned_maneuver(ego,Type,T)`.
- **Violation Condition:**
  ```
  violation(protected_lane_misuse, ego, T) :- in_lane(ego,T,L), lane_type(L,protected_left), planned_maneuver(ego,Type,T), Type != left_turn, Type != stopped.
  ```
- **Detection Logic:** Directly flags any planned maneuver from a protected-left-only lane that isn't the left turn itself (or a legitimate stop while waiting for the arrow, per R5.5).
- **Correction Strategy:** Remove the non-conforming maneuver segment and hold in-lane until the protected phase (R5.5) permits the turn.
- **Priority Level:** Medium
- **Example Scenario:** Planner enters a marked left-turn pocket but then plans to go straight through the intersection instead.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R5.5
- **Test Case:** `lane_type(l9,protected_left).` with `in_lane(ego,2,l9)`, `planned_maneuver(ego,straight,2)`.

### R10.4 — Right Turn Yielding to Pedestrians in the Cross Street
- **Description:** Requires a right-turning ego to yield to pedestrians crossing the street ego is turning into, even after any vehicular right-of-way/signal check has cleared.
- **Traffic Law Basis:** Turn-yields-to-pedestrian statutes, applied specifically to the destination crosswalk of a right turn.
- **Input Facts Required:** `planned_turn(ego,right)`, `destination_crosswalk(I,Direction,C)`, `pedestrian(P,T,X,Y)`, `in_crosswalk(X,Y,C)`.
- **Violation Condition:**
  ```
  violation(right_turn_ped_yield, ego, T) :- planned_turn(ego,right), destination_crosswalk(I,right,C), pedestrian(P,T,X,Y), in_crosswalk(X,Y,C), path_intersects(ego,C,T).
  ```
- **Detection Logic:** Same crosswalk-membership mechanism as R6.5/R9.1 but explicitly scoped to the destination-side crosswalk of a right turn, which is easy to under-check if attention is focused on the vehicular conflict check (R7.1) alone.
- **Correction Strategy:** Hold ego short of the destination crosswalk until the pedestrian has cleared it, even if the vehicular right-of-way (R7.1/R5.1) is already satisfied.
- **Priority Level:** Critical
- **Example Scenario:** Ego completes a right-turn-on-red stop and clear-path check (R7.1) but a pedestrian is still crossing the street being turned into.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R7.1, R6.5
- **Test Case:** `destination_crosswalk(i5,right,c3).` with `pedestrian(p4,3,X,Y)` inside `c3`, `path_intersects(ego,c3,3)`.

## 11. Emergency Situations

### R11.1 — Emergency Braking Feasibility
- **Description:** Determines whether a full emergency stop can be physically achieved before a detected imminent hazard, and if not, flags that hard braking alone is insufficient so an evasive maneuver (R11.2) must be considered instead.
- **Traffic Law Basis:** Not a direct statute — implements the physical precondition underlying every "avoid a collision" duty.
- **Input Facts Required:** `speed(ego,T,S)`, `hazard_distance(ego,H,T,D)`, `max_decel(ego,MaxD)`, `braking_distance(S,MaxD,BD)`.
- **Violation Condition:**
  ```
  insufficient(emergency_brake, ego, H, T) :- hazard_distance(ego,H,T,D), speed(ego,T,S), max_decel(ego,MaxD), braking_distance(S,MaxD,BD), D < BD.
  ```
- **Detection Logic:** `braking_distance/3` uses a precomputed integer lookup table keyed by (speed-band, deceleration-band) rather than a floating v²/(2a) formula, honoring the integer-only constraint; comparison against the actual remaining distance to the hazard.
- **Correction Strategy:** If sufficient, insert the maximum-safe (not necessarily maximum-possible) deceleration profile targeting a stop before the hazard. If insufficient, hand off to R11.2 for an evasive-steering evaluation rather than attempting a stop that physics rules out.
- **Priority Level:** Critical
- **Example Scenario:** A pedestrian steps into the roadway 8 m ahead of ego traveling at 35 mph, inside the computed braking distance of 14 m.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R11.2, R12.1 (this rule can override the comfort deceleration cap)
- **Test Case:** `hazard_distance(ego,h1,2,800). speed(ego,2,1560). max_decel(ego,80). braking_distance(1560,80,1400).`

### R11.2 — Evasive Maneuver Boundaries
- **Description:** When braking alone cannot avoid a hazard (per R11.1), constrains any evasive steering maneuver to stay within road boundaries and avoid creating a new, worse collision.
- **Traffic Law Basis:** Necessity-doctrine style reasoning embedded in AV safety design — an evasive maneuver is only justified if it does not trade one hazard for a more severe one.
- **Input Facts Required:** `insufficient(emergency_brake, ego, H, T)`, `road_boundary(X1,Y1,X2,Y2)`, `pos(V,T,X,Y)` for all nearby agents.
- **Violation Condition:**
  ```
  violation(unsafe_evasion, ego, T) :- insufficient(emergency_brake,ego,H,T), evasive_path(ego,T,X,Y), (not in_any_lane(X,Y); new_hazard(X,Y,V,T)).
  ```
- **Detection Logic:** An evasive path candidate is rejected if it would exit all drivable surface (`not in_any_lane`, reusing R2.1's geometry against the full drivable area not just the lane) or if it introduces a new conflict with a previously-clear agent `V`.
- **Correction Strategy:** Search a bounded set of candidate lateral offsets (within the drivable surface) for one that clears the original hazard `H` without triggering `new_hazard`; if none exists, fall back to the maximum feasible braking from R11.1 even though it doesn't fully avoid the hazard, since that minimizes impact severity.
- **Priority Level:** Critical
- **Example Scenario:** Braking cannot stop in time for a suddenly-stopped vehicle ahead; a small lateral shift within the lane (not crossing into oncoming traffic) is evaluated as the resolving evasive path.
- **ASP Complexity:** Complex
- **Performance Impact:** High
- **Dependency:** R11.1, R2.1, R4.1
- **Test Case:** `insufficient(emergency_brake,ego,h1,2).` with candidate `evasive_path(ego,2,150,900)` checked against lane geometry and other agents.

### R11.3 — Guardrail / Fixed-Obstacle Avoidance
- **Description:** Treats mapped fixed infrastructure (guardrails, barriers, bollards, permanent obstacles) as hard no-go geometry with zero tolerance, distinct from dynamic-agent clearance rules.
- **Traffic Law Basis:** Not a direct statute — a physical-safety hard constraint that supersedes almost all other trajectory preferences.
- **Input Facts Required:** `fixed_obstacle(O,X1,Y1,X2,Y2)`, `pos(ego,T,X,Y)`, `ego_footprint(Width)`.
- **Violation Condition:**
  ```
  violation(fixed_obstacle_contact, ego, T) :- pos(ego,T,X,Y), fixed_obstacle(O,X1,Y1,X2,Y2), footprint_overlap(X,Y,Width,X1,Y1,X2,Y2).
  ```
- **Detection Logic:** `footprint_overlap` expands the point to the vehicle's physical width/length envelope and tests for any overlap with the mapped obstacle geometry — zero-margin, since contact itself (not proximity) is the violation.
- **Correction Strategy:** Treat as an absolute hard constraint in the correction search — any candidate trajectory (including outputs of R11.2's evasive search) that violates this is discarded outright regardless of what other hazard it was resolving.
- **Priority Level:** Critical
- **Example Scenario:** An evasive maneuver computed under R11.2 would clear a suddenly-stopped vehicle but clip a guardrail on the shoulder.
- **ASP Complexity:** Simple
- **Performance Impact:** Medium
- **Dependency:** R11.2 (constrains its candidate search space), R2.1
- **Test Case:** `fixed_obstacle(gr1,0,4900,0,5100).` with `pos(ego,4,20,5000)`, `ego_footprint(180)`.

### R11.4 — Active Construction Zone Compliance
- **Description:** Applies a bundle of construction-zone-specific constraints — reduced speed (R1.3), lane shifts around a mapped work area, and increased clearance to workers — as a single coordinated rule set rather than independently-triggered pieces.
- **Traffic Law Basis:** Work-zone safety statutes, often carrying enhanced penalties for violations within an active zone.
- **Input Facts Required:** `zone(Z,X1,Y1,X2,Y2,construction)`, `zone_active(Z,T)`, `lane_shift(Z,L,NewPath)`, `worker(W,T,X,Y)`.
- **Violation Condition:**
  ```
  violation(construction_lane, ego, T) :- zone_active(Z,T), lane_shift(Z,L,NewPath), in_lane(ego,T,L), not following_path(ego,T,NewPath).
  violation(construction_worker_clearance, ego, T) :- zone_active(Z,T), worker(W,T,X,Y), pos(ego,T,Xe,Ye), D2=(X-Xe)*(X-Xe)+(Y-Ye)*(Y-Ye), D2 < WorkerClearance2.
  ```
- **Detection Logic:** First clause checks conformance to a mapped temporary lane-shift path overriding the default lane centerline; second reuses the squared-distance proximity pattern from R1.5/R4.1 but with a construction-specific (typically larger) clearance radius.
- **Correction Strategy:** Re-project the trajectory onto `NewPath` for the shift violation; widen lateral offset away from any worker position for the clearance violation, combined with the R1.3 speed reduction already in effect for the zone.
- **Priority Level:** Critical
- **Example Scenario:** A construction zone has shifted the travel lane 1.5 m left around a work area with a flagger standing 3 m from the shifted path; ego's plan follows the old, unshifted lane centerline.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R1.3, R2.1, R4.1
- **Test Case:** `zone_active(z3,2). lane_shift(z3,l1,path_shifted).` with `in_lane(ego,2,l1)`, no `following_path(ego,2,path_shifted)`.

## 12. Comfort & Ride Quality

### R12.1 — Maximum Acceleration/Deceleration Limits
- **Description:** Bounds the longitudinal acceleration or deceleration implied between any two consecutive predicted timesteps to a comfort-preserving range, except where explicitly overridden by a Critical safety rule.
- **Traffic Law Basis:** Not statutory — a comfort/quality constraint referenced by several higher-priority rules (R1.1, R3.x, R4.x, R11.1) as the default bound they either respect or explicitly override.
- **Input Facts Required:** `speed(ego,T,S)`, `speed(ego,T+1,S2)`, `max_comfort_accel(A)`, `max_comfort_decel(D)`, `override_active(ego,T)`.
- **Violation Condition:**
  ```
  violation(excess_accel, ego, T) :- speed(ego,T,S), speed(ego,T+1,S2), S2 > S, (S2-S) > A, not override_active(ego,T).
  violation(excess_decel, ego, T) :- speed(ego,T,S), speed(ego,T+1,S2), S2 < S, (S-S2) > D, not override_active(ego,T).
  ```
- **Detection Logic:** Simple integer delta check between consecutive speed facts against fixed comfort thresholds; `override_active` is asserted by any Critical-priority rule (R4.2, R5.1, R11.1, etc.) that legitimately requires exceeding the comfort bound.
- **Correction Strategy:** Re-interpolate the speed profile across more of the 6-timestep horizon so the same net speed change is spread over more steps, staying within `A`/`D` wherever no override is active.
- **Priority Level:** Low
- **Example Scenario:** Planner outputs a speed change of 15 mph in a single 0.1s timestep with no safety-critical rule active to justify it.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** Referenced by nearly every speed-correcting rule as the default profile shape
- **Test Case:** `speed(ego,2,800). speed(ego,3,1400).` with `max_comfort_accel(300)`, no `override_active(ego,2)`.

### R12.2 — Lateral Acceleration / Jerk Limits
- **Description:** Bounds both the lateral acceleration (cornering force) and its rate of change (jerk) implied by the predicted path curvature and speed, to avoid uncomfortable or destabilizing steering inputs.
- **Traffic Law Basis:** Not statutory — vehicle dynamics/comfort constraint.
- **Input Facts Required:** `pos(ego,T,X,Y)` for three consecutive T, `speed(ego,T,S)`, `max_lateral_accel(LA)`, `max_jerk(J)`.
- **Violation Condition:**
  ```
  violation(lateral_accel, ego, T) :- curvature_radius(X0,Y0,X1,Y1,X2,Y2,R), speed(ego,T,S), R != 0, lat_accel_est(S,R,LatA), LatA > LA.
  violation(lateral_jerk, ego, T) :- lat_accel_est(S,R,LatA0), lat_accel_est(S2,R2,LatA1), Delta = |LatA1-LatA0|, Delta > J.
  ```
- **Detection Logic:** `curvature_radius` (shared with R7.5) combined with an integer-bucketed lookup `lat_accel_est/3` approximating `v²/r` without floating division; jerk is the frame-to-frame delta of that estimate.
- **Correction Strategy:** Widen the turn radius or reduce speed through the curve (whichever the current constraint set permits) until both `LatA` and the frame-to-frame `Delta` fall within bounds.
- **Priority Level:** Low
- **Example Scenario:** A tight lane-change-and-recover maneuver produces a sharp lateral acceleration spike between T=2 and T=3.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R7.5, R2.1
- **Test Case:** Three `pos/4` facts implying `R=250` at `speed(ego,2,900)` against `max_lateral_accel` threshold.

### R12.3 — Smooth Trajectory Requirement
- **Description:** Requires the overall predicted path to vary smoothly frame-to-frame (no discontinuous jumps in heading or position beyond what speed allows), independent of any single-rule correction, since multiple independent corrections can otherwise compose into a jagged path.
- **Traffic Law Basis:** Not statutory — an integration-quality constraint over the composed output of all other rules' corrections.
- **Input Facts Required:** `pos(ego,T,X,Y)` across the full horizon, `speed(ego,T,S)`, `max_heading_change(HC)`.
- **Violation Condition:**
  ```
  violation(discontinuous_path, ego, T) :- heading(ego,T-1,DX0,DY0), heading(ego,T,DX1,DY1), heading_delta(DX0,DY0,DX1,DY1,Delta), Delta > HC.
  ```
- **Detection Logic:** `heading_delta` computes an integer-approximate angular change between consecutive heading vectors (via cross/dot product sign and magnitude buckets rather than an actual arctangent) and compares to a maximum per-timestep heading change.
- **Correction Strategy:** Re-smooth the full 6-point path with a constrained spline/interpolation pass that respects all already-established hard constraints (lane bounds, clearances) as corridor limits rather than exact waypoints.
- **Priority Level:** Low
- **Example Scenario:** Independent corrections from R2.1 (lane-keeping snap) and R4.5 (side-swipe bias) at adjacent timesteps compose into a zig-zag rather than a smooth curve.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** Runs last, after all other geometric corrections (R2.x, R4.x especially)
- **Test Case:** `heading(ego,2,50,10). heading(ego,3,-30,15).` implying a large sign-reversal in the lateral component.

### R12.4 — Passenger Comfort Scoring
- **Description:** Produces an aggregate, non-binding comfort score across the horizon (combining R12.1–R12.3 margins) used to rank otherwise-equally-safe candidate corrections rather than to itself trigger a hard violation.
- **Traffic Law Basis:** Not statutory — a tie-breaking optimization signal.
- **Input Facts Required:** Derived from `violation(excess_accel,_,_)`, `violation(excess_decel,_,_)`, `violation(lateral_accel,_,_)`, `violation(discontinuous_path,_,_)` margins across the horizon.
- **Violation Condition:** None (advisory score, not a violation predicate) —
  ```
  comfort_score(ego, Score) :- Score = #sum{ M,T : margin(ego,T,M) }.
  ```
- **Detection Logic:** Sums per-timestep margin values (how far each comfort quantity is from its threshold) across the horizon using Clingo's `#sum` aggregate; used only to select among multiple otherwise-valid correction candidates from R11.2's search, R12.3's re-smoothing, etc.
- **Correction Strategy:** N/A — this rule informs selection among candidates produced by other rules rather than modifying the trajectory itself.
- **Priority Level:** Low
- **Example Scenario:** Two candidate evasive paths from R11.2 both clear all hard constraints; the one with the higher `comfort_score` is selected.
- **ASP Complexity:** Complex (aggregate over derived margins)
- **Performance Impact:** Medium
- **Dependency:** R12.1, R12.2, R12.3, R11.2
- **Test Case:** Two candidate trajectories with precomputed `margin/3` facts; the sum favors the smoother candidate.

## 13. Predictive Safety

### R13.1 — Trajectory Collision Prediction
- **Description:** Checks the full predicted ego path against the full predicted paths of all tracked agents across all 6 timesteps, not just pairwise current-position checks, to catch collisions that only manifest a few steps into the future.
- **Traffic Law Basis:** Not statutory — implements the anticipatory element of the general collision-avoidance duty.
- **Input Facts Required:** `pos(ego,T,X,Y)` for T=0..5, `predicted_pos(V,T,X2,Y2)` for T=0..5 for every tracked agent.
- **Violation Condition:**
  ```
  violation(predicted_collision, ego, V, T) :- pos(ego,T,X,Y), predicted_pos(V,T,X2,Y2), D2=(X-X2)*(X-X2)+(Y-Y2)*(Y-Y2), D2 < CollisionThreshold2.
  ```
- **Detection Logic:** A direct squared-distance check applied at every timestep between ego's own plan and every other agent's independently-generated short-horizon prediction, rather than only checking the current timestep — this is the aggregation point that most of the specific R4.x rules feed into as a coarse full-horizon backstop.
- **Correction Strategy:** Identify the earliest violating timestep and trigger the most specific applicable rule (R4.1–R4.5) for that pairing; if none of the specific rules resolve it, fall back to a full re-plan via the emergency pipeline (R11.1/R11.2).
- **Priority Level:** Critical
- **Example Scenario:** Ego's path and another vehicle's predicted path don't conflict at T=0 or T=1 but converge to within collision distance at T=4.
- **ASP Complexity:** Complex
- **Performance Impact:** High (full cross product of horizon × agents)
- **Dependency:** R4.1–R4.5, R11.1
- **Test Case:** `predicted_pos(v15,4,1000,2000).` with `pos(ego,4,1050,2020)` giving `D2` under threshold.

### R13.2 — Time-to-Collision (TTC) Thresholds
- **Description:** Computes a discretized time-to-collision estimate for each ego/agent pair and classifies it against graduated thresholds, providing a finer-grained signal than the binary distance check in R13.1.
- **Traffic Law Basis:** Not statutory — standard AV safety-engineering metric operationalizing "imminence" for prioritization.
- **Input Facts Required:** `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)`, `speed(ego,T,S)`, `speed(V,T,SV)`, `closing_rate(ego,V,T,CR)`.
- **Violation Condition:**
  ```
  ttc_band(ego, V, T, imminent) :- closing_rate(ego,V,T,CR), CR > 0, gap_dist(ego,V,T,D), D * 100 <= CR * ImminentUnits.
  ttc_band(ego, V, T, critical) :- closing_rate(ego,V,T,CR), CR > 0, gap_dist(ego,V,T,D), D * 100 <= CR * CriticalUnits, D * 100 > CR * ImminentUnits.
  ttc_band(ego, V, T, threatening) :- closing_rate(ego,V,T,CR), CR > 0, gap_dist(ego,V,T,D), D * 100 <= CR * ThreateningUnits, D * 100 > CR * CriticalUnits.
  ```
- **Detection Logic:** Cross-multiplied integer comparison (`D/CR < threshold` rearranged to avoid division) sorts each pairing into one of three graduated bands rather than a single binary flag.
- **Correction Strategy:** `imminent` triggers R11.1 immediately; `critical` triggers the relevant R4.x hard correction; `threatening` triggers only the softer R4.5-style preemptive bias.
- **Priority Level:** Critical (imminent/critical bands), High (threatening band)
- **Example Scenario:** Closing rate and gap to a lead vehicle compute to a 1.2s TTC, landing in the `critical` band.
- **ASP Complexity:** Medium
- **Performance Impact:** Medium
- **Dependency:** R4.2, R4.3, R11.1, feeds R13.3's threat classification
- **Test Case:** `closing_rate(ego,v16,3,400). gap_dist(ego,v16,3,480).` against band thresholds `ImminentUnits=100, CriticalUnits=200, ThreateningUnits=300`.

### R13.3 — Threat Level Assessment (Aggregate Classification)
- **Description:** Combines TTC bands (R13.2), predicted-collision flags (R13.1), and hazard-distance results (R11.1) across all agents into a single per-timestep threat classification used to arbitrate which correction strategy runs first when multiple rules fire simultaneously.
- **Traffic Law Basis:** Not statutory — arbitration/prioritization logic layered on top of the individually statutory or safety-derived rules above.
- **Input Facts Required:** `ttc_band(ego,V,T,Band)`, `violation(predicted_collision,ego,V,T)`, `insufficient(emergency_brake,ego,H,T)`.
- **Violation Condition:**
  ```
  threat_level(ego, T, imminent) :- ttc_band(ego,_,T,imminent).
  threat_level(ego, T, imminent) :- insufficient(emergency_brake,ego,_,T).
  threat_level(ego, T, critical) :- ttc_band(ego,_,T,critical), not threat_level(ego,T,imminent).
  threat_level(ego, T, threatening) :- ttc_band(ego,_,T,threatening), not threat_level(ego,T,critical), not threat_level(ego,T,imminent).
  ```
- **Detection Logic:** A simple stratified default-negation cascade (Clingo handles this natively) that collapses multiple simultaneous signals into the single highest applicable level per timestep.
- **Correction Strategy:** The classified level selects which correction pipeline runs with top priority for that timestep (`imminent`→R11.x, `critical`→specific R4.x, `threatening`→R4.5-style bias), preventing two correction strategies from being applied in conflicting ways at the same timestep.
- **Priority Level:** Critical
- **Example Scenario:** At T=3, both a `ttc_band(...,critical)` and an `insufficient(emergency_brake,...)` fact hold simultaneously; the rule resolves this to a single `imminent` classification driving the emergency pipeline.
- **ASP Complexity:** Medium (stratified negation)
- **Performance Impact:** Low
- **Dependency:** R13.1, R13.2, R11.1
- **Test Case:** `ttc_band(ego,v16,3,critical). insufficient(emergency_brake,ego,h1,3).` → resolves to `threat_level(ego,3,imminent)`.

### R13.4 — Multi-Step Violation Prediction
- **Description:** Runs every applicable violation rule against the full 6-step predicted horizon up front (rather than only the current timestep) so corrections can be planned proactively instead of reactively at the moment a violation would occur.
- **Traffic Law Basis:** Not statutory — a computational-strategy rule ensuring the ASP layer's horizon-wide guarantee actually holds.
- **Input Facts Required:** All position/speed/signal facts across `T=0..5` (the full input set for every other rule in this catalog).
- **Violation Condition:**
  ```
  any_violation(ego) :- violation(_, ego, _).
  any_violation(ego) :- violation(_, ego, _, _).
  future_violation(ego, T) :- violation(Rule, ego, T), T > 0.
  ```
- **Detection Logic:** Not a new detection mechanism itself — a horizon-wide aggregation wrapper confirming that every rule above has, in fact, been evaluated at every `T` in `0..5` rather than only `T=0`, and surfacing how far into the horizon the earliest violation occurs.
- **Correction Strategy:** Drives the overall solve loop: if `future_violation(ego,T)` holds for any `T`, re-run the correction search starting from the earliest such `T` and re-check the corrected path against every rule again (fixpoint iteration) until a full 6-step horizon is clean or no further improvement is found within the solving time budget.
- **Priority Level:** Critical (as an integration/process guarantee, not a violation type of its own)
- **Example Scenario:** A rule that would only fire at T=5 (e.g. a distant red light) is caught and corrected during the initial solve rather than being missed because only T=0 was checked.
- **ASP Complexity:** Complex (drives the fixpoint loop over the entire rule set)
- **Performance Impact:** High
- **Dependency:** Every rule in this catalog
- **Test Case:** A synthetic scenario where `violation(red_light, ego, 5)` is the only violation present across the whole horizon, confirming it is still caught.

## 14. Environmental Context

### R14.1 — Wet Road Speed Reduction
- **Description:** Reduces effective speed limits and increases required following gaps when the road surface is detected or reported as wet, reflecting increased stopping distance.
- **Traffic Law Basis:** "Reasonable and prudent for conditions" statutes present in most jurisdictions' basic speed rules.
- **Input Facts Required:** `road_condition(T,wet)`, `speed_limit(L,Limit)`, `wet_speed_factor(Factor)`, `min_gap_table(S,MinD)`.
- **Violation Condition:**
  ```
  violation(wet_speed, ego, T) :- road_condition(T,wet), in_lane(ego,T,L), speed_limit(L,Limit), speed(ego,T,S), wet_speed_factor(Factor), S * 100 > Limit * Factor.
  ```
- **Detection Logic:** Applies an integer percentage factor (e.g. `Factor=80` for an 80% cap) to the posted limit rather than a floating multiplier, then compares via cross-multiplication.
- **Correction Strategy:** Substitute the reduced effective limit into R1.1's correction procedure, and scale up the R3.2 minimum-gap table lookup by the same factor to reflect longer wet-pavement stopping distances.
- **Priority Level:** High
- **Example Scenario:** Wet-road condition is flagged and the posted limit is 50 mph; effective cap becomes 40 mph but the plan holds at 48 mph.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R1.1, R3.2, R11.1 (braking-distance lookups should also be wet-adjusted)
- **Test Case:** `road_condition(2,wet). speed_limit(l1,2230). wet_speed_factor(80).` with `speed(ego,2,2000)`.

### R14.2 — Night Driving Behavior
- **Description:** Applies reduced-visibility-adjusted speed and increased following distance during night conditions, and requires stricter reaction-distance assumptions for any un-illuminated hazard type (pedestrians, cyclists, animals).
- **Traffic Law Basis:** "Overdriving your headlights" doctrine — statutes generally prohibit traveling faster than the distance illuminated by headlights allows for a safe stop.
- **Input Facts Required:** `time_of_day(T,night)`, `headlight_range(ego,Range)`, `speed(ego,T,S)`, `braking_distance(S,MaxD,BD)`.
- **Violation Condition:**
  ```
  violation(overdriving_headlights, ego, T) :- time_of_day(T,night), headlight_range(ego,Range), speed(ego,T,S), max_decel(ego,MaxD), braking_distance(S,MaxD,BD), BD > Range.
  ```
- **Detection Logic:** Reuses the braking-distance lookup from R11.1, comparing it directly against the headlight illumination range rather than a mapped hazard distance, since at night the relevant "hazard" is generically anything beyond visible range.
- **Correction Strategy:** Cap speed such that `BD <= Range`, functionally similar to R1.1/R14.1's correction but driven by illumination rather than a posted or wet-adjusted limit.
- **Priority Level:** High
- **Example Scenario:** Ego's headlights illuminate 60 m ahead but current speed implies a 75 m stopping distance.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R11.1, R1.5 (VRU proximity caps compound with this at night)
- **Test Case:** `time_of_day(2,night). headlight_range(ego,6000).` with `braking_distance(2000,80,7500)`.

### R14.3 — Visibility-Limited Rules (Fog, Glare, Sensor Degradation)
- **Description:** Applies a speed and following-distance reduction, and widens all proximity/clearance thresholds, whenever perception confidence is degraded by fog, sun glare, or sensor occlusion, independent of the nominal (clear-condition) detection ranges assumed elsewhere.
- **Traffic Law Basis:** "Reduce speed for conditions affecting visibility" statutes, extended here to cover a machine-perception-specific degradation the human-oriented statute doesn't explicitly name but the underlying duty covers.
- **Input Facts Required:** `perception_confidence(T,Level)`, `visibility_condition(T,Type)`, `speed(ego,T,S)`, `confidence_speed_cap(Level,Cap)`.
- **Violation Condition:**
  ```
  violation(low_visibility_speed, ego, T) :- perception_confidence(T,Level), confidence_speed_cap(Level,Cap), speed(ego,T,S), S > Cap.
  ```
- **Detection Logic:** Direct lookup from a discretized perception-confidence level (itself derived upstream from sensor-fusion diagnostics, fog/glare detection, etc.) to a speed cap, sidestepping any need to model the visibility physics inside the ASP layer itself.
- **Correction Strategy:** Apply the capped speed via the same mechanism as R1.1, and additionally widen the clearance thresholds used by R4.1/R4.2/R1.5 by a degradation-proportional margin for the duration of the low-confidence period.
- **Priority Level:** High (Critical if `Level=severe`)
- **Example Scenario:** Dense fog drops perception confidence to a "low" band; the plan continues at the nominal clear-condition speed.
- **ASP Complexity:** Simple
- **Performance Impact:** Low
- **Dependency:** R1.1, R4.1, R4.2, R1.5
- **Test Case:** `perception_confidence(2,low). confidence_speed_cap(low,1100).` with `speed(ego,2,1560)`.

### R14.4 — Weather-Dependent Threshold Adjustment
- **Description:** A generalized mechanism that scales multiple other rules' numeric thresholds (speed caps, gaps, clearances, braking-distance assumptions) together based on a single reported weather-severity level, ensuring consistency across rules rather than requiring each weather effect to be hand-tuned per rule.
- **Traffic Law Basis:** Umbrella "adjust for conditions" duty underlying R14.1–R14.3 individually.
- **Input Facts Required:** `weather_severity(T,Level)`, `threshold_adjustment(Level,RuleFamily,Factor)`.
- **Violation Condition:** None directly — this rule redefines the effective parameters consumed by other rules:
  ```
  effective_limit(L,T,AdjLimit) :- speed_limit(L,Limit), weather_severity(T,Level), threshold_adjustment(Level,speed,Factor), AdjLimit = Limit * Factor / 100.
  effective_min_gap(S,T,AdjGap) :- min_gap_table(S,MinD), weather_severity(T,Level), threshold_adjustment(Level,gap,Factor), AdjGap = MinD * Factor / 100.
  ```
- **Detection Logic:** Not a detection rule itself — a parameter-derivation layer that R1.1/R1.2/R3.2/R4.x etc. should consume `effective_limit`/`effective_min_gap` from rather than the raw static tables directly, whenever `weather_severity` facts are present.
- **Correction Strategy:** N/A directly — corrections happen in the consuming rules using the adjusted values.
- **Priority Level:** Medium
- **Example Scenario:** A `weather_severity(T,severe)` fact (e.g. heavy snow) is present; every consuming rule's threshold scales by the associated factor for that severity level in one consistent step rather than three separately-tuned rules disagreeing with each other.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R14.1, R14.2, R14.3, R1.1, R1.2, R3.2
- **Test Case:** `weather_severity(2,severe). threshold_adjustment(severe,speed,60).` with `speed_limit(l1,2230)` deriving `effective_limit(l1,2,1338)`.

## 15. Complex Interactions

### R15.1 — Multi-Vehicle Coordination at Shared Conflict Points
- **Description:** When multiple tracked vehicles and ego all approach a shared conflict point (e.g. a merge apex or uncontrolled intersection) within the same short window, resolves priority for all parties jointly rather than pairwise, avoiding a case where ego's pairwise-safe decision against Vehicle A conflicts with its pairwise-safe decision against Vehicle B.
- **Traffic Law Basis:** Extension of R6.1's uncontrolled-priority doctrine to the N-way case.
- **Input Facts Required:** `conflict_point(C,X,Y)`, `arrival_time(V,C,T)` for all approaching vehicles including ego, `approach_side(V,C,Side)`.
- **Violation Condition:**
  ```
  violation(multiway_priority, ego, T) :- conflict_point(C,X,Y), arrival_time(V,C,Tv), arrival_time(ego,C,Te), priority_rank(V,C) < priority_rank(ego,C), about_to_enter(ego,C,T), not has_cleared(V,C,T).
  priority_rank(V,C,Rank) :- arrival_time(V,C,T), Rank = #count{ V2 : arrival_time(V2,C,T2), (T2,V2) < (T,V) } + 1.
  ```
- **Detection Logic:** Computes a total order (`priority_rank`) across all approaching vehicles jointly using arrival time as primary key and a stable tiebreak, rather than resolving each pairwise relationship independently, which is what R6.1 alone would do and which can produce cycles with 3+ parties.
- **Correction Strategy:** Ego holds until every vehicle with a strictly lower `priority_rank` has cleared the conflict point.
- **Priority Level:** Critical
- **Example Scenario:** Three vehicles arrive at a four-way stop in quick succession from different approaches; pairwise reasoning alone could produce an inconsistent (cyclic) priority ordering.
- **ASP Complexity:** Complex (aggregate ranking)
- **Performance Impact:** High (grows with number of simultaneous approaching agents)
- **Dependency:** R6.1, R6.2
- **Test Case:** `arrival_time(v17,c1,1). arrival_time(v18,c1,1). arrival_time(ego,c1,2).` with a defined tiebreak ordering between `v17` and `v18`.

### R15.2 — Platoon/Convoy Behavior
- **Description:** When ego is operating as part of a coordinated platoon (a known, communicating group of vehicles maintaining tight formation), relaxes the standard following-distance floor (R3.2) in favor of a platoon-specific tight-gap policy contingent on verified inter-vehicle communication.
- **Traffic Law Basis:** Emerging platooning-specific statutory exceptions in some jurisdictions permitting reduced following distance under verified automated coordination.
- **Input Facts Required:** `platoon_member(ego,PlatoonId)`, `platoon_member(V,PlatoonId)`, `comm_link_active(ego,V,T)`, `lead_vehicle(ego,V,T)`.
- **Violation Condition:**
  ```
  violation(platoon_gap, ego, T) :- platoon_member(ego,PlatoonId), platoon_member(V,PlatoonId), lead_vehicle(ego,V,T), comm_link_active(ego,V,T), gap_dist(ego,V,T,D), D < PlatoonMinGap.
  violation(platoon_gap, ego, T) :- platoon_member(ego,PlatoonId), platoon_member(V,PlatoonId), lead_vehicle(ego,V,T), not comm_link_active(ego,V,T), gap_dist(ego,V,T,D), D < StandardMinGap.
  ```
- **Detection Logic:** Two-tier gap floor keyed on whether the communication link with the platoon lead is currently verified live; if the link drops, the rule immediately reverts to the standard (larger) R3.2-equivalent floor rather than continuing to permit the tight gap on stale coordination assumptions.
- **Correction Strategy:** Enforce whichever gap floor currently applies; a comm-link dropout should trigger an immediate gap-opening deceleration to reach `StandardMinGap`.
- **Priority Level:** Critical (comm-loss case), Medium (nominal tight-gap enforcement)
- **Example Scenario:** Platoon communication link to the lead vehicle drops mid-platoon; ego is still at the tight platoon gap and must open it immediately.
- **ASP Complexity:** Medium
- **Performance Impact:** Low
- **Dependency:** R3.1, R3.2 (this rule's fallback state), R4.2
- **Test Case:** `platoon_member(ego,p1). platoon_member(v19,p1). lead_vehicle(ego,v19,3).` with `not comm_link_active(ego,v19,3)`, `gap_dist(ego,v19,3,300)` vs `StandardMinGap=600`.

### R15.3 — Merge Lane Negotiation
- **Description:** Coordinates ego's merge with mainline traffic using a zipper-style alternating priority when a merge-lane ends, rather than treating it as a single unilateral lane-change decision (R2.2).
- **Traffic Law Basis:** Merge-point right-of-way statutes, often supplemented by "zipper merge" traffic-engineering guidance for high-volume merges.
- **Input Facts Required:** `merge_lane(L,EndPoint)`, `pos(ego,T,X,Y)`, `pos(V,T,X2,Y2)` for mainline traffic, `merge_gap_sequence(ego,V,Order)`.
- **Violation Condition:**
  ```
  violation(merge_negotiation, ego, T) :- in_lane(ego,T,L), merge_lane(L,EndPoint), approaching(ego,EndPoint,T), mainline_vehicle(V), merge_gap_sequence(ego,V,before), gap_dist(ego,V,T,D), D < MinMergeGap.
  ```
- **Detection Logic:** `merge_gap_sequence` is derived upstream (not shown here) from relative positions/speeds to assign an alternating "before/after" slot per mainline vehicle approaching the same merge point; the rule then checks whether ego is honoring its assigned slot with adequate gap once assigned to merge before a given vehicle.
- **Correction Strategy:** If assigned `before` a given mainline vehicle but the gap is insufficient, decelerate slightly to let that vehicle pass and re-target the next available slot (functionally similar to R2.2's gap-based delay, applied iteratively down the merge queue).
- **Priority Level:** High
- **Example Scenario:** A merge lane ends in 100 m with steady mainline traffic; ego's plan tries to merge ahead of a mainline vehicle with insufficient gap rather than waiting for the next slot.
- **ASP Complexity:** Complex
- **Performance Impact:** High
- **Dependency:** R2.2, R3.1
- **Test Case:** `merge_lane(l10,9000). approaching(ego,9000,3).` with `merge_gap_sequence(ego,v20,before)`, `gap_dist(ego,v20,3,250)` vs `MinMergeGap=500`.

### R15.4 — Traffic Wave Dampening
- **Description:** Generalizes the caravan-oscillation rule (R3.4) to actively counteract stop-and-go traffic-wave propagation by biasing ego's speed profile toward a smoothed, slightly-larger-than-minimum gap rather than tightly tracking the immediate lead vehicle's fluctuations, reducing the amplitude of speed oscillations passed to following traffic.
- **Traffic Law Basis:** Not statutory — a traffic-flow-engineering behavior (congestion wave/"phantom jam" mitigation) increasingly incorporated into AV behavioral guidelines.
- **Input Facts Required:** `speed(V,T,S)` for the lead vehicle across the horizon, `speed(ego,T,S)` across the horizon, `rolling_avg_speed(V,T,Avg)`.
- **Violation Condition:**
  ```
  violation(wave_amplification, ego, T) :- lead_vehicle(ego,V,T), speed(V,T,SV), rolling_avg_speed(V,T,Avg), speed(ego,T,S), |S - Avg| > |SV - Avg| * AmplificationFactor / 100.
  ```
- **Detection Logic:** Flags when ego's own speed deviates from the lead vehicle's rolling average by more than a bounded multiple of the lead vehicle's own deviation — i.e. ego is amplifying rather than damping the fluctuation it's observing.
- **Correction Strategy:** Re-target ego's speed toward `rolling_avg_speed(V,T,Avg)` rather than the lead vehicle's instantaneous speed, subject to still satisfying the hard gap floors of R3.1/R3.2/R4.2.
- **Priority Level:** Medium
- **Example Scenario:** Lead vehicle's speed oscillates ±5 mph around a 20 mph rolling average in congested traffic; ego's plan tracks the instantaneous speed and would compound the oscillation for vehicles behind it.
- **ASP Complexity:** Complex
- **Performance Impact:** Medium
- **Dependency:** R3.4, R3.1, R3.2
- **Test Case:** `rolling_avg_speed(v21,3,900). speed(v21,3,1050). speed(ego,3,1300).` against `AmplificationFactor=100` (ego should not exceed the lead's own deviation).

---

## Appendix: Rule Count and Priority Summary

| Category | # Rules | Critical | High | Medium | Low |
|---|---|---|---|---|---|
| 1. Speed Regulation | 5 | 1 | 3 | 1 | 0 |
| 2. Lane Discipline | 5 | 1 | 2 | 1 | 1 |
| 3. Following Behavior | 4 | 0 | 3 | 1 | 0 |
| 4. Collision Avoidance | 5 | 3 | 2 | 0 | 0 |
| 5. Traffic Control Compliance | 5 | 3 | 1 | 1 | 0 |
| 6. Right-of-Way | 5 | 5 | 0 | 0 | 0 |
| 7. Turning Rules | 5 | 2 | 2 | 1 | 0 |
| 8. Overtaking Rules | 4 | 2 | 2 | 0 | 0 |
| 9. Pedestrian & Cyclist Safety | 5 | 4 | 1 | 0 | 0 |
| 10. Intersection Behavior | 4 | 2 | 1 | 1 | 0 |
| 11. Emergency Situations | 4 | 4 | 0 | 0 | 0 |
| 12. Comfort & Ride Quality | 4 | 0 | 0 | 0 | 4 |
| 13. Predictive Safety | 4 | 4 | 0 | 0 | 0 |
| 14. Environmental Context | 4 | 0 | 2 | 1 | 1 |
| 15. Complex Interactions | 4 | 1 | 1 | 2 | 0 |
| **Total** | **63** | **32** | **20** | **9** | **6** |

## Implementation Notes

- **Grounding order:** Rules with `Performance Impact: High` (R2.2, R4.1, R6.1, R11.2, R13.1, R13.4, R15.1, R15.3) dominate solve time under Clingo when the tracked-agent count is large. Consider a pre-filtering stratum (coarse bounding-box or radius cull) before these rules ground, rather than letting Clingo's grounder enumerate all `V`×`T` pairs unfiltered.
- **Conflict resolution:** R13.3 (Threat Level Assessment) is the intended single arbitration point when multiple Critical rules fire on the same timestep; individual rule authors should assert into `override_active/2` and `threat_level/2` rather than hand-coding cross-rule priority checks.
- **Extending the schema:** Several rules above introduce facts beyond the five in the original schema (e.g. `pedestrian/4`, `zone/5`, `light_state/4` with a direction argument for R5.5). These should be treated as the actual required perception/map-ingestion outputs the surrounding system must produce — this catalog assumes they exist as inputs to the solver, not that the solver derives them from raw sensor data.
- **Testing:** Each rule's "Test Case" is a minimal fact set intended for unit-level Clingo regression tests (one rule firing in isolation); integration tests should additionally combine multiple rules' test cases to verify the priority/override interactions described in each rule's "Dependency" field.

---

## Implementation Plan

### Phase 1: Core Rules (24 rules)

| File | Rules | Description |
|------|-------|-------------|
| `rules/helpers.lp` | - | Shared predicates (dist2, heading, perp_dist2, inside_zone, curvature_radius) |
| `rules/speed_regulation.lp` | R1.1–R1.5 | Urban/highway speed limits, variable zones, min speed, VRU proximity |
| `rules/lane_discipline.lp` | R2.1–R2.5 | Lane keeping, lane change permission, wrong-way, HOV, departure warning |
| `rules/following_behavior.lp` | R3.1–R3.4 | Time-based following, speed-dependent floor, tailgating, queue behavior |
| `rules/collision_avoidance.lp` | R4.1–R4.5 | Lateral/longitudinal clearance, cut-in, rear-end risk, side-swipe |
| `rules/traffic_control.lp` | R5.1–R5.5 | Red/yellow/green light, flashing signals, protected arrows |

### Phase 2: Extended Rules (23 rules)

| File | Rules | Description |
|------|-------|-------------|
| `rules/right_of_way.lp` | R6.1–R6.5 | Uncontrolled intersection, roundabout, turn yield, emergency vehicle, crosswalk |
| `rules/turning_rules.lp` | R7.1–R7.5 | Right-on-red, left turn, U-turn, signal timing, turning radius |
| `rules/overtaking.lp` | R8.1–R8.4 | Permitted conditions, no-passing zones, right-side pass, clearance |
| `rules/vru_safety.lp` | R9.1–R9.5 | Crosswalks, school zones, bike lanes, blind spots, sidewalks |
| `rules/intersection_behavior.lp` | R10.1–R10.4 | Queue position, box blocking, protected left, right-turn yield |

### Phase 3: Advanced Rules (16 rules)

| File | Rules | Description |
|------|-------|-------------|
| `rules/emergency.lp` | R11.1–R11.4 | Braking feasibility, evasive maneuver, guardrails, construction zones |
| `rules/comfort_quality.lp` | R12.1–R12.4 | Accel/decel limits, lateral accel/jerk, smooth path, comfort scoring |
| `rules/predictive_safety.lp` | R13.1–R13.4 | Trajectory collision prediction, TTC thresholds, threat assessment, multi-step |
| `rules/environmental.lp` | R14.1–R14.4 | Wet road, night driving, visibility, weather-dependent thresholds |
| `rules/complex_interactions.lp` | R15.1–R15.4 | Multi-vehicle coordination, platoon, merge negotiation, traffic waves |

### Supporting Changes

| File | Change |
|------|--------|
| `rules/basic.lp` | Update to include all 16 rule files |
| `src/checker.py` | Add new input fact support (zones, pedestrians, weather, etc.) |
| `tests/test_rules.py` | Unit tests per category |

### Implementation Order

1. Create `helpers.lp` with all shared predicates
2. Implement Phase 1 files (R1–R5)
3. Implement Phase 2 files (R6–R10)
4. Implement Phase 3 files (R11–R15)
5. Update `basic.lp` to include all files
6. Update `checker.py` for new input facts
7. Create `tests/test_rules.py`
8. Run full integration tests

### Key Constraints

- All arithmetic is integer-only (no floats, no sqrt)
- Squared distances for all proximity checks
- Cross-multiplication for ratio comparisons
- Each rule file is self-contained with its own violation predicates
- Shared predicates defined once in `helpers.lp`

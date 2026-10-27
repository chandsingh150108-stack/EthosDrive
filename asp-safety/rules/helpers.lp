% helpers.lp - Shared predicates for the 63-rule ASP traffic rule catalog
% All coordinates in centimeters (scaled x100). Integer-only arithmetic.
% Squared distances replace Euclidean distance (no sqrt). Clingo uses
% 32-bit integers, so all comparisons are written to stay below ~2e9.
% Timesteps: T in 0..5 (6 total).
%
% Speed semantics (GAP-3/GAP-9): speed(V,T,S) is the SQUARED speed in
% (cm/timestep)^2, derived from consecutive positions. Limit/cap facts
% (speed_limit, zone_speed_limit, vru_speed_cap, min_speed_limit,
% confidence_speed_cap) stay LINEAR (cm/timestep); rule bodies square
% them at comparison time (S2 > Limit*Limit).

% Time horizon
time(0..5).

% Alias: point(V,T,X,Y) mirrors pos(V,T,X,Y)
point(V, T, X, Y) :- pos(V, T, X, Y).

% Squared distance between two vehicles at time T
dist2(V1, V2, T, D) :-
    pos(V1, T, X1, Y1), pos(V2, T, X2, Y2), V1 != V2,
    D = (X1-X2)*(X1-X2) + (Y1-Y2)*(Y1-Y2).

% Heading vector from consecutive positions (safe: pos binds all variables)
heading(V, T, DX, DY) :-
    pos(V, T, X, Y), pos(V, T+1, X2, Y2),
    DX = X2 - X, DY = Y2 - Y.

% Velocity from consecutive positions (centimeters/timestep)
velocity(V, T, VX, VY) :-
    pos(V, T, X, Y), pos(V, T-1, X0, Y0), time(T), T > 0,
    VX = X - X0, VY = Y - Y0.

% Velocity squared (avoids sqrt for speed comparisons)
velocity_sq(V, T, V2) :-
    velocity(V, T, VX, VY), time(T), V2 = VX*VX + VY*VY.

% Squared-speed aliases (GAP-3): catalog rule bodies use velocity_sq/3
% and speed2/3 directly. speed/3 is a legacy alias kept for extras
% compatibility; no catalog rule references it.
speed2(V, T, S2) :- velocity_sq(V, T, S2).
speed(V, T, S2) :- speed2(V, T, S2).

% === Lane membership (axis-aligned segments; see note below) ===
% Python-computed membership for arbitrary orientations (MC-1 option
% C): checker.py emits in_lane_fact/3 via exact point-segment
% distance. Either derivation suffices (both are sound).
in_lane(V, T, L) :- in_lane_fact(V, T, L).
% in_lane/3 holds when a vehicle position is within half lane width of
% an axis-aligned lane segment (plus a W^2 tolerance past segment ends).
% Diagonal segments are a documented limitation (general perpendicular
% distance needs 4th powers that overflow 32-bit integers).
% Gated on use_asp_in_lane/0 (perf): every in_lane consumer in the
% catalog pairs with array-derived facts (gap/longitudinal/oncoming)
% or array trajectories, all covered by in_lane_fact/3. The geometric
% clauses only matter for hand-written pos/4 extras, so the checker
% asserts the flag exactly then -- otherwise 500+ segments x dozens
% of positions ground for nothing (~60 ms on real scenes).
in_lane(V, T, L) :-
    use_asp_in_lane,
    pos(V, T, X, Y), lane(L, X1, Y1, X2, Y2), lane_width(L, W),
    Y1 == Y2, D = Y - Y1, D*D*4 <= W*W,
    (X-X1)*(X-X2) <= W*W, time(T).
in_lane(V, T, L) :-
    use_asp_in_lane,
    pos(V, T, X, Y), lane(L, X1, Y1, X2, Y2), lane_width(L, W),
    X1 == X2, D = X - X1, D*D*4 <= W*W,
    (Y-Y1)*(Y-Y2) <= W*W, time(T).

% Lane change between consecutive timesteps (R2.2, R8.1)
changes_lane(V, T, L1, L2) :-
    in_lane(V, T, L1), in_lane(V, T+1, L2), L1 != L2, time(T), T < 5.

% === Conflict zones unify intersections and conflict points ===
% Roundabouts count as conflict zones too (R6.2 entry needs
% about_to_enter/arrival_time/has_cleared; without this R6.2 could
% never fire since roundabout/4 is not an intersection/4).
conflict_zone(C, X, Y, R) :- intersection(C, X, Y, R).
conflict_zone(C, X, Y, 1000) :- conflict_point(C, X, Y).
conflict_zone(R, Xc, Yc, Radius) :- roundabout(R, Xc, Yc, Radius).

% First timestep a vehicle comes within the zone radius (R6.1, R15.1)
arrival_time(V, Z, T) :-
    vehicle(V), conflict_zone(Z, Xc, Yc, R), pos(V, T, X, Y), time(T),
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= R*R,
    not arrived_earlier(V, Z, T).
arrived_earlier(V, Z, T) :-
    vehicle(V), conflict_zone(Z, Xc, Yc, R),
    pos(V, T2, X, Y), pos(V, T, _, _), time(T), time(T2), T2 < T,
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= R*R.

% Static lane membership for rules without a time index (R7.2 lane_of).
lane_of(V, L) :- in_lane(V, 0, L).

% Vehicle has exited the zone: arrived at Tv < T and is now outside.
% Lets R6.1/R15.1 stand down for traffic that already passed through.
% (IF-2: previously never derived, so every arrived vehicle blocked
% ego forever.)
has_cleared(V, Z, T) :-
    arrival_time(V, Z, Tv), Tv < T, time(T),
    conflict_zone(Z, Xc, Yc, R), pos(V, T, X, Y),
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY > R*R.

% Vehicle in the approach band (inside 3R, outside R) AND moving toward
% the center (P-4): the bare distance band cannot tell approaching from
% leaving. approaching_zone/3 holds when the motion vector (heading for T<5,
% velocity for T>0 -- heading needs T+1, velocity needs T-1) has a
% positive component toward the zone center. A stopped vehicle (zero
% motion) is holding at the line, not about to enter.
approaching_zone(V, Z, T) :-
    vehicle(V), conflict_zone(Z, Xc, Yc, R), pos(V, T, X, Y),
    heading(V, T, HX, HY), time(T), T < 5,
    (Xc-X)*HX + (Yc-Y)*HY > 0.
approaching_zone(V, Z, T) :-
    vehicle(V), conflict_zone(Z, Xc, Yc, R), pos(V, T, X, Y),
    velocity(V, T, VX, VY), time(T), T > 0,
    (Xc-X)*VX + (Yc-Y)*VY > 0.
about_to_enter(V, Z, T) :-
    vehicle(V), conflict_zone(Z, Xc, Yc, R), pos(V, T, X, Y), time(T),
    DX = X - Xc, DY = Y - Yc, D2 = DX*DX + DY*DY,
    D2 <= 9*R*R, D2 > R*R,
    approaching_zone(V, Z, T).
about_to_enter(V, Z, T) :-
    intersection(Z, _, _, _), vehicle(V),
    conflict_zone(Z, Xc, Yc, R), pos(V, T, X, Y), time(T),
    DX = X - Xc, DY = Y - Yc, D2 = DX*DX + DY*DY,
    D2 <= 9*R*R, D2 > R*R,
    at_stopline(V, Z, T).
% Holding at the stop line (dist <= 1.5R) counts as about to enter even
% with no motion vector: RTOR gates (R7.1) govern the stopped-at-line
% maneuver itself, which has no heading toward the center. Leaving
% from mid-band stays quiet (neither clause holds); leaving from
% inside 1.5R conservatively still counts.

% Ordering of two vehicles relative to a zone center (R10.1)
behind(V1, V2, Z, T) :-
    pos(V1, T, X1, Y1), pos(V2, T, X2, Y2),
    conflict_zone(Z, Xc, Yc, R),
    D1 = (X1-Xc)*(X1-Xc) + (Y1-Yc)*(Y1-Yc),
    D2 = (X2-Xc)*(X2-Xc) + (Y2-Yc)*(Y2-Yc),
    D1 > D2, time(T).

% === Motion state derivations ===
decelerating(V, T) :-
    speed2(V, T, S), speed2(V, T+1, S2), time(T), T < 5, S2 < S.
stopped(V, T) :- speed2(V, T, S2), time(T), S2 == 0.
% T=0 has no velocity (needs T-1): close the edge with heading (needs
% T+1). A trajectory starting static is stopped at T=0; otherwise none
% of stopped_before/has_stopped can ever hold at the first timestep.
stopped(V, 0) :- heading(V, 0, 0, 0), time(0).
has_stopped(V, Z) :- stopped(V, T), at_stopline(V, Z, T).

% stopped_before/3 requires a full stop NEAR the feature (P-4), not
% anywhere on the map: a stop 2 km away must not satisfy a crosswalk
% yield (used by R6.5, R9.1, R9.2, R10.4). Proximity uses the same 5 m
% tolerance pattern as path_intersects/3.
stopped_before(V, C, T) :-
    stopped(V, T), path_intersects(V, C, T).
stopped_before(V, I, T) :-
    stopped(V, T), stopline_dist2(V, I, T, D2),
    intersection(I, _, _, R), D2 <= (R+500)*(R+500), time(T).
stopped_before(V, Z, T) :-
    stopped(V, T), zone(Z, X1, Y1, X2, Y2, _), pos(V, T, X, Y),
    (X-X1)*(X-X2) <= 250000, (Y-Y1)*(Y-Y2) <= 250000, time(T).
stopped_before(V, C, T) :-
    stopped(V, T), conflict_point(C, X, Y), pos(V, T, X2, Y2),
    DX = X2 - X, DY = Y2 - Y,
    DX*DX + DY*DY <= 1500*1500, time(T).

% === Intersection proximity ===
at_intersection(V, I, T) :-
    pos(V, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= R*R.
at_stopline(V, I, T) :-
    pos(V, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    DX = X - Xc, DY = Y - Yc, DX*DX*4 + DY*DY*4 <= R*R*9.
stopline_dist2(V, I, T, D2) :-
    pos(V, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    D2 = (X-Xc)*(X-Xc) + (Y-Yc)*(Y-Yc).
obstacle_in_box(V, I, T) :-
    pos(V, T, X, Y), intersection(I, Xc, Yc, R), time(T), V != ego,
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= R*R.
% Pedestrians occupy the box too (R5.3/R7.1 clear-path checks cover
% conflicting pedestrians per spec, not just vehicles).
obstacle_in_box(V, I, T) :-
    pedestrian(V, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= R*R.

% Clear path holds when no obstacle occupies the box (R5.3, R7.1)
clear_path(V, I, T) :- about_to_enter(V, I, T), not obstacle_in_box(_, I, T).

% Yielding maneuver is satisfied by a full stop (R6.4). Stopped-anywhere
% is INTENTIONAL here (unlike stopped_before/3): an emergency yield is a
% pull-over-and-stop with no stop line involved. See DC-4 log.
yielding_maneuver(V, T) :- stopped(V, T).

% === Path/crosswalk overlap (expanded by 500cm tolerance) ===
path_intersects(V, C, T) :-
    pos(V, T, X, Y), crosswalk(C, X1, Y1, X2, Y2), time(T),
    (X-X1)*(X-X2) <= 250000, (Y-Y1)*(Y-Y2) <= 250000.
path_intersects(V, I, T) :-
    pos(V, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= (R+500)*(R+500).

% Longitudinal gap is checker-provided (P-4): longitudinal_gap/4 is the
% SIGNED along-track separation in cm (+ = ahead). Fall back to the
% Euclidean gap_dist/4 only for pairs with no projection fact
% (has_longitudinal_gap/3 marker). R4.2 consumes this; R3/R10/R13/R15
% keep total-separation gap_dist/4 by design.
longitudinal_gap(V1, V2, T, D) :-
    gap_dist(V1, V2, T, D), not has_longitudinal_gap(V1, V2, T).

% Turn crossing exempts brief bike-lane crossings during a signaled turn (R9.3)
turn_crossing(V, T, L) :-
    planned_turn(V, _), in_lane(V, T, L),
    in_lane(V, T-1, L2), L != L2, time(T), T > 0.
turn_crossing(V, T, L) :-
    planned_turn(V, _), in_lane(V, T, L),
    in_lane(V, T+1, L2), L != L2, time(T), T < 5.

% === Speed bands with conservative stop/gap needs (int32-safe) ===
% Bands on squared speed: slow <= (500cm/s)^2, med <= (1265cm/s)^2.
% Single-rule fact-join form: separate same-head range rules trigger
% slow domain inference in the grounder (50ms+), this grounds in <1ms.
speed_band_def(slow, 0, 250000).
speed_band_def(med, 250001, 1600000).
speed_band_def(fast, 1600001, 2000000000).
speed_band(V, T, Band) :-
    velocity_sq(V, T, S2), speed_band_def(Band, Lo, Hi),
    S2 >= Lo, S2 <= Hi, time(T).
% Conservative stopping distance per band (cm, linear).
stop_need(slow, 500).
stop_need(med, 2000).
stop_need(fast, 6000).
% Minimum following gap per band (cm, linear).
follow_gap_min(slow, 500).
follow_gap_min(med, 1000).
follow_gap_min(fast, 2000).

% === Solver output selection (P-8) ===
% check() collects shown atoms only: iterating the full model (every
% pos/gap fact) costs ~15 ms of Python per solve on 20-agent scenes.
#show violation/3.
#show violation/4.
#show hazard/3.
#show hazard/4.
#show warning/3.
#show insufficient/4.
#show ttc_band/4.
#show threat_level/3.
#show comfort_score/2.
#show future_violation/2.
#show any_violation/1.

% === Constraint atoms for ASP-guided correction (P-11) ===
% All values in centimeters (scaled x100) or cm/timestep.
% Emitted by rules alongside violation heads to guide continuous repair.
#show constraint_v_max/3.
#show constraint_v_min/3.
#show constraint_x_bounds/4.
#show constraint_y_bounds/4.
#show constraint_a_max/3.
#show constraint_lane_center/3.
% === Coupled range constraints (P-23 increment 3) ===
% constraint_*_range(Vehicle, T0, T1, Min_cm, Max_cm) — corridor
#show constraint_x_bounds_range/5.
#show constraint_y_bounds_range/5.

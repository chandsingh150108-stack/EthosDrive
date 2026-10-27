% vru_safety.lp - Rules R9.1-R9.5
% Covers: crosswalks, school zones, bike lanes, blind spots, sidewalks.
% pedestrian/4 and vru/5 facts are emitted by checker.py.

% === R9.1 General Crosswalk Yielding (unmarked corners) ===
% A pedestrian within 15m of an intersection center while ego's path
% reaches that intersection.
violation(implied_crosswalk_yield, ego, T) :-
    pedestrian(P, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= 2250000,
    path_intersects(ego, I, T), not stopped_before(ego, I, T).

% === R9.2 School Zone Pedestrian Behavior ===
% Crossing-guard stop compels a stop independent of signal state.
violation(school_guard_stop, ego, T) :-
    crossing_guard_signal(Z, T, stop), pos(ego, T, X, Y), time(T),
    zone(Z, X1, Y1, X2, Y2, school),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    not stopped_before(ego, Z, T).

% === R9.3 Bicycle Lane Respect ===
violation(bike_lane_encroachment, ego, T) :-
    in_lane(ego, T, L), lane_type(L, bicycle),
    not turn_crossing(ego, T, L).

% === R9.4 Blind Spot Detection for VRUs ===
% Currently-tracked VRU inside the maneuver-side blind-spot polygon.
violation(blind_spot_maneuver, ego, T) :-
    planned_maneuver(ego, lane_change, T), maneuver_direction(ego, T, Side),
    blind_spot_zone(ego, Side, X1, Y1, X2, Y2),
    vru(P, T, X, Y, _),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0.
% Stale-detection fallback: last known position still inside the zone.
% Default staleness threshold 10 (1.0s).
stale_detection_threshold(10).
violation(blind_spot_maneuver, ego, T) :-
    planned_maneuver(ego, lane_change, T), maneuver_direction(ego, T, Side),
    blind_spot_zone(ego, Side, X1, Y1, X2, Y2),
    last_confident_detection_age(P, T, Age), stale_detection_threshold(S),
    Age > S, last_known_pos(P, X, Y),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0.

% === R9.5 Sidewalk Encroachment Prevention ===
violation(sidewalk_encroachment, ego, T) :-
    pos(ego, T, X, Y), sidewalk(S, X1, Y1, X2, Y2), time(T),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    not driveway_crossing(ego, T).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% VRU holds are v_max 0 (stop and hold until the pedestrian / guard
% signal clears).
% R9.1 unmarked-corner yield.
constraint_v_max(ego, T, 0) :-
    pedestrian(P, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    DX = X - Xc, DY = Y - Yc, DX*DX + DY*DY <= 2250000,
    path_intersects(ego, I, T), not stopped_before(ego, I, T).
% R9.2 school-guard stop (overrides signal state by design).
constraint_v_max(ego, T, 0) :-
    crossing_guard_signal(Z, T, stop), pos(ego, T, X, Y), time(T),
    zone(Z, X1, Y1, X2, Y2, school),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    not stopped_before(ego, Z, T).
% R9.4 blind spot: caution cap (5 m/s) to delay the lane change until
% the zone is clear or a fresh detection confirms it (both the
% tracked-VRU and stale-detection bodies).
constraint_v_max(ego, T, 500) :-
    planned_maneuver(ego, lane_change, T), maneuver_direction(ego, T, Side),
    blind_spot_zone(ego, Side, X1, Y1, X2, Y2),
    vru(P, T, X, Y, _),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0.
constraint_v_max(ego, T, 500) :-
    planned_maneuver(ego, lane_change, T), maneuver_direction(ego, T, Side),
    blind_spot_zone(ego, Side, X1, Y1, X2, Y2),
    last_confident_detection_age(P, T, Age), stale_detection_threshold(S),
    Age > S, last_known_pos(P, X, Y),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0.
% R9.3 bike-lane / R9.5 sidewalk: NO constraint emitted
% (documented exemption): both are exclusion geometries (stay OUT of a polygon);
% an allowed-region box around the roadway side is unknowable in ASP
% without the adjacent-lane target. The corrector re-projects (P4).

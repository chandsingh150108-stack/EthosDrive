% right_of_way.lp - Rules R6.1-R6.5
% Covers: uncontrolled intersection, roundabout, turn yield, emergency, crosswalk.
% arrival_time/3 and about_to_enter/3 are derived in helpers.lp from
% conflict zones (intersections + conflict points).

% === R6.1 Uncontrolled Intersection Priority ===
% Strict arrival order: another vehicle arrived earlier and has not cleared.
violation(uncontrolled_priority, ego, T) :-
    about_to_enter(ego, Z, T),
    arrival_time(V, Z, Tv), arrival_time(ego, Z, Te), V != ego,
    Tv < Te, not has_cleared(V, Z, T).
% Simultaneous arrival tiebreak: yield (conservative) to any tied vehicle.
violation(uncontrolled_priority, ego, T) :-
    about_to_enter(ego, Z, T),
    arrival_time(V, Z, Te), arrival_time(ego, Z, Te), V != ego,
    not has_cleared(V, Z, T).

% === R6.2 Roundabout Entry Priority ===
% Circulating traffic has absolute priority; approach_conflict is
% satisfied by any circulating vehicle (conservative gap model).
violation(roundabout_entry, ego, T) :-
    about_to_enter(ego, R, T), roundabout(R, _, _, _),
    in_circulation(V, R, T), V != ego.

% === R6.3 Yield to Oncoming Traffic During Turns ===
% SafeTurnWindow default 30 (3.0s at 10 units/s encoding).
safe_turn_window(30).
violation(turn_yield, ego, T) :-
    planned_turn(ego, left),
    oncoming(V, ego, T), time_to_conflict_point(V, T, TTC),
    turn_duration(ego, Est), safe_turn_window(M),
    TTC < Est + M, time(T).

% === R6.4 Emergency Vehicle Yielding ===
violation(emergency_yield, ego, T) :-
    emergency_vehicle(V, T, X, Y, active), approaching(V, ego, T),
    not yielding_maneuver(ego, T).

% === R6.5 Pedestrian Crosswalk Priority ===
violation(crosswalk_yield, ego, T) :-
    pedestrian(P, T, X, Y), crosswalk(C, X1, Y1, X2, Y2), time(T),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    path_intersects(ego, C, T), not stopped_before(ego, C, T).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% Priority holds are v_max 0 (stop at the line / yield point and
% re-evaluate each timestep as the prioritized party clears).
% R6.1 strict-order + simultaneous-tiebreak bodies.
constraint_v_max(ego, T, 0) :-
    about_to_enter(ego, Z, T),
    arrival_time(V, Z, Tv), arrival_time(ego, Z, Te), V != ego,
    Tv < Te, not has_cleared(V, Z, T).
constraint_v_max(ego, T, 0) :-
    about_to_enter(ego, Z, T),
    arrival_time(V, Z, Te), arrival_time(ego, Z, Te), V != ego,
    not has_cleared(V, Z, T).
% R6.2 roundabout entry: hold at the yield line.
constraint_v_max(ego, T, 0) :-
    about_to_enter(ego, R, T), roundabout(R, _, _, _),
    in_circulation(V, R, T), V != ego.
% R6.3 turn yield: hold in the turn pocket until the oncoming window
% covers the full turn duration + margin.
constraint_v_max(ego, T, 0) :-
    planned_turn(ego, left),
    oncoming(V, ego, T), time_to_conflict_point(V, T, TTC),
    turn_duration(ego, Est), safe_turn_window(M),
    TTC < Est + M, time(T).
% R6.4 emergency yield: stop component of the pull-over-and-stop
% response (the lateral shift itself is the corrector's job, P4).
constraint_v_max(ego, T, 0) :-
    emergency_vehicle(V, T, X, Y, active), approaching(V, ego, T),
    not yielding_maneuver(ego, T).
% R6.5 crosswalk: hold short of the crosswalk boundary.
constraint_v_max(ego, T, 0) :-
    pedestrian(P, T, X, Y), crosswalk(C, X1, Y1, X2, Y2), time(T),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    path_intersects(ego, C, T), not stopped_before(ego, C, T).

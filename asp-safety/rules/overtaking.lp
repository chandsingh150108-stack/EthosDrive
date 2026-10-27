% overtaking.lp - Rules R8.1-R8.4
% Covers: permitted conditions, no-passing zones, right-side pass, clearance.

% === R8.1 Permitted Overtaking Conditions ===
% Solid-boundary crossing during an overtake.
violation(unsafe_overtake, ego, T) :-
    planned_maneuver(ego, overtake, T),
    changes_lane(ego, T, L1, L2), lane_boundary(L1, L2, solid).
% Oncoming conflict within maneuver duration + margin (default 20 = 2.0s).
overtake_margin(20).
violation(unsafe_overtake, ego, T) :-
    planned_maneuver(ego, overtake, T), oncoming(V, ego, T),
    time_to_conflict_point(V, T, TTC), overtake_duration(ego, Est),
    overtake_margin(M), TTC < Est + M, time(T).

% === R8.2 No-Passing Zones ===
violation(no_passing_zone, ego, T) :-
    planned_maneuver(ego, overtake, T), pos(ego, T, X, Y), time(T),
    no_passing_zone(Z, X1, Y1, X2, Y2),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0.
violation(no_passing_sightline, ego, T) :-
    planned_maneuver(ego, overtake, T), in_lane(ego, T, L),
    sightline_distance(L, SD), speed_limit(L, Limit),
    required_sightline(Limit, ReqSD), SD < ReqSD.

% === R8.3 Overtaking on the Right (two-lane roads only) ===
violation(right_overtake, ego, T) :-
    planned_maneuver(ego, overtake, T), overtake_side(ego, T, right),
    in_lane(ego, T, L), road_type(L, two_lane),
    not lead_vehicle_turning(_, left).

% === R8.4 Overtaking Clearance Requirements (decimeter units) ===
% Per-type lateral minimums; cyclist/pedestrian minima are strictest.
overtake_clearance_min(vehicle, 150).
overtake_clearance_min(cyclist, 90).
overtake_clearance_min(pedestrian, 150).
violation(overtake_clearance, ego, V, T) :-
    planned_maneuver(ego, overtake, T), overtaken_type(V, Type),
    overtake_clearance_min(Type, MinL),
    pos(ego, T, X, Y), pos(V, T, X2, Y2),
    pos(ego, T2, Xh, Yh), T2 == T + 1, time(T), T < 5,
    HX = (Xh-X)/10, HY = (Yh-Y)/10, H2 = HX*HX + HY*HY, H2 > 0,
    DY = HY*((X2-X)/10) - HX*((Y2-Y)/10),
    ML = MinL/10, DY*DY < ML*ML*H2.

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% Aborted overtakes fall back to following (R3.1/R3.2 take over): a
% caution cap (5 m/s) kills the pass momentum. This is intentionally
% weak for the solid-line case (the hard veto stays the violation
% itself); the maneuver-cancel geometry is the corrector's job (P4).
% R8.1 solid-boundary + oncoming-timing bodies.
constraint_v_max(ego, T, 500) :-
    planned_maneuver(ego, overtake, T),
    changes_lane(ego, T, L1, L2), lane_boundary(L1, L2, solid).
constraint_v_max(ego, T, 500) :-
    planned_maneuver(ego, overtake, T), oncoming(V, ego, T),
    time_to_conflict_point(V, T, TTC), overtake_duration(ego, Est),
    overtake_margin(M), TTC < Est + M, time(T).
% R8.2 no-passing zone + sightline bodies.
constraint_v_max(ego, T, 500) :-
    planned_maneuver(ego, overtake, T), pos(ego, T, X, Y), time(T),
    no_passing_zone(Z, X1, Y1, X2, Y2),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0.
constraint_v_max(ego, T, 500) :-
    planned_maneuver(ego, overtake, T), in_lane(ego, T, L),
    sightline_distance(L, SD), speed_limit(L, Limit),
    required_sightline(Limit, ReqSD), SD < ReqSD.
% R8.3 right-side pass: cancel to following.
constraint_v_max(ego, T, 500) :-
    planned_maneuver(ego, overtake, T), overtake_side(ego, T, right),
    in_lane(ego, T, L), road_type(L, two_lane),
    not lead_vehicle_turning(_, left).
% R8.4 clearance: lateral band around the overtaken party (mirrors
% the R4.1/R4.5 precedent pattern).
constraint_y_bounds(ego, T, Y2-MinL, Y2+MinL) :-
    planned_maneuver(ego, overtake, T), overtaken_type(V, Type),
    overtake_clearance_min(Type, MinL),
    pos(ego, T, X, Y), pos(V, T, X2, Y2),
    pos(ego, T2, Xh, Yh), T2 == T + 1, time(T), T < 5,
    HX = (Xh-X)/10, HY = (Yh-Y)/10, H2 = HX*HX + HY*HY, H2 > 0,
    DY = HY*((X2-X)/10) - HX*((Y2-Y)/10),
    ML = MinL/10, DY*DY < ML*ML*H2.

% turning_rules.lp - Rules R7.1-R7.5
% Covers: right-on-red, left turn, U-turn, signal timing, turning radius.

% === R7.1 Right Turn on Red (three independent gates) ===
violation(rtor_no_sign, ego, T) :-
    planned_turn(ego, right), light_state(I, red, T),
    no_turn_on_red(I, right), about_to_enter(ego, I, T).
violation(rtor_no_stop, ego, T) :-
    planned_turn(ego, right), light_state(I, red, T),
    about_to_enter(ego, I, T), not has_stopped(ego, I).
violation(rtor_unclear, ego, T) :-
    planned_turn(ego, right), light_state(I, red, T),
    about_to_enter(ego, I, T), not clear_path(ego, I, T).

% === R7.2 Left Turn Across Traffic (every crossed oncoming lane) ===
% Safety margin default 10 (1.0s).
left_turn_margin(10).
violation(left_turn_across, ego, T) :-
    planned_turn(ego, left), oncoming_lane(L, I), oncoming(V, ego, T),
    lane_of(V, L), time_to_conflict_point(V, T, TTC),
    turn_duration(ego, Est), left_turn_margin(M),
    TTC < Est + M, time(T).

% === R7.3 U-Turn Restrictions ===
violation(u_turn_prohibited, ego, T) :-
    planned_maneuver(ego, u_turn, T), in_lane(ego, T, L), no_u_turn(L).
violation(u_turn_sightline, ego, T) :-
    planned_maneuver(ego, u_turn, T), in_lane(ego, T, L),
    sightline_distance(L, SD), speed_limit(L, Limit),
    required_sightline(Limit, ReqSD), SD < ReqSD.

% === R7.4 Turn Signal Timing ===
% Signal must be active at every timestep up to the maneuver start.
% Default lead: maneuver at T requires signal on [0..T].
violation(late_signal, ego, T) :-
    planned_maneuver(ego, _, T), maneuver_direction(ego, T, Direction),
    time(T2), T2 <= T, not signal_active(ego, Direction, T2).

% === R7.5 Turning Radius Compliance (meter units, int32-safe) ===
% Fits a circle through 3 consecutive points in meter units
% (coord/100); violation when R < Rmin, i.e. A2*B2*C2 < 4*CR2*RM^2.
violation(turn_radius, ego, T) :-
    pos(ego, T2, X0, Y0), T2 == T - 1,
    pos(ego, T, X1, Y1), pos(ego, T2b, X2, Y2), T2b == T + 1,
    time(T), T > 0, T < 5, min_turning_radius(ego, Rmin),
    AX = (X1-X0)/100, AY = (Y1-Y0)/100,
    BX = (X2-X1)/100, BY = (Y2-Y1)/100,
    CX = (X2-X0)/100, CY = (Y2-Y0)/100,
    A2 = AX*AX + AY*AY, B2 = BX*BX + BY*BY, C2 = CX*CX + CY*CY,
    CR = AX*BY - AY*BX, CR2 = CR*CR, RM = Rmin/100,
    A2 > 0, B2 > 0, CR2 > 0, A2*B2*C2 < 4*CR2*RM*RM.

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% Turn holds are v_max 0 (delay the maneuver until its gates clear).
% R7.1 three RTOR gates.
constraint_v_max(ego, T, 0) :-
    planned_turn(ego, right), light_state(I, red, T),
    no_turn_on_red(I, right), about_to_enter(ego, I, T).
constraint_v_max(ego, T, 0) :-
    planned_turn(ego, right), light_state(I, red, T),
    about_to_enter(ego, I, T), not has_stopped(ego, I).
constraint_v_max(ego, T, 0) :-
    planned_turn(ego, right), light_state(I, red, T),
    about_to_enter(ego, I, T), not clear_path(ego, I, T).
% R7.2 left-across: hold until every crossed oncoming lane is clear
% for the full turn duration + margin.
constraint_v_max(ego, T, 0) :-
    planned_turn(ego, left), oncoming_lane(L, I), oncoming(V, ego, T),
    lane_of(V, L), time_to_conflict_point(V, T, TTC),
    turn_duration(ego, Est), left_turn_margin(M),
    TTC < Est + M, time(T).
% R7.3 U-turn / R7.4 late signal / R7.5 turn radius: NO constraint
% emitted (documented exemption). U-turns need a re-route the box
% schema cannot express; signal timing has no geometric bound; turn
% radius is a path-shape property for the QP smoothing objective (P4),
% not a per-timestep box.

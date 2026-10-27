% traffic_control.lp - Rules R5.1-R5.5
% Covers: red/yellow/green light, flashing signals, protected arrows.

% === R5.1 Red Light Compliance ===
% Mapped case: the approach lane is governed by the red-showing head.
violation(red_light, ego, T) :-
    pos(ego, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    D2 = (X-Xc)*(X-Xc) + (Y-Yc)*(Y-Yc), D2 <= R*R,
    light_state(I, red, T), controls_approach(I, L), in_lane(ego, T, L).
% Fallback case: no approach mapping for THIS intersection; any red
% approach whose box ego enters is a violation.
violation(red_light, ego, T) :-
    pos(ego, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    D2 = (X-Xc)*(X-Xc) + (Y-Yc)*(Y-Yc), D2 <= R*R,
    light_state(I, red, T), not controls_approach(I, _).

% === R5.2 Yellow Light Decision Zone ===
% Can stop (need <= dist) but plan does not decelerate -> violation.
% Banded stop needs from helpers (int32-safe); already committed
% (need > dist) may proceed and is left uncorrected.
violation(unsafe_yellow_response, ego, T) :-
    light_state(I, yellow, T), stopline_dist2(ego, I, T, D2),
    speed_band(ego, T, Band), stop_need(Band, SD), D2 > SD*SD,
    not decelerating(ego, T).

% === R5.3 Green Light With Obstacle in Intersection ===
violation(green_with_obstacle, ego, T) :-
    light_state(I, green, T), about_to_enter(ego, I, T),
    obstacle_in_box(V, I, T), not clear_path(ego, I, T).

% === R5.4 Flashing Signal Handling ===
% Flashing red: full stop required (stop-sign equivalent).
violation(flashing_red_no_stop, ego, T) :-
    light_state(I, flashing_red, T), at_stopline(ego, I, T),
    not has_stopped(ego, I).
% Flashing yellow: caution speed within 2R of the intersection.
% Default caution cap 670 cm/s (~15 mph); squared at comparison.
flashing_yellow_cap(670).
violation(flashing_yellow_speed, ego, T) :-
    light_state(I, flashing_yellow, T), pos(ego, T, X, Y),
    intersection(I, Xc, Yc, R), time(T),
    D2 = (X-Xc)*(X-Xc) + (Y-Yc)*(Y-Yc), D2 <= (R*2)*(R*2),
    velocity_sq(ego, T, S2), flashing_yellow_cap(Cap), S2 > Cap*Cap.

% === R5.5 Protected Arrow Signal Compliance ===
violation(unprotected_turn, ego, T) :-
    planned_turn(ego, Direction), about_to_enter(ego, I, T),
    light_state(I, arrow_red, Direction, T), not light_state(I, green, T).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% Holds are v_max 0 (stop and re-evaluate each timestep). Orientation-
% safe where a stopline box is not expressible as an allowed-region
% box; the hold point itself comes from the corrector (brake-to-line).
% R5.1 mapped + fallback bodies.
constraint_v_max(ego, T, 0) :-
    pos(ego, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    D2 = (X-Xc)*(X-Xc) + (Y-Yc)*(Y-Yc), D2 <= R*R,
    light_state(I, red, T), controls_approach(I, L), in_lane(ego, T, L).
constraint_v_max(ego, T, 0) :-
    pos(ego, T, X, Y), intersection(I, Xc, Yc, R), time(T),
    D2 = (X-Xc)*(X-Xc) + (Y-Yc)*(Y-Yc), D2 <= R*R,
    light_state(I, red, T), not controls_approach(I, _).
% R5.2: only when stopping is feasible (need <= dist); the committed
% case (need > dist) must NOT be held (R4.4 rear-end risk).
constraint_v_max(ego, T, 0) :-
    light_state(I, yellow, T), stopline_dist2(ego, I, T, D2),
    speed_band(ego, T, Band), stop_need(Band, SD), D2 > SD*SD,
    not decelerating(ego, T).
% R5.3 green-with-obstacle: hold despite green until the box clears.
constraint_v_max(ego, T, 0) :-
    light_state(I, green, T), about_to_enter(ego, I, T),
    obstacle_in_box(V, I, T), not clear_path(ego, I, T).
% R5.4 flashing red: full-stop hold. Flashing yellow: caution cap.
constraint_v_max(ego, T, 0) :-
    light_state(I, flashing_red, T), at_stopline(ego, I, T),
    not has_stopped(ego, I).
constraint_v_max(ego, T, Cap) :-
    light_state(I, flashing_yellow, T), pos(ego, T, X, Y),
    intersection(I, Xc, Yc, R), time(T),
    D2 = (X-Xc)*(X-Xc) + (Y-Yc)*(Y-Yc), D2 <= (R*2)*(R*2),
    flashing_yellow_cap(Cap).
% R5.5 unprotected turn: hold for the arrow (or yield-and-turn per R7.2
% once only the general green governs, which the corrector sequences).
constraint_v_max(ego, T, 0) :-
    planned_turn(ego, Direction), about_to_enter(ego, I, T),
    light_state(I, arrow_red, Direction, T), not light_state(I, green, T).

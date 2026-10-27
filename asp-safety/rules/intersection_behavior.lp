% intersection_behavior.lp - Rules R10.1-R10.4
% Covers: queue position, box blocking, protected left, right-turn yield.

% === R10.1 Queue Position at Red Light ===
% Ego must stop behind the last queued vehicle with MinQueueGap (600cm).
violation(queue_cut, ego, T) :-
    light_state(I, red, T), last_queued_vehicle(V, I, T),
    gap_dist(ego, V, T, D), D < 600, not behind(ego, V, I, T).

% === R10.2 Box Blocking Prevention ===
violation(box_blocking, ego, T) :-
    about_to_enter(ego, I, T), queue_ahead(V, I, T),
    space_beyond_box(ego, V, I, T, Space), ego_length(Len), Space < Len.

% === R10.3 Protected Left Turn Lane Compliance ===
violation(protected_lane_misuse, ego, T) :-
    in_lane(ego, T, L), lane_type(L, protected_left),
    planned_maneuver(ego, Type, T), Type != left_turn, Type != stopped.

% === R10.4 Right Turn Yielding to Pedestrians in the Cross Street ===
violation(right_turn_ped_yield, ego, T) :-
    planned_turn(ego, right), destination_crosswalk(I, right, C),
    pedestrian(P, T, X, Y), crosswalk(C, X1, Y1, X2, Y2), time(T),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    path_intersects(ego, C, T).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% Intersection holds are v_max 0 (re-target the stop point / wait out
% the downstream queue; the stop-point geometry is the corrector's job).
% R10.1 queue-cut (600 cm floor mirrors the violation body).
constraint_v_max(ego, T, 0) :-
    light_state(I, red, T), last_queued_vehicle(V, I, T),
    gap_dist(ego, V, T, D), D < 600, not behind(ego, V, I, T).
% R10.2 box blocking: hold at the near stop line.
constraint_v_max(ego, T, 0) :-
    about_to_enter(ego, I, T), queue_ahead(V, I, T),
    space_beyond_box(ego, V, I, T, Space), ego_length(Len), Space < Len.
% R10.3 protected-lane misuse: hold in-lane for the arrow phase.
constraint_v_max(ego, T, 0) :-
    in_lane(ego, T, L), lane_type(L, protected_left),
    planned_maneuver(ego, Type, T), Type != left_turn, Type != stopped.
% R10.4 destination-crosswalk yield (after the R7.1 vehicular gates).
constraint_v_max(ego, T, 0) :-
    planned_turn(ego, right), destination_crosswalk(I, right, C),
    pedestrian(P, T, X, Y), crosswalk(C, X1, Y1, X2, Y2), time(T),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    path_intersects(ego, C, T).

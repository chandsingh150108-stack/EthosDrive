% lane_discipline.lp - Rules R2.1-R2.5
% Covers: lane keeping, lane change permission, wrong-way, HOV/bus, departure warning.

% === R2.1 Lane Keeping (Boundary Containment) ===
% Fires only when lanes are mapped; ego-only per spec. A timestep
% BRIDGED between two different lanes (in L1 at T-1, in L2 /= L1 at
% T+1) is a lane-change transition, not a departure: flagging the
% midpoint would veto every legal lane change (R2.2 governs
% permission). Spec divergence, see DC-4 log.
lane_transition(V, T) :-
    in_lane(V, T2, L1), in_lane(V, T3, L2), T2 == T - 1, T3 == T + 1,
    L1 != L2, time(T), T > 0, T < 5.
violation(lane_keeping, ego, T) :-
    pos(ego, T, X, Y), time(T),
    lane(_, _, _, _, _), not in_lane(ego, T, _),
    not lane_transition(ego, T).

% === R2.2 Lane Change Permission & Minimum Gap ===
violation(illegal_lane_change, ego, T) :-
    changes_lane(ego, T, L1, L2), lane_boundary(L1, L2, solid).
violation(unsafe_lane_change_gap, ego, T) :-
    changes_lane(ego, T, L1, L2),
    dist2(ego, V, T2, D), V != ego, time(T2), T2 == T + 1,
    D < 250000.

% === R2.3 Wrong-Way Driving Detection ===
% lane_direction is scaled x100; sign of dot product is unaffected.
% Minimum-motion guard (calibration 2026-09-25): a stationary/creeping
% vehicle has noise headings that can point anywhere; wrong-way means
% sustained counter-flow driving, so require speed above 0.5 m/s
% (squared: 50^2 = 2500). Tunable; see calibration_baseline.json.
min_wrong_way_speed2(2500).
violation(wrong_way, ego, T) :-
    in_lane(ego, T, L), heading(ego, T, HX, HY),
    lane_direction(L, DX, DY), Dot = HX*DX + HY*DY, Dot < 0,
    speed2(ego, T, S2), min_wrong_way_speed2(MinS2), S2 > MinS2.

% === R2.4 HOV / Bus Lane Restriction ===
% Restriction facts carry (Type, MinOccupancy, Hours); Hours == always
% means continuously enforced, otherwise enforcement requires a
% matching current_time fact.
restriction_active(always).
restriction_active(Hours) :- current_time(Now), Hours == Now.
violation(restricted_lane, ego, T) :-
    in_lane(ego, T, L), lane_restriction(L, hov, MinOcc, Hours),
    occupancy(ego, N), N < MinOcc, restriction_active(Hours), time(T).
violation(restricted_lane, ego, T) :-
    in_lane(ego, T, L), lane_restriction(L, bus_only, _, Hours),
    restriction_active(Hours), time(T).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% Lane corridors keyed on membership (in_lane covers both checker-emitted
% in_lane_fact for array trajectories and the ASP geometric fallback for
% hand-written pos/4 extras). Pull-back: a timestep outside all lanes
% reuses the corridor of the lane occupied at T-1/T+1 (drift-out case).
% Fully off-map timesteps stay silent (honest map-gap silence, N-1).
% Horizontal lanes bound Y (+ lane_center preference); vertical lanes
% bound X. All products are W/2-scale, int32-safe.
constraint_y_bounds(ego, T, Y1-W/2, Y1+W/2) :-
    in_lane(ego, T, L), lane(L, X1, Y1, X2, Y2), Y1 == Y2,
    lane_width(L, W), time(T).
constraint_y_bounds(ego, T, Y1-W/2, Y1+W/2) :-
    in_lane(ego, T2, L), T2 == T - 1, lane(L, X1, Y1, X2, Y2),
    Y1 == Y2, lane_width(L, W), time(T), T > 0.
constraint_y_bounds(ego, T, Y1-W/2, Y1+W/2) :-
    in_lane(ego, T2, L), T2 == T + 1, lane(L, X1, Y1, X2, Y2),
    Y1 == Y2, lane_width(L, W), time(T), T < 5.
constraint_x_bounds(ego, T, X1-W/2, X1+W/2) :-
    in_lane(ego, T, L), lane(L, X1, Y1, X2, Y2), X1 == X2,
    lane_width(L, W), time(T).
constraint_x_bounds(ego, T, X1-W/2, X1+W/2) :-
    in_lane(ego, T2, L), T2 == T - 1, lane(L, X1, Y1, X2, Y2),
    X1 == X2, lane_width(L, W), time(T), T > 0.
constraint_x_bounds(ego, T, X1-W/2, X1+W/2) :-
    in_lane(ego, T2, L), T2 == T + 1, lane(L, X1, Y1, X2, Y2),
    X1 == X2, lane_width(L, W), time(T), T < 5.
constraint_lane_center(ego, T, Y1) :-
    in_lane(ego, T, L), lane(L, X1, Y1, X2, Y2), Y1 == Y2, time(T).

% R2.2 illegal_lane_change: hold the origin lane (stay, do not cross).
% Corridor + center preference for the lane ego is leaving.
constraint_y_bounds(ego, T, Y1-W/2, Y1+W/2) :-
    changes_lane(ego, T, L1, L2), lane_boundary(L1, L2, solid),
    lane(L1, X1, Y1, X2, Y2), Y1 == Y2, lane_width(L1, W), time(T).
constraint_x_bounds(ego, T, X1-W/2, X1+W/2) :-
    changes_lane(ego, T, L1, L2), lane_boundary(L1, L2, solid),
    lane(L1, X1, Y1, X2, Y2), X1 == X2, lane_width(L1, W), time(T).
constraint_lane_center(ego, T, Y1) :-
    changes_lane(ego, T, L1, L2), lane_boundary(L1, L2, solid),
    lane(L1, X1, Y1, X2, Y2), Y1 == Y2, time(T).
% R2.2 unsafe_gap: caution cap (5 m/s) to delay the crossing until the
% target-lane gap opens; the gap geometry itself is R3/R4's job.
constraint_v_max(ego, T, 500) :-
    changes_lane(ego, T, L1, L2),
    dist2(ego, V, T2, D), V != ego, T2 == T + 1, D < 250000.
% R2.3 wrong_way / R2.4 restricted_lane / R2.5 departure warning: NO
% constraint emitted (documented exemption). Wrong-way demands stop /
% takeover, not a corridor nudge; restricted lanes need a re-route the
% box schema cannot express (adjacent-lane target unknown in ASP);
% the departure warning is advisory-only.

% === R2.5 Lane Departure Early Warning (advisory) ===
% Outer ~20% band of the half-width: (0.8*W/2)^2 < D^2 <= (W/2)^2,
% i.e. D*D*400 > W*W*64 and D*D*4 <= W*W (axis-aligned only).
warning(lane_departure, ego, T) :-
    pos(ego, T, X, Y), lane(L, X1, Y1, X2, Y2), lane_width(L, W),
    Y1 == Y2, D = Y - Y1, D*D*400 > W*W*64, D*D*4 <= W*W, time(T).
warning(lane_departure, ego, T) :-
    pos(ego, T, X, Y), lane(L, X1, Y1, X2, Y2), lane_width(L, W),
    X1 == X2, D = X - X1, D*D*400 > W*W*64, D*D*4 <= W*W, time(T).

% emergency.lp - Rules R11.1-R11.4
% Covers: braking feasibility, evasive maneuver, guardrails, construction zones.

% Default ego deceleration capability, cm/timestep^2 (R11.1, R14.2).
max_decel(ego, 80).

% === R11.1 Emergency Braking Feasibility ===
% Banded stop needs (helpers) vs linear hazard distance: insufficient
% when the band need exceeds the remaining distance.
insufficient(emergency_brake, ego, H, T) :-
    hazard_distance(ego, H, T, D), speed_band(ego, T, Band),
    stop_need(Band, SD), D < SD.

% === R11.2 Evasive Maneuver Boundaries ===
% An evasive candidate is rejected when it leaves all mapped lanes
% (and lanes exist) or when it conflicts with another agent.
violation(unsafe_evasion, ego, T) :-
    insufficient(emergency_brake, ego, H, T), evasive_path(ego, T, X, Y),
    lane(_, _, _, _, _), not evasion_in_lane(X, Y).
violation(unsafe_evasion, ego, T) :-
    insufficient(emergency_brake, ego, H, T), evasive_path(ego, T, X, Y),
    pos(V, T, X2, Y2), V != ego,
    DX = X - X2, DY = Y - Y2, DX*DX + DY*DY < 22500.
evasion_in_lane(X, Y) :-
    evasive_path(ego, T, X, Y),
    lane(L, X1, Y1, X2, Y2), lane_width(L, W),
    Y1 == Y2, D = Y - Y1, D*D*4 <= W*W,
    (X-X1)*(X-X2) <= W*W.
evasion_in_lane(X, Y) :-
    evasive_path(ego, T, X, Y),
    lane(L, X1, Y1, X2, Y2), lane_width(L, W),
    X1 == X2, D = X - X1, D*D*4 <= W*W,
    (Y-Y1)*(Y-Y2) <= W*W.

% === R11.3 Guardrail / Fixed-Obstacle Avoidance ===
% Zero-tolerance footprint overlap (2* form avoids Width/2 division).
% Default footprint width 180cm.
ego_footprint(180).
violation(fixed_obstacle_contact, ego, T) :-
    pos(ego, T, X, Y), fixed_obstacle(O, X1, Y1, X2, Y2),
    ego_footprint(W), time(T),
    2*X >= 2*X1 - W, 2*X <= 2*X2 + W,
    2*Y >= 2*Y1 - W, 2*Y <= 2*Y2 + W.

% === R11.4 Active Construction Zone Compliance ===
% Default worker clearance radius 300cm (squared = 90000).
worker_clearance2(90000).
violation(construction_lane, ego, T) :-
    zone_active(Z, T), lane_shift(Z, L, NewPath),
    in_lane(ego, T, L), not following_path(ego, T, NewPath).
violation(construction_worker_clearance, ego, T) :-
    zone_active(Z, T), worker(W, T, X, Y), pos(ego, T, Xe, Ye), time(T),
    D2 = (X-Xe)*(X-Xe) + (Y-Ye)*(Y-Ye),
    worker_clearance2(C2), D2 < C2.

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% R11.1 brake-insufficient: maximum braking anyway minimizes impact
% severity (the spec fallback when no evasive path exists) -> v_max 0.
constraint_v_max(ego, T, 0) :-
    hazard_distance(ego, H, T, D), speed_band(ego, T, Band),
    stop_need(Band, SD), D < SD.
% R11.4 worker clearance: lateral band around the worker position
% (300 cm default radius; mirrors the R4 precedent pattern).
constraint_y_bounds(ego, T, Y-300, Y+300) :-
    zone_active(Z, T), worker(W, T, X, Y), pos(ego, T, Xe, Ye), time(T),
    D2 = (X-Xe)*(X-Xe) + (Y-Ye)*(Y-Ye),
    worker_clearance2(C2), D2 < C2.
% R11.2 unsafe evasion / R11.3 fixed obstacle / R11.4 lane-shift: NO
% constraint emitted (documented exemption). Evasion candidates are
% evaluated, not ego-plan bounds; fixed obstacles are zero-tolerance
% exclusions (any candidate touching them is discarded outright, no
% box can express that); construction lane-shifts reference an opaque
% NewPath the box schema cannot name.

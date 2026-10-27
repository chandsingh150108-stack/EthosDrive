% collision_avoidance.lp - Rules R4.1-R4.5
% Covers: lateral/longitudinal clearance, cut-in, rear-end risk, side-swipe.
% Lateral math uses decimeter units (/10) to stay within 32-bit integers.

% Default minimum lateral clearance, cm (R4.1).
min_lateral_clearance(100).
% Default minimum longitudinal gap, cm (R4.2).
min_longitudinal_gap(300).
% Default bare-trajectory collision threshold, cm^2 (5m).
default_collision_dist2(250000).

% Longitudinal overlap along ego heading (decimeter units, R4.1).
longitudinal_overlap(V1, V2, T) :-
    pos(V1, T, X1, Y1), pos(V1, T2, Xh, Yh), T2 == T + 1,
    pos(V2, T, X2, Y2),
    HX = (Xh-X1)/10, HY = (Yh-Y1)/10,
    HX*HX + HY*HY > 0,
    P1 = HX*((X1-X2)/10) + HY*((Y1-Y2)/10),
    P2 = HX*((Xh-X2)/10) + HY*((Yh-Y2)/10),
    P1*P2 <= 0, time(T).

% === R4.1 Minimum Lateral Clearance ===
% |DY| < MinL  <=>  DY^2 < MinL^2 * H2 (normalized by heading length).
violation(lateral_clearance, ego, V, T) :-
    pos(ego, T, X, Y), pos(V, T, X2, Y2), V != ego,
    pos(ego, T2, Xh, Yh), T2 == T + 1, time(T),
    min_lateral_clearance(MinL),
    HX = (Xh-X)/10, HY = (Yh-Y)/10, H2 = HX*HX + HY*HY, H2 > 0,
    DY = HY*((X2-X)/10) - HX*((Y2-Y)/10),
    ML = MinL/10, DY*DY < ML*ML*H2,
    longitudinal_overlap(ego, V, T).

% Constraint: lateral bounds to maintain clearance from vehicle V
constraint_y_bounds(ego, T, Y2 - MinL, Y2 + MinL) :-
    pos(ego, T, X, Y), pos(V, T, X2, Y2), V != ego,
    pos(ego, T2, Xh, Yh), T2 == T + 1, time(T),
    min_lateral_clearance(MinL),
    HX = (Xh-X)/10, HY = (Yh-Y)/10, H2 = HX*HX + HY*HY, H2 > 0,
    DY = HY*((X2-X)/10) - HX*((Y2-Y)/10),
    ML = MinL/10, DY*DY < ML*ML*H2,
    longitudinal_overlap(ego, V, T).

% Range constraint: lateral corridor for moving threat across consecutive timesteps
% Emits a corridor [T0, T1] with y-bounds covering the threat's full range
constraint_y_bounds_range(ego, T0, T1, YMin - MinL, YMax + MinL) :-
    vehicle(V), V != ego,
    min_lateral_clearance(MinL),
    % Find the first and last timestep where V longitudinally overlaps ego
    T0 = #min { T : longitudinal_overlap(ego, V, T), time(T) },
    T1 = #max { T : longitudinal_overlap(ego, V, T), time(T) },
    T0 < T1,  % at least 2 timesteps of overlap
    % Compute the threat's y-range across the overlap window
    YMin = #min { Y2 : pos(V, T, X2, Y2), longitudinal_overlap(ego, V, T),
                 time(T), T >= T0, T <= T1 },
    YMax = #max { Y2 : pos(V, T, X2, Y2), longitudinal_overlap(ego, V, T),
                 time(T), T >= T0, T <= T1 }.

% === R4.2 Minimum Longitudinal Clearance (same-lane hard floor) ===
% Along-track separation (P-4): an adjacent-lane vehicle within 3 m
% Euclidean is R4.1's job, not this rule's.
violation(longitudinal_clearance, ego, V, T) :-
    in_lane(ego, T, L), in_lane(V, T, L), V != ego,
    min_longitudinal_gap(MinG), longitudinal_gap(ego, V, T, D),
    D*D < MinG*MinG, time(T).

% Constraint: one-sided hold-back bounds (P6 sweep finding, S17:
% a box centered on the threat is an attractor for a closest-point
% optimizer, not a repeller -- QP moved ego TOWARD the lead and no
% pure-QP fix was possible). Signed longitudinal_gap gives the side
% (conventions: + = threat ahead): cap ego short of an ahead threat,
% floor ego ahead of a behind threat. Bounds use +-1000000 cm
% (+-10 km, int32-safe) for the open side.
constraint_x_bounds(ego, T, -1000000, X2-MinG) :-
    in_lane(ego, T, L), in_lane(V, T, L), V != ego,
    pos(ego, T, X, Y), pos(V, T, X2, Y2),
    min_longitudinal_gap(MinG), longitudinal_gap(ego, V, T, D), D > 0,
    D*D < MinG*MinG, time(T).
constraint_x_bounds(ego, T, X2+MinG, 1000000) :-
    in_lane(ego, T, L), in_lane(V, T, L), V != ego,
    pos(ego, T, X, Y), pos(V, T, X2, Y2),
    min_longitudinal_gap(MinG), longitudinal_gap(ego, V, T, D), D < 0,
    D*D < MinG*MinG, time(T).

% === R4.3 Cut-In Detection (hazard: externally caused) ===
hazard(cut_in, V, T) :-
    in_lane(V, T, L), in_lane(V, T2, L2), T2 == T - 1, L != L2,
    in_lane(ego, T, L), time(T), T > 0,
    dist2(ego, V, T, D), D < 640000.

% Constraint: speed cap to open gap for cut-in vehicle
constraint_v_max(ego, T, 500) :-
    in_lane(V, T, L), in_lane(V, T2, L2), T2 == T - 1, L != L2,
    in_lane(ego, T, L), time(T), T > 0,
    dist2(ego, V, T, D), D < 640000.

% === R4.4 Rear-End Collision Risk (hazard: ego braking endangers follower) ===
% Hard ego deceleration (accel-vector magnitude^2 > 300^2) while a
% follower is closer than its own speed warrants (D^2 < SV2).
hazard(rear_end_risk, ego, T) :-
    velocity(ego, T, VX1, VY1), velocity(ego, T2, VX2, VY2), T2 == T + 1,
    AX = VX2 - VX1, AY = VY2 - VY1, AX*AX + AY*AY > 90000,
    following_gap(V, ego, T, D), speed2(V, T, SV2), D*D < SV2, time(T).

% Constraint: accel cap to avoid endangering follower
constraint_a_max(ego, T, 300) :-
    velocity(ego, T, VX1, VY1), velocity(ego, T2, VX2, VY2), T2 == T + 1,
    AX = VX2 - VX1, AY = VY2 - VY1, AX*AX + AY*AY > 90000,
    following_gap(V, ego, T, D), speed2(V, T, SV2), D*D < SV2, time(T).

% === R4.5 Side-Swipe Risk (converging trajectories, decimeter units) ===
% 150cm convergence threshold: 150/10 = 15, squared = 225.
hazard(sideswipe_risk, ego, V, T) :-
    pos(ego, T, X0, Y0), pos(ego, T2, X1, Y1), T2 == T + 1,
    predicted_pos(V, T, X2, Y2), predicted_pos(V, T2, X3, Y3), time(T),
    S0 = ((X0-X2)/10)*((X0-X2)/10) + ((Y0-Y2)/10)*((Y0-Y2)/10),
    S1 = ((X1-X3)/10)*((X1-X3)/10) + ((Y1-Y3)/10)*((Y1-Y3)/10),
    S1 < S0, S1 < 225.

% Constraint: lateral bounds to avoid converging trajectory
constraint_y_bounds(ego, T, Y2 - 150, Y2 + 150) :-
    pos(ego, T, X0, Y0), pos(ego, T2, X1, Y1), T2 == T + 1,
    predicted_pos(V, T, X2, Y2), predicted_pos(V, T2, X3, Y3), time(T),
    S0 = ((X0-X2)/10)*((X0-X2)/10) + ((Y0-Y2)/10)*((Y0-Y2)/10),
    S1 = ((X1-X3)/10)*((X1-X3)/10) + ((Y1-Y3)/10)*((Y1-Y3)/10),
    S1 < S0, S1 < 225.

% === Default bare-trajectory collision rule (no map facts needed) ===
% Any tracked agent within 5m of ego at any timestep.
violation(collision_avoidance, ego, V, T) :-
    dist2(ego, V, T, D), time(T), D < 250000.

% Constraint: default 5m clearance from any tracked agent
constraint_x_bounds(ego, T, X2 - 500, X2 + 500) :-
    pos(ego, T, X, Y), pos(V, T, X2, Y2), V != ego,
    time(T), D = (X-X2)*(X-X2) + (Y-Y2)*(Y-Y2), D < 250000.
constraint_y_bounds(ego, T, Y2 - 500, Y2 + 500) :-
    pos(ego, T, X, Y), pos(V, T, X2, Y2), V != ego,
    time(T), D = (X-X2)*(X-X2) + (Y-Y2)*(Y-Y2), D < 250000.

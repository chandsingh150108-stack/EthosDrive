% comfort_quality.lp - Rules R12.1-R12.4
% Covers: accel/decel limits, lateral accel/jerk, smooth path, comfort scoring.
% All Low priority / advisory; override_active/2 (below) lets Critical
% rules legitimately exceed the comfort bounds.

% Default comfort bounds, cm/timestep^2.
max_comfort_accel(300).
max_comfort_decel(400).

% Critical rules assert override_active so hard safety braking never
% counts as a comfort violation.
override_active(ego, T) :- violation(red_light, ego, T).
override_active(ego, T) :- violation(lateral_clearance, ego, V, T).
override_active(ego, T) :- violation(longitudinal_clearance, ego, V, T).
override_active(ego, T) :- violation(predicted_collision, ego, V, T).
override_active(ego, T) :- violation(fixed_obstacle_contact, ego, T).
override_active(ego, T) :- violation(box_blocking, ego, T).
override_active(ego, T) :- violation(green_with_obstacle, ego, T).
override_active(ego, T) :- violation(collision_avoidance, ego, V, T).

% === R12.1 Maximum Acceleration/Deceleration Limits ===
% Accel-vector magnitude squared vs bound squared (int32-safe:
% steps <= ~20m/s keep A2 <= 3.2e7).
violation(excess_accel, ego, T) :-
    velocity(ego, T, VX1, VY1), velocity(ego, T2, VX2, VY2), T2 == T + 1,
    speed2(ego, T, S1), speed2(ego, T2, S2), S2 > S1,
    AX = VX2 - VX1, AY = VY2 - VY1, max_comfort_accel(A),
    AX*AX + AY*AY > A*A, not override_active(ego, T), time(T).
violation(excess_decel, ego, T) :-
    velocity(ego, T, VX1, VY1), velocity(ego, T2, VX2, VY2), T2 == T + 1,
    speed2(ego, T, S1), speed2(ego, T2, S2), S2 < S1,
    AX = VX2 - VX1, AY = VY2 - VY1, max_comfort_decel(D),
    AX*AX + AY*AY > D*D, not override_active(ego, T), time(T).

% === R12.2 Lateral Acceleration / Jerk Limits ===
% Meter-unit heading kink scaled by speed band: fast/med travel with a
% kink ratio above 25 (~30 deg) flags lateral accel; the frame-to-frame
% change of that ratio above 15 flags jerk. Defaults keep straight and
% gentle-lane-change trajectories clean.
violation(lateral_accel, ego, T) :-
    heading(ego, T2, DX0, DY0), T2 == T - 1,
    heading(ego, T, DX1, DY1), time(T), T > 0, T < 5,
    speed_band(ego, T, Band), Band != slow,
    MX0 = DX0/100, MY0 = DY0/100, MX1 = DX1/100, MY1 = DY1/100,
    C = MX0*MY1 - MY0*MX1,
    S0 = MX0*MX0 + MY0*MY0, S1 = MX1*MX1 + MY1*MY1,
    S0 > 0, S1 > 0, C*C*100 > S0*S1*25.
violation(lateral_jerk, ego, T) :-
    heading(ego, T2, DX0, DY0), T2 == T - 1,
    heading(ego, T, DX1, DY1), heading(ego, T3, DX2, DY2), T3 == T + 1,
    time(T), time(T3), T > 0, T < 4,
    MX0 = DX0/100, MY0 = DY0/100, MX1 = DX1/100, MY1 = DY1/100,
    MX2 = DX2/100, MY2 = DY2/100,
    C1 = MX0*MY1 - MY0*MX1, C2 = MX1*MY2 - MY1*MX2,
    DK = C2 - C1, DK*DK > 22500.

% === R12.3 Smooth Trajectory Requirement ===
% Fires on true heading reversals (>90 deg between consecutive headings,
% i.e. negative dot product): the signature of composed corrections
% forming a zig-zag rather than a smooth curve.
violation(discontinuous_path, ego, T) :-
    heading(ego, T2, DX0, DY0), T2 == T - 1,
    heading(ego, T, DX1, DY1), time(T), T > 0, T < 5,
    MX0 = DX0/100, MY0 = DY0/100, MX1 = DX1/100, MY1 = DY1/100,
    S0 = MX0*MX0 + MY0*MY0, S1 = MX1*MX1 + MY1*MY1,
    S0 > 0, S1 > 0, MX0*MX1 + MY0*MY1 < 0.

% === R12.4 Passenger Comfort Scoring (advisory, no violation) ===
% Tie-break signal only; counts comfort violations across the horizon.
comfort_score(ego, N) :- N = #count{ T : violation(excess_accel, ego, T) }.

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% R12.1: comfort accel/decel caps as QP bounds. Emitted whenever the
% bound context exists WITHOUT an active override (mirrors the
% violation bodies minus the magnitude comparison): overridden
% timesteps must not constrain the emergency maneuver.
constraint_a_max(ego, T, A) :-
    velocity(ego, T, VX1, VY1), velocity(ego, T2, VX2, VY2), T2 == T + 1,
    speed2(ego, T, S1), speed2(ego, T2, S2), S2 > S1,
    max_comfort_accel(A), not override_active(ego, T), time(T).
constraint_a_max(ego, T, D) :-
    velocity(ego, T, VX1, VY1), velocity(ego, T2, VX2, VY2), T2 == T + 1,
    speed2(ego, T, S1), speed2(ego, T2, S2), S2 < S1,
    max_comfort_decel(D), not override_active(ego, T), time(T).
% R12.2 lateral/jerk / R12.3 smooth-path / R12.4 score: NO constraint
% emitted (documented exemption). Kink ratios and heading reversals
% are path-shape properties for the QP smoothing objective (P4), not
% per-timestep boxes; the score is a tie-break signal, not a bound.

% following_behavior.lp - Rules R3.1-R3.4
% Covers: time-based following, speed-dependent floor, tailgating, queue behavior.
% gap_dist/4 is provided by checker.py (Euclidean, cm). lead_vehicle/3 via intent.

% === R3.1 Safe Following Distance (Time-Based) ===
% D/S < 2s  <=>  D^2 < 4*S2 (squared, integer-only).
violation(following_time, ego, T) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    speed2(ego, T, S2), time(T), S2 > 0, D*D < 4*S2.

% === R3.2 Safe Following Distance (Speed-Dependent Floor) ===
% Base form applies when no weather-adjusted gap exists for (Band,T);
% the weather variant below takes over otherwise (IF-3, no double-fire).
violation(following_min_gap, ego, T) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    speed_band(ego, T, Band), follow_gap_min(Band, MinD),
    not gap_adjusted(Band, T), D < MinD.
violation(following_min_gap, ego, T) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    speed_band(ego, T, Band), effective_min_gap(Band, T, AdjGap),
    D < AdjGap.

% === R3.3 Tailgating Pattern Detection ===
% Arity-3 form anchored at each flagged timestep (R13.4-compatible).
violation(tailgating, ego, T) :-
    violation(following_time, ego, T),
    N = #count{ T2 : violation(following_time, ego, T2) }, N >= 4.

% === R3.4 Caravan / Queue Behavior ===
% Large squared-speed swing while the queued lead is slow. Sign-split
% absolute form: DS = S2-S1 lives in (cm/step)^2 units where squaring
% again overflows 32-bit ints (a 3->0.5 m/s swing gives DS = -87500,
% whose square 7.6e9 wraps negative and NEVER fires -- the old form
% only caught mild swings and missed severe ones). Defaults: swing
% threshold 90000, lead-slow ceiling 130^2 = 16900 (both tunable facts,
% now actually referenced instead of hardcoded).
queue_osc_delta2(90000).
queue_slow_ceiling2(16900).
violation(queue_oscillation, ego, T) :-
    lead_vehicle(ego, V, T), lead_vehicle(ego, V, T2), T2 == T + 1,
    queue_member(V),
    speed2(ego, T, S1), speed2(ego, T2, S2),
    queue_osc_delta2(DD), S2 - S1 > DD,
    speed2(V, T, SV), queue_slow_ceiling2(Ceil), SV < Ceil, time(T).
violation(queue_oscillation, ego, T) :-
    lead_vehicle(ego, V, T), lead_vehicle(ego, V, T2), T2 == T + 1,
    queue_member(V),
    speed2(ego, T, S1), speed2(ego, T2, S2),
    queue_osc_delta2(DD), S1 - S2 > DD,
    speed2(V, T, SV), queue_slow_ceiling2(Ceil), SV < Ceil, time(T).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% R3.1: exact headway cap. D/S < 2s <=> S <= D/2 (linear cm/step);
% emitting D/2 as v_max guarantees the squared comparison on re-check.
constraint_v_max(ego, T, D/2) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    speed2(ego, T, S2), time(T), S2 > 0.
% R3.2: one-sided following corridor (same P6/S17 attractor finding
% as R4.2: center-on-lead boxes pull a closest-point optimizer into
% the lead). Ego stays behind an ahead lead by the band floor, or
% ahead of a behind lead (overtaken case). Ego-frame +x = forward.
constraint_x_bounds(ego, T, -1000000, X2-MinD) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 >= X,
    speed_band(ego, T, Band), follow_gap_min(Band, MinD),
    not gap_adjusted(Band, T), time(T).
constraint_x_bounds(ego, T, X2+MinD, 1000000) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 < X,
    speed_band(ego, T, Band), follow_gap_min(Band, MinD),
    not gap_adjusted(Band, T), time(T).
constraint_x_bounds(ego, T, -1000000, X2-AdjGap) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 >= X,
    speed_band(ego, T, Band), effective_min_gap(Band, T, AdjGap),
    time(T).
constraint_x_bounds(ego, T, X2+AdjGap, 1000000) :-
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 < X,
    speed_band(ego, T, Band), effective_min_gap(Band, T, AdjGap),
    time(T).
% R3.3 tailgating (aggregate): strengthen with the same exact headway
% cap at each flagged timestep.
constraint_v_max(ego, T, D/2) :-
    violation(following_time, ego, T),
    lead_vehicle(ego, V, T), gap_dist(ego, V, T, D), time(T).
% R3.4 queue oscillation: damp swings with the default comfort accel
% cap (300 cm/step^2); the rolling-average re-target itself is the
% corrector's job (QP smoothing objective, P4).
constraint_a_max(ego, T, 300) :-
    lead_vehicle(ego, V, T), lead_vehicle(ego, V, T2), T2 == T + 1,
    queue_member(V),
    speed2(V, T, SV), queue_slow_ceiling2(Ceil), SV < Ceil, time(T).

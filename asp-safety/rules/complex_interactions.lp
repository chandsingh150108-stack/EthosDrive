% complex_interactions.lp - Rules R15.1-R15.4
% Covers: multi-vehicle coordination, platoon, merge negotiation, waves.

% === R15.1 Multi-Vehicle Coordination at Conflict Points ===
% Joint priority by arrival order (ties share rank; ties yield).
priority_count(V, C, N) :-
    arrival_time(V, C, T),
    N = #count{ V2, T2 : arrival_time(V2, C, T2), T2 < T }.
priority_rank(V, C, Rank) :-
    priority_count(V, C, N), Rank = N + 1.
violation(multiway_priority, ego, T) :-
    conflict_point(C, X, Y), arrival_time(V, C, Tv),
    arrival_time(ego, C, Te), V != ego,
    Tv < Te, about_to_enter(ego, C, T), not has_cleared(V, C, T).
violation(multiway_priority, ego, T) :-
    conflict_point(C, X, Y), arrival_time(V, C, Te),
    arrival_time(ego, C, Te), V != ego,
    about_to_enter(ego, C, T), not has_cleared(V, C, T).

% === R15.2 Platoon/Convoy Behavior ===
% Tight platoon gap when comms verified; standard floor on comm loss.
% Defaults: platoon 300cm, standard 600cm.
platoon_min_gap(300).
standard_min_gap(600).
violation(platoon_gap, ego, T) :-
    platoon_member(ego, PlatoonId), platoon_member(V, PlatoonId),
    lead_vehicle(ego, V, T), comm_link_active(ego, V, T),
    gap_dist(ego, V, T, D), platoon_min_gap(MinG), D < MinG.
violation(platoon_gap, ego, T) :-
    platoon_member(ego, PlatoonId), platoon_member(V, PlatoonId),
    lead_vehicle(ego, V, T), not comm_link_active(ego, V, T),
    gap_dist(ego, V, T, D), standard_min_gap(MinG), D < MinG.

% === R15.3 Merge Lane Negotiation ===
% Zipper-slot violator: assigned 'before' a mainline vehicle without gap.
% Default minimum merge gap 500cm.
min_merge_gap(500).
violation(merge_negotiation, ego, T) :-
    in_lane(ego, T, L), merge_lane(L, _),
    mainline_vehicle(V), merge_gap_sequence(ego, V, before),
    gap_dist(ego, V, T, D), min_merge_gap(MinG), D < MinG.

% === R15.4 Traffic Wave Dampening ===
% Ego deviation from the lead's rolling average must not exceed the
% lead's own deviation scaled by AmplificationFactor (default 100%).
% Absolute values via four sign-split rules: #abs() is not valid ASP
% in clingo 5.7, and squaring first ((S-Avg)^2 reaches 1e10+) overflows
% 32-bit integers. Each product stays below ~9e8 (|diff| <= 9e6 for
% speeds under 30 m/s, times 100 or AF=100), int32-safe.
% SV != Avg in every rule: with a perfectly steady lead there is no
% fluctuation to amplify, so any ego offset (e.g. steady following at
% a different speed, which R3 already governs) must not fire here.
amplification_factor(100).
violation(wave_amplification, ego, T) :-
    lead_vehicle(ego, V, T), rolling_avg_speed(V, T, Avg),
    speed2(ego, T, S), speed2(V, T, SV), SV != Avg,
    S >= Avg, SV >= Avg,
    (S-Avg)*100 > (SV-Avg)*AF, amplification_factor(AF), time(T).
violation(wave_amplification, ego, T) :-
    lead_vehicle(ego, V, T), rolling_avg_speed(V, T, Avg),
    speed2(ego, T, S), speed2(V, T, SV), SV != Avg,
    S >= Avg, SV < Avg,
    (S-Avg)*100 > (Avg-SV)*AF, amplification_factor(AF), time(T).
violation(wave_amplification, ego, T) :-
    lead_vehicle(ego, V, T), rolling_avg_speed(V, T, Avg),
    speed2(ego, T, S), speed2(V, T, SV), SV != Avg,
    S < Avg, SV >= Avg,
    (Avg-S)*100 > (SV-Avg)*AF, amplification_factor(AF), time(T).
violation(wave_amplification, ego, T) :-
    lead_vehicle(ego, V, T), rolling_avg_speed(V, T, Avg),
    speed2(ego, T, S), speed2(V, T, SV), SV != Avg,
    S < Avg, SV < Avg,
    (Avg-S)*100 > (Avg-SV)*AF, amplification_factor(AF), time(T).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% R15.1 multiway priority: hold until every lower-ranked arrival
% clears (both the strict-order and tie bodies).
constraint_v_max(ego, T, 0) :-
    conflict_point(C, X, Y), arrival_time(V, C, Tv),
    arrival_time(ego, C, Te), V != ego,
    Tv < Te, about_to_enter(ego, C, T), not has_cleared(V, C, T).
constraint_v_max(ego, T, 0) :-
    conflict_point(C, X, Y), arrival_time(V, C, Te),
    arrival_time(ego, C, Te), V != ego,
    about_to_enter(ego, C, T), not has_cleared(V, C, T).
% R15.2 platoon gaps: one-sided hold-back at whichever floor
% currently applies (tight on verified comms, standard on loss).
% Same P6/S17 attractor rationale as R4.2/R3.2: center-on-lead
% boxes pull the optimizer into the lead.
constraint_x_bounds(ego, T, -1000000, X2-MinG) :-
    platoon_member(ego, PlatoonId), platoon_member(V, PlatoonId),
    lead_vehicle(ego, V, T), comm_link_active(ego, V, T),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 >= X,
    gap_dist(ego, V, T, D), platoon_min_gap(MinG), D < MinG.
constraint_x_bounds(ego, T, X2+MinG, 1000000) :-
    platoon_member(ego, PlatoonId), platoon_member(V, PlatoonId),
    lead_vehicle(ego, V, T), comm_link_active(ego, V, T),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 < X,
    gap_dist(ego, V, T, D), platoon_min_gap(MinG), D < MinG.
constraint_x_bounds(ego, T, -1000000, X2-MinG) :-
    platoon_member(ego, PlatoonId), platoon_member(V, PlatoonId),
    lead_vehicle(ego, V, T), not comm_link_active(ego, V, T),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 >= X,
    gap_dist(ego, V, T, D), standard_min_gap(MinG), D < MinG.
constraint_x_bounds(ego, T, X2+MinG, 1000000) :-
    platoon_member(ego, PlatoonId), platoon_member(V, PlatoonId),
    lead_vehicle(ego, V, T), not comm_link_active(ego, V, T),
    pos(ego, T, X, Y), pos(V, T, X2, Y2), X2 < X,
    gap_dist(ego, V, T, D), standard_min_gap(MinG), D < MinG.
% R15.3 merge slot: caution cap (5 m/s) to yield the slot and re-target
% the next gap down the merge queue (R2.2-style delay).
constraint_v_max(ego, T, 500) :-
    in_lane(ego, T, L), merge_lane(L, _),
    mainline_vehicle(V), merge_gap_sequence(ego, V, before),
    gap_dist(ego, V, T, D), min_merge_gap(MinG), D < MinG.
% R15.4 wave dampening: comfort accel cap whenever a lead rolling
% average exists (the re-target toward Avg itself is the QP smoothing
% objective, P4). Context-only emission like R12.1.
constraint_a_max(ego, T, 300) :-
    lead_vehicle(ego, V, T), rolling_avg_speed(V, T, Avg), time(T).

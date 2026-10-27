% predictive_safety.lp - Rules R13.1-R13.4
% Covers: trajectory collision prediction, TTC thresholds, threat
% assessment, multi-step aggregation.

% Default full-horizon collision threshold, cm^2 (5m).
collision_threshold2(250000).
% TTC band thresholds in deciseconds (100 = 1.0s).
ttc_imminent_units(100).
ttc_critical_units(200).
ttc_threatening_units(300).

% === R13.1 Trajectory Collision Prediction ===
% Full-horizon ego plan vs each agent's predicted path.
violation(predicted_collision, ego, V, T) :-
    pos(ego, T, X, Y), predicted_pos(V, T, X2, Y2), time(T),
    D2 = (X-X2)*(X-X2) + (Y-Y2)*(Y-Y2),
    collision_threshold2(Thr), D2 < Thr.

% === R13.2 Time-to-Collision (TTC) Thresholds ===
% Cross-multiplied integer bands (D/CR thresholds, no division).
% closing_rate/4 and gap_dist/4 are emitted by checker.py.
ttc_band(ego, V, T, imminent) :-
    closing_rate(ego, V, T, CR), CR > 0,
    gap_dist(ego, V, T, D), ttc_imminent_units(IU), D*100 <= CR*IU.
ttc_band(ego, V, T, critical) :-
    closing_rate(ego, V, T, CR), CR > 0,
    gap_dist(ego, V, T, D),
    ttc_imminent_units(IU), ttc_critical_units(CU),
    D*100 <= CR*CU, D*100 > CR*IU.
ttc_band(ego, V, T, threatening) :-
    closing_rate(ego, V, T, CR), CR > 0,
    gap_dist(ego, V, T, D),
    ttc_critical_units(CU), ttc_threatening_units(TU),
    D*100 <= CR*TU, D*100 > CR*CU.

% === R13.3 Threat Level Assessment (arbitration point) ===
threat_level(ego, T, imminent) :- ttc_band(ego, _, T, imminent).
threat_level(ego, T, imminent) :- insufficient(emergency_brake, ego, _, T).
threat_level(ego, T, critical) :-
    ttc_band(ego, _, T, critical), not threat_level(ego, T, imminent).
threat_level(ego, T, threatening) :-
    ttc_band(ego, _, T, threatening),
    not threat_level(ego, T, critical), not threat_level(ego, T, imminent).

% === R13.4 Multi-Step Violation Prediction (horizon aggregation) ===
any_violation(ego) :- violation(_, ego, _).
any_violation(ego) :- violation(_, ego, _, _).
future_violation(ego, T) :- violation(Rule, ego, T), T > 0.
future_violation(ego, T) :- violation(Rule, ego, _, T), T > 0.

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% R13.1 predicted collision: clearance box around the agent's
% predicted position (500 cm default threshold; mirrors the R4
% default-clearance precedent pattern).
constraint_x_bounds(ego, T, X2-500, X2+500) :-
    pos(ego, T, X, Y), predicted_pos(V, T, X2, Y2), time(T),
    D2 = (X-X2)*(X-X2) + (Y-Y2)*(Y-Y2),
    collision_threshold2(Thr), D2 < Thr.
constraint_y_bounds(ego, T, Y2-500, Y2+500) :-
    pos(ego, T, X, Y), predicted_pos(V, T, X2, Y2), time(T),
    D2 = (X-X2)*(X-X2) + (Y-Y2)*(Y-Y2),
    collision_threshold2(Thr), D2 < Thr.
% R13.2 TTC bands / R13.3 threat arbitration / R13.4 aggregation: NO
% constraint emitted (documented exemption). Bands and threat levels
% are arbitration inputs that route to R4/R11 corrections; emitting
% a second bound from the same geometry would double-constrain the
% timestep. Aggregation atoms are loop drivers, not bounds.

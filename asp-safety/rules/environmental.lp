% environmental.lp - Rules R14.1-R14.4
% Covers: wet road, night driving, visibility, weather thresholds.
% road_condition/time_of_day/perception_confidence/weather_severity
% facts are emitted by checker.py from the weather dict.

% === R14.1 Wet Road Speed Reduction ===
% S > Limit*Factor/100  <=>  (S2/100)*10000 > (Limit^2/100)*Factor^2
% (reduced form keeps all products within 32-bit range).
violation(wet_speed, ego, T) :-
    road_condition(T, wet), in_lane(ego, T, L),
    speed_limit(L, Limit), velocity_sq(ego, T, S2),
    wet_speed_factor(Factor), time(T),
    A = S2/100, B = Limit*Limit/100, A*10000 > B*Factor*Factor.

% === R14.2 Night Driving Behavior (overdriving headlights) ===
% Banded stop need vs linear headlight range.
violation(overdriving_headlights, ego, T) :-
    time_of_day(T, night), headlight_range(ego, Range),
    speed_band(ego, T, Band), stop_need(Band, SD), Range < SD.

% === R14.3 Visibility-Limited Rules ===
% Caps are LINEAR (cm/timestep); squared at comparison.
violation(low_visibility_speed, ego, T) :-
    perception_confidence(T, Level), confidence_speed_cap(Level, Cap),
    velocity_sq(ego, T, S2), S2 > Cap*Cap.

% === R14.4 Weather-Dependent Threshold Adjustment ===
% Parameter layer consumed by R1.1/R1.2 (speed limits) and R3.2
% (following gaps) when weather_severity facts are present (integer
% percentage factors). Guards (speed_adjusted/gap_adjusted) let the
% base rules stand down exactly when an adjusted threshold exists,
% so each timestep fires at most one of {base, adjusted}.
effective_limit(L, T, AdjLimit) :-
    speed_limit(L, Limit), weather_severity(T, Level),
    threshold_adjustment(Level, speed, Factor), time(T),
    AdjLimit = Limit*Factor/100.
speed_adjusted(L, T) :- effective_limit(L, T, _).
% Band-indexed gap table (follow_gap_min from helpers) scaled by the
% gap factor. (Previous min_gap_table/2 form referenced a table that
% was never defined, so it could never fire.)
effective_min_gap(Band, T, AdjGap) :-
    follow_gap_min(Band, MinD), weather_severity(T, Level),
    threshold_adjustment(Level, gap, Factor), time(T),
    AdjGap = MinD*Factor/100.
gap_adjusted(Band, T) :- effective_min_gap(Band, T, _).

% === Constraint emission for ASP-guided correction (P1, 2026-10-03) ===
% R14.1 wet road: reduced effective cap (Limit*Factor/100, integer
% percentage) under the same wet context as the violation.
constraint_v_max(ego, T, Limit*Factor/100) :-
    road_condition(T, wet), in_lane(ego, T, L),
    speed_limit(L, Limit), velocity_sq(ego, T, S2),
    wet_speed_factor(Factor), time(T).
% R14.2 night headlights: conservative slow-band cap (5 m/s) when the
% band need exceeds the illuminated range. Band-exact caps need a
% sqrt the integer schema cannot express; the slow cap is always
% need-satisfying (500 cm need) and safe-direction.
constraint_v_max(ego, T, 500) :-
    time_of_day(T, night), headlight_range(ego, Range),
    speed_band(ego, T, Band), stop_need(Band, SD), Range < SD.
% R14.3 low visibility: perception-confidence cap.
constraint_v_max(ego, T, Cap) :-
    perception_confidence(T, Level), confidence_speed_cap(Level, Cap),
    velocity_sq(ego, T, S2), time(T).
% R14.4 threshold adjustment: NO constraint emitted
% (documented exemption): it is a parameter layer consumed by R1/R3, which emit
% their own adjusted bounds; a duplicate here would double-count.

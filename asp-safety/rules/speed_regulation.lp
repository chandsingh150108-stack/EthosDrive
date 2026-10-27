% speed_regulation.lp - Rules R1.1-R1.5
% Covers: urban/highway speed limits, variable zones, min speed, VRU proximity speed.
% Limits/caps are LINEAR (cm/timestep); squared at comparison time.

% === R1.1 Urban Speed Limit Compliance ===
% Base form applies when no weather-adjusted limit exists for (L,T);
% the weather variant below takes over otherwise (IF-3, no double-fire).
violation(urban_speed, ego, T) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, urban), speed_limit(L, Limit),
    not speed_adjusted(L, T),
    S2 > Limit*Limit.
violation(urban_speed, ego, T) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, urban),
    effective_limit(L, T, AdjLimit),
    S2 > AdjLimit*AdjLimit.

% Constraint: speed cap from urban speed limit
constraint_v_max(ego, T, Limit) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, urban), speed_limit(L, Limit),
    not speed_adjusted(L, T).
constraint_v_max(ego, T, AdjLimit) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, urban),
    effective_limit(L, T, AdjLimit).

% === R1.2 Highway Maximum Speed Limit ===
violation(hwy_speed, ego, T) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, highway), speed_limit(L, Limit),
    not speed_adjusted(L, T),
    S2 > Limit*Limit.
violation(hwy_speed, ego, T) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, highway),
    effective_limit(L, T, AdjLimit),
    S2 > AdjLimit*AdjLimit.

% Constraint: speed cap from highway speed limit
constraint_v_max(ego, T, Limit) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, highway), speed_limit(L, Limit),
    not speed_adjusted(L, T).
constraint_v_max(ego, T, AdjLimit) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, highway),
    effective_limit(L, T, AdjLimit).

% === R1.3 Variable Speed Zones (School / Construction) ===
% Order-agnostic box test: (X-X1)*(X-X2) <= 0 iff X1 <= X <= X2 or reverse.
violation(variable_zone_speed, ego, T) :-
    velocity_sq(ego, T, S2), pos(ego, T, X, Y), time(T),
    zone(Z, X1, Y1, X2, Y2, _),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    zone_active(Z, T), zone_speed_limit(Z, Limit),
    S2 > Limit*Limit.

% Constraint: speed cap from active variable speed zone
constraint_v_max(ego, T, Limit) :-
    velocity_sq(ego, T, S2), pos(ego, T, X, Y), time(T),
    zone(Z, X1, Y1, X2, Y2, _),
    (X-X1)*(X-X2) <= 0, (Y-Y1)*(Y-Y2) <= 0,
    zone_active(Z, T), zone_speed_limit(Z, Limit).

% === R1.4 Minimum Speed on Highways ===
violation(min_hwy_speed, ego, T) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, highway),
    min_speed_limit(L, MinLimit), S2 < MinLimit*MinLimit,
    not emergency_flag(ego, T).

% Constraint: speed floor from highway minimum (when safe to accelerate)
constraint_v_min(ego, T, MinLimit) :-
    velocity_sq(ego, T, S2), time(T),
    in_lane(ego, T, L), road_type(L, highway),
    min_speed_limit(L, MinLimit),
    not emergency_flag(ego, T).

% === R1.5 Reduced Speed Near Pedestrians/Cyclists ===
% Squared-distance bucket (250000 cm^2 = 5m radius); per-type caps below.
vru_speed_cap(pedestrian, close, 670).
vru_speed_cap(cyclist, close, 670).
violation(vru_proximity_speed, ego, T) :-
    velocity_sq(ego, T, S2), pos(ego, T, X, Y), time(T),
    vru(P, T, X2, Y2, Type),
    D2 = (X-X2)*(X-X2) + (Y-Y2)*(Y-Y2), D2 <= 250000,
    vru_speed_cap(Type, close, Cap), S2 > Cap*Cap.

% Constraint: VRU proximity speed cap (min across all nearby VRUs)
constraint_v_max(ego, T, Cap) :-
    velocity_sq(ego, T, S2), pos(ego, T, X, Y), time(T),
    vru(P, T, X2, Y2, Type),
    D2 = (X-X2)*(X-X2) + (Y-Y2)*(Y-Y2), D2 <= 250000,
    vru_speed_cap(Type, close, Cap).

% === Default speed cap (no map facts needed) ===
% 50 km/h = 1389 cm/s; squared threshold = 1389^2 = 1929321.
% Applies ONLY to bare trajectories (no mapped lanes at all): with a
% real map whose limits are unposted, inventing a 50 km/h cap produces
% false positives on arterials (N-1, scene 7); unknown classification
% stays silent instead. Placeholder maps carry posted 13.4/urban facts
% so their speeding scenarios still fire (as urban_speed).
violation(speed_limit, ego, T) :-
    velocity_sq(ego, T, S2), time(T), S2 > 1929321,
    not lane(_, _, _, _, _).

% Constraint: default 50 km/h cap for bare trajectories
constraint_v_max(ego, T, 1389) :-
    velocity_sq(ego, T, S2), time(T), S2 > 1929321,
    not lane(_, _, _, _, _).

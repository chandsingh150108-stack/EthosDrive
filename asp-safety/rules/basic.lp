% basic.lp - Master aggregator for the 63-rule ASP traffic rule catalog
% Includes helpers and all 15 rule categories (R1-R15)

% === Shared predicates ===
#include "helpers.lp".

% === Phase 1: Core Rules (24 rules) ===
#include "speed_regulation.lp".       % R1.1-R1.5: Speed limits, zones, VRU proximity
#include "lane_discipline.lp".        % R2.1-R2.5: Lane keeping, changes, wrong-way, HOV
#include "following_behavior.lp".     % R3.1-R3.4: Following distance, tailgating, queue
#include "collision_avoidance.lp".    % R4.1-R4.5: Clearance, cut-in, rear-end, side-swipe
#include "traffic_control.lp".        % R5.1-R5.5: Signals, flashing, arrows

% === Phase 2: Extended Rules (23 rules) ===
#include "right_of_way.lp".           % R6.1-R6.5: Priority, roundabout, yield, emergency
#include "turning_rules.lp".          % R7.1-R7.5: Right-on-red, left turn, U-turn, radius
#include "overtaking.lp".             % R8.1-R8.4: Passing conditions, zones, clearance
#include "vru_safety.lp".            % R9.1-R9.5: Crosswalks, school zones, bike lanes
#include "intersection_behavior.lp".  % R10.1-R10.4: Queue, box blocking, protected left

% === Phase 3: Advanced Rules (16 rules) ===
#include "emergency.lp".              % R11.1-R11.4: Braking, evasion, guardrails
#include "comfort_quality.lp".        % R12.1-R12.4: Accel/decel, jerk, smooth path
#include "predictive_safety.lp".      % R13.1-R13.4: Collision prediction, TTC, threats
#include "environmental.lp".          % R14.1-R14.4: Wet road, night, visibility, weather
#include "complex_interactions.lp".   % R15.1-R15.4: Multi-vehicle, platoon, merge, waves

% === Global aggregates ===
% Total violation count (single aggregate over both arities so
% total_violations(0) cannot coexist with a non-zero count).
total_violations(V) :-
    V = #count { T, R, Veh : violation(R, Veh, T);
                 T, R, V1, V2 : violation(R, V1, V2, T) }.

% Safe trajectory: no violations
safe(ego) :-
    total_violations(0).

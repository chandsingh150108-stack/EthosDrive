"""Central rule-to-correction-strategy mapping (IF-1).

Single source of truth mapping every ASP catalog finding name to the
geometric correction strategy that handles it. Consumed by
TrajectoryCorrector; checker.py must NOT duplicate this table
(CR-4: checker delegates all corrections to the corrector).

Group -> (strategy method name on TrajectoryCorrector, frozenset of rules).
"""

# Speed-cap findings: scale velocity down to the applicable limit.
SPEED_RULES = frozenset({
    'speed_limit', 'urban_speed', 'hwy_speed',
    'variable_zone_speed', 'vru_proximity_speed',
    'wet_speed', 'overdriving_headlights', 'low_visibility_speed',
    'flashing_yellow_speed',
})

# Lateral-conflict findings: shift away in y.
COLLISION_RULES = frozenset({
    'collision_avoidance', 'lateral_clearance',
    'longitudinal_clearance', 'predicted_collision',
    'sideswipe_risk', 'cut_in', 'overtake_clearance',
    'platoon_gap', 'merge_negotiation',
})

# Lane-membership findings: project to lane center.
LANE_RULES = frozenset({
    'lane_keeping', 'restricted_lane',
    'bike_lane_encroachment', 'sidewalk_encroachment',
})

# Signal / yield / right-of-way / emergency / maneuver-suppression
# findings: stop and hold.
STOP_RULES = frozenset({
    'traffic_light', 'red_light', 'unsafe_yellow_response',
    'flashing_red_no_stop', 'unprotected_turn',
    'rtor_no_sign', 'rtor_no_stop', 'rtor_unclear',
    'green_with_obstacle', 'box_blocking',
    'uncontrolled_priority', 'roundabout_entry', 'turn_yield',
    'left_turn_across', 'emergency_yield', 'crosswalk_yield',
    'implied_crosswalk_yield', 'right_turn_ped_yield',
    'multiway_priority', 'queue_cut', 'protected_lane_misuse',
    'wrong_way', 'emergency_brake', 'fixed_obstacle_contact',
    'unsafe_evasion', 'construction_lane',
    'construction_worker_clearance', 'school_guard_stop',
    'u_turn_prohibited', 'u_turn_sightline',
    'unsafe_overtake', 'no_passing_zone', 'no_passing_sightline',
    'right_overtake', 'blind_spot_maneuver',
})

# Gap findings: gentle slowdown.
FOLLOW_RULES = frozenset({
    'safe_following', 'following_time', 'following_min_gap',
    'tailgating', 'queue_oscillation', 'wave_amplification',
    'rear_end_risk',
})

# Comfort/quality findings: smoothing only.
COMFORT_RULES = frozenset({
    'excess_accel', 'excess_decel', 'lateral_accel',
    'lateral_jerk', 'discontinuous_path', 'turn_radius',
})

# Group dispatch table: group -> corrector method name.
# Order matters only for readability; groups are disjoint.
RULE_GROUPS = {
    'speed': ('_correct_speed_limit', SPEED_RULES),
    'collision': ('_correct_collision', COLLISION_RULES),
    'lane': ('_correct_lane_keeping', LANE_RULES),
    'stop': ('_hold_position', STOP_RULES),
    'follow': ('_correct_safe_following', FOLLOW_RULES),
    'comfort': ('_smooth_point', COMFORT_RULES),
    'min_speed': ('_accelerate_point', frozenset({'min_hwy_speed'})),
}

# Advisory findings: reported, never corrected (fixpoint ignores them).
ADVISORY_RULES = frozenset({'late_signal'})

# Tunable numeric parameters (RQ-1). Values here are the built-in
# defaults; configs/correction_strategies.yaml may override them.
DEFAULT_PARAMS = {
    # Fallback speed target (m/s) when no map/zone limit is known.
    'default_speed_target_ms': 12.5,
    # Fallback following slowdown factor.
    'follow_scale': 0.7,
    # Minimum-speed boost factor (R1.4).
    'min_speed_boost': 1.1,
    # Lateral conflict shift (m).
    'lateral_shift_m': 2.0,
    # Comfort smoothing blend toward neighbor midpoint.
    'smooth_blend': 0.25,
    # Edge smoothing damp factor.
    'smooth_edge_dampen': 0.9,
    # Fallback lane-center y (m) when no map lane matches.
    'fallback_lane_center_y': 0.0,
    # Per-group application cap per correct() call (P-6): bounds
    # single-strategy churn so the fixpoint tries other strategies
    # instead of spinning on one.
    'group_cap': 40,
}


def strategy_for(rule: str):
    """Return (group, method_name) for a finding name.

    Returns (None, None) for advisory rules and ('unknown', None)
    for names outside the catalog (caller applies conservative hold).
    """
    if rule in ADVISORY_RULES:
        return None, None
    for group, (method, rules) in RULE_GROUPS.items():
        if rule in rules:
            return group, method
    return 'unknown', None


def all_known_rules():
    """Every finding name this table knows how to route."""
    known = set(ADVISORY_RULES)
    for _, rules in RULE_GROUPS.values():
        known |= set(rules)
    return known

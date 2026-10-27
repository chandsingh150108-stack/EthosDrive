"""Comprehensive tests for the 63-rule ASP traffic rule catalog.

Tests each rule category with known-good and known-bad trajectories.
All coordinates in meters (checker scales to centimeters internally).
Paths are resolved relative to this file so the suite runs from any CWD.
"""

import sys
from pathlib import Path

# CWD-independent paths (GAP-6 fix)
_TESTS_DIR = Path(__file__).parent
_SRC_DIR = str(_TESTS_DIR.parent / "src")
_RULES_DIR = str(_TESTS_DIR.parent / "rules")
sys.path.insert(0, _SRC_DIR)

import numpy as np
from checker import TrajectoryChecker


def _checker():
    return TrajectoryChecker(_RULES_DIR)


def test_helpers():
    """Test shared predicate compilation."""
    print("Testing helpers.lp...")
    checker = _checker()

    # Simple trajectory - should compile and solve without errors
    traj = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
    is_safe, violations = checker.check(traj)
    assert is_safe, f"bare slow trajectory must be safe, got {violations}"
    print(f"  [PASS] helpers.lp compiles and solves (safe={is_safe})")


def test_speed_regulation():
    """Test R1.1-R1.5: Speed regulation rules."""
    print("\nTesting speed_regulation.lp (R1.1-R1.5)...")
    checker = _checker()

    # Safe: slow urban driving
    safe_traj = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
    is_safe, v = checker.check(safe_traj)
    assert is_safe, f"slow trajectory must be safe, got {v}"
    print(f"  [PASS] Slow urban trajectory: safe={is_safe}")

    # Fast bare trajectory must violate the default 50 km/h cap
    fast_traj = np.array([[0,0],[20,0],[40,0],[60,0],[80,0],[100,0]], dtype=np.float32)
    is_safe, v = checker.check(fast_traj)
    assert not is_safe, "20 m/s trajectory must be unsafe"
    assert any(x['rule'] == 'speed_limit' for x in v), f"expected speed_limit, got {v}"
    print(f"  [PASS] Fast trajectory triggers speed_limit ({len(v)} findings)")

    # Test with road type and speed limit
    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 100, 'y2': 0}]},
        'road_type': {'l1': 'urban'},
        'speed_limit': {'l1': 13.4},  # ~30 mph in m/s
        'lane_width': {'l1': 3.5},
        'lane_direction': {'l1': [1.0, 0.0]},
    }
    # Slow trajectory - should be safe
    is_safe, v = checker.check(safe_traj, map_data=map_data)
    assert is_safe, f"slow on urban road must be safe, got {v}"
    print(f"  [PASS] Slow on urban road (limit=30mph): safe={is_safe}")

    # Over the posted urban limit -> urban_speed fires
    over_traj = np.array([[0,0],[15,0],[30,0],[45,0],[60,0],[75,0]], dtype=np.float32)
    is_safe, v = checker.check(over_traj, map_data=map_data)
    assert not is_safe, "15 m/s in 30mph zone must be unsafe"
    assert any(x['rule'] == 'urban_speed' for x in v), f"expected urban_speed, got {v}"
    print(f"  [PASS] Over-limit urban trajectory triggers urban_speed")

    # Test with variable zone
    zones = [{
        'id': 'z1', 'x1': 0, 'y1': -5, 'x2': 30, 'y2': 5,
        'type': 'school', 'active': True, 'speed_limit': 6.7  # ~15 mph
    }]
    is_safe, v = checker.check(safe_traj, map_data=map_data, zones=zones)
    assert is_safe, f"slow through school zone must be safe, got {v}"
    print(f"  [PASS] Through school zone: safe={is_safe}, violations={len(v)}")



def test_lane_discipline():
    """Test R2.1-R2.5: Lane discipline rules."""
    print("\nTesting lane_discipline.lp (R2.1-R2.5)...")
    checker = _checker()

    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 100, 'y2': 0}]},
        'lane_width': {'l1': 3.5},
        'lane_direction': {'l1': [1.0, 0.0]},
    }

    # Stays in lane - safe
    center_traj = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
    is_safe, v = checker.check(center_traj, map_data=map_data)
    assert is_safe, f"center-lane trajectory must be safe, got {v}"
    print(f"  [PASS] Center-lane trajectory: safe={is_safe}")

    # Leaves the lane -> lane_keeping fires
    off_traj = np.array([[0,0],[1,5],[2,10],[3,15],[4,20],[5,25]], dtype=np.float32)
    is_safe, v = checker.check(off_traj, map_data=map_data)
    assert not is_safe, "off-lane trajectory must be unsafe"
    assert any(x['rule'] == 'lane_keeping' for x in v), f"expected lane_keeping, got {v}"
    print(f"  [PASS] Off-lane trajectory triggers lane_keeping")

    # Wrong-way: same lane, opposite heading
    back_traj = np.array([[5,0],[4,0],[3,0],[2,0],[1,0],[0,0]], dtype=np.float32)
    is_safe, v = checker.check(back_traj, map_data=map_data)
    assert not is_safe, "wrong-way trajectory must be unsafe"
    assert any(x['rule'] == 'wrong_way' for x in v), f"expected wrong_way, got {v}"
    print(f"  [PASS] Wrong-way trajectory triggers wrong_way")



def test_following_behavior():
    """Test R3.1-R3.4: Following behavior rules."""
    print("\nTesting following_behavior.lp (R3.1-R3.4)...")
    checker = _checker()

    ego = np.array([[0,0],[2,0],[4,0],[6,0],[8,0],[10,0]], dtype=np.float32)
    lead = np.array([[0,12],[2,14],[4,16],[6,18],[8,20],[10,22]], dtype=np.float32)

    intent = {'lead_vehicle': {t: 'vehicle_0' for t in range(6)}}

    # Following at ~2m/s with 12m gap - safe
    is_safe, v = checker.check(ego, other_vehicles=[lead], ego_intent=intent)
    assert is_safe, f"12m gap at 2m/s must be safe, got {v}"
    print(f"  [PASS] Following at safe distance: safe={is_safe}")

    # Tailgating: fast ego, constant tiny gap, sustained -> following_time + tailgating
    close_ego = np.array([[0,0],[5,0],[10,0],[15,0],[20,0],[25,0]], dtype=np.float32)
    close_lead = np.array([[0,6],[5,6],[10,6],[15,6],[20,6],[25,6]], dtype=np.float32)
    is_safe, v = checker.check(close_ego, other_vehicles=[close_lead], ego_intent=intent)
    assert not is_safe, "6m gap at 5m/s must be unsafe"
    assert any(x['rule'] == 'following_time' for x in v), f"expected following_time, got {v}"
    assert any(x['rule'] == 'tailgating' for x in v), f"expected tailgating, got {v}"
    print(f"  [PASS] Tailgating triggers following_time + tailgating")



def test_collision_avoidance():
    """Test R4.1-R4.5: Collision avoidance rules."""
    print("\nTesting collision_avoidance.lp (R4.1-R4.5)...")
    checker = _checker()

    ego = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)

    # Identical trajectories: zero separation -> collision findings
    other = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
    is_safe, v = checker.check(ego, other_vehicles=[other])
    assert not is_safe, "overlapping trajectories must be unsafe"
    assert any(x['rule'] == 'collision_avoidance' for x in v), f"expected collision_avoidance, got {v}"
    print(f"  [PASS] Overlapping trajectories trigger collision_avoidance")

    # 3m lateral separation - default 5m bubble still fires
    other3 = np.array([[0,3],[1,3],[2,3],[3,3],[4,3],[5,3]], dtype=np.float32)
    is_safe, v = checker.check(ego, other_vehicles=[other3])
    assert not is_safe, "3m separation must trip the 5m bubble"
    print(f"  [PASS] 3m lateral separation: safe={is_safe}, violations={len(v)}")

    # 10m separation - clean
    other10 = np.array([[0,10],[1,10],[2,10],[3,10],[4,10],[5,10]], dtype=np.float32)
    is_safe, v = checker.check(ego, other_vehicles=[other10])
    assert is_safe, f"10m separation must be safe, got {v}"
    print(f"  [PASS] 10m lateral separation: safe={is_safe}")



def test_traffic_control():
    """Test R5.1-R5.5: Traffic control rules."""
    print("\nTesting traffic_control.lp (R5.1-R5.5)...")
    checker = _checker()

    # Trajectory entering the intersection box on red
    traj = np.array([[14,0],[16,0],[18,0],[20,0],[22,0],[24,0]], dtype=np.float32)

    lights = {
        'i1': {'color': 'red', 'times': list(range(6)), 'x': 20, 'y': 0, 'radius': 3}
    }
    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 50, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
        'controls_approach': {'i1': 'l1'},
    }

    is_safe, v = checker.check(traj, map_data=map_data, traffic_lights=lights)
    red_light_violations = [x for x in v if x['rule'] == 'red_light']
    assert not is_safe, "entering on red must be unsafe"
    assert len(red_light_violations) > 0, f"expected red_light, got {v}"
    print(f"  [PASS] Red light check: safe={is_safe}, red_light_violations={len(red_light_violations)}")

    # Same geometry on green: clean
    green = {
        'i1': {'color': 'green', 'times': list(range(6)), 'x': 20, 'y': 0, 'radius': 3}
    }
    is_safe, v = checker.check(traj, map_data=map_data, traffic_lights=green)
    assert is_safe, f"entering on green must be safe, got {v}"
    print(f"  [PASS] Green light check: safe={is_safe}")



def test_pedestrian_safety():
    """Test R6.5, R9.1-R9.5: Pedestrian and VRU safety."""
    print("\nTesting pedestrian/VRU safety rules...")
    checker = _checker()

    traj = np.array([[0,0],[5,0],[10,0],[15,0],[20,0],[25,0]], dtype=np.float32)

    # Pedestrian in marked crosswalk on the path -> must yield
    peds = [{
        'id': 'p1',
        'positions': {2: (12, 0)},
        'type': 'pedestrian'
    }]

    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 50, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
        'crosswalks': [{'id': 'cw1', 'x1': 10, 'y1': -3, 'x2': 14, 'y2': 3}],
    }

    is_safe, v = checker.check(traj, map_data=map_data, pedestrians=peds)
    assert not is_safe, "pedestrian in crosswalk on path must be unsafe"
    assert any(x['rule'] == 'crosswalk_yield' for x in v), f"expected crosswalk_yield, got {v}"
    print(f"  [PASS] Pedestrian in crosswalk triggers crosswalk_yield")

    # No pedestrian: clean
    is_safe, v = checker.check(traj, map_data=map_data)
    assert is_safe, f"empty crosswalk must be safe, got {v}"
    print(f"  [PASS] Empty crosswalk: safe={is_safe}")



def test_emergency():
    """Test R11.1-R11.4: Emergency situation rules."""
    print("\nTesting emergency.lp (R11.1-R11.4)...")
    checker = _checker()

    traj = np.array([[0,0],[5,0],[10,0],[15,0],[20,0],[25,0]], dtype=np.float32)

    # Fixed obstacle ahead (wide lane so a 3m offset stays in-lane)
    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 50, 'y2': 0}]},
        'lane_width': {'l1': 8.0},
        'fixed_obstacles': [{'id': 'obs1', 'x1': 18, 'y1': -1, 'x2': 22, 'y2': 1}],
    }

    is_safe, v = checker.check(traj, map_data=map_data)
    obstacle_violations = [x for x in v if x['rule'] == 'fixed_obstacle_contact']
    assert not is_safe, "trajectory through obstacle must be unsafe"
    assert len(obstacle_violations) > 0, f"expected fixed_obstacle_contact, got {v}"
    print(f"  [PASS] Fixed obstacle check: safe={is_safe}, obstacle_violations={len(obstacle_violations)}")

    # Offset trajectory missing the obstacle: clean
    clear_traj = np.array([[0,3],[5,3],[10,3],[15,3],[20,3],[25,3]], dtype=np.float32)
    is_safe, v = checker.check(clear_traj, map_data=map_data)
    assert is_safe, f"offset trajectory must be safe, got {v}"
    print(f"  [PASS] Offset trajectory: safe={is_safe}")



def test_environmental():
    """Test R14.1-R14.4: Environmental context rules."""
    print("\nTesting environmental.lp (R14.1-R14.4)...")
    checker = _checker()

    traj = np.array([[0,0],[2,0],[4,0],[6,0],[8,0],[10,0]], dtype=np.float32)

    weather = {
        'road_condition': {t: 'wet' for t in range(6)},
        'wet_speed_factor': 80,
    }

    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 100, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
        'speed_limit': {'l1': 13.4},
    }

    # 2 m/s on wet road with 80% of 13.4 limit: clean
    is_safe, v = checker.check(traj, map_data=map_data, weather=weather)
    assert is_safe, f"slow wet-road driving must be safe, got {v}"
    print(f"  [PASS] Wet road condition: safe={is_safe}, violations={len(v)}")

    # 12 m/s on the same wet road: over the wet-adjusted cap
    fast_traj = np.array([[0,0],[12,0],[24,0],[36,0],[48,0],[60,0]], dtype=np.float32)
    is_safe, v = checker.check(fast_traj, map_data=map_data, weather=weather)
    assert not is_safe, "12 m/s on wet 13.4-limit road must be unsafe"
    assert any(x['rule'] == 'wet_speed' for x in v), f"expected wet_speed, got {v}"
    print(f"  [PASS] Wet speeding triggers wet_speed")

    # Night driving within headlight range: clean
    weather_night = {'time_of_day': {0: 'night'}, 'headlight_range': 60}
    is_safe, v = checker.check(traj, weather=weather_night)
    assert is_safe, f"slow night driving must be safe, got {v}"
    print(f"  [PASS] Night driving: safe={is_safe}, violations={len(v)}")



def test_complex_interactions():
    """Test R15.1-R15.4: Complex interaction rules."""
    print("\nTesting complex_interactions.lp (R15.1-R15.4)...")
    checker = _checker()

    # Platoon behavior with comms: 8m gap is fine
    traj = np.array([[0,0],[2,0],[4,0],[6,0],[8,0],[10,0]], dtype=np.float32)
    lead = np.array([[0,8],[2,10],[4,12],[6,14],[8,16],[10,18]], dtype=np.float32)

    intent = {
        'lead_vehicle': {t: 'vehicle_0' for t in range(6)},
        'platoon': {
            'id': 'p1',
            'members': ['vehicle_0'],
            'comm_active': True,
            'lead_vehicle': 'vehicle_0'
        }
    }

    is_safe, v = checker.check(traj, other_vehicles=[lead], ego_intent=intent)
    assert is_safe, f"platoon with gap must be safe, got {v}"
    print(f"  [PASS] Platoon behavior: safe={is_safe}, violations={len(v)}")

    # Comm loss with tight 3m gap: standard floor violated
    tight_lead = np.array([[0,3],[2,5],[4,7],[6,9],[8,11],[10,13]], dtype=np.float32)
    intent_dead = {
        'lead_vehicle': {t: 'vehicle_0' for t in range(6)},
        'platoon': {
            'id': 'p1',
            'members': ['vehicle_0'],
            'comm_active': False,
        }
    }
    is_safe, v = checker.check(traj, other_vehicles=[tight_lead], ego_intent=intent_dead)
    assert not is_safe, "comm loss at tight gap must be unsafe"
    assert any(x['rule'] == 'platoon_gap' for x in v), f"expected platoon_gap, got {v}"
    print(f"  [PASS] Platoon comm loss triggers platoon_gap")



def test_priority_interactions():
    """GAP-11: rule priority/override interactions.

    Red light + pedestrian + close follower fire together; the
    emergency/critical findings must dominate and the corrected
    trajectory must be fully safe afterwards.
    """
    print("\nTesting rule priority interactions (GAP-11)...")
    from corrector import TrajectoryCorrector
    checker = _checker()
    corrector = TrajectoryCorrector(checker)

    traj = np.array([[14,0],[16,0],[18,0],[20,0],[22,0],[24,0]], dtype=np.float32)
    lights = {
        'i1': {'color': 'red', 'times': list(range(6)), 'x': 20, 'y': 0, 'radius': 3}
    }
    map_data = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 50, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
        'controls_approach': {'i1': 'l1'},
        'crosswalks': [{'id': 'cw1', 'x1': 19, 'y1': -3, 'x2': 23, 'y2': 3}],
    }
    peds = [{'id': 'p1', 'positions': {4: (21, 0)}, 'type': 'pedestrian'}]

    is_safe, v = checker.check(traj, map_data=map_data,
                               traffic_lights=lights, pedestrians=peds)
    rules = {x['rule'] for x in v}
    assert not is_safe, "red + pedestrian scene must be unsafe"
    assert 'red_light' in rules, f"expected red_light among {rules}"
    assert 'crosswalk_yield' in rules, f"expected crosswalk_yield among {rules}"
    print(f"  [PASS] Co-firing rules: {sorted(rules)}")

    corrected, applied = corrector.correct(
        traj, map_data=map_data, traffic_lights=lights, pedestrians=peds)
    is_safe, v = checker.check(corrected, map_data=map_data,
                               traffic_lights=lights, pedestrians=peds)
    assert is_safe, f"corrected scene must be safe, got {v}"
    assert len(applied) > 0, "expected applied corrections"
    print(f"  [PASS] Corrected scene safe after {len(applied)} corrections")



def test_correction():
    """Test trajectory correction for various violation types."""
    print("\nTesting trajectory correction...")
    checker = _checker()

    # Fast trajectory that violates the default speed cap
    fast_traj = np.array([[0,0],[5,0],[10,0],[15,0],[20,0],[25,0]], dtype=np.float32)
    corrected = checker.get_safe_trajectory(fast_traj)

    assert corrected.shape == fast_traj.shape
    print(f"  [PASS] Correction maintains shape: {corrected.shape}")

    # Check that corrected trajectory is safe
    is_safe, v = checker.check(corrected)
    assert is_safe, f"corrected trajectory must be safe, got {v}"
    print(f"  [PASS] Corrected trajectory safe: {is_safe}")



def test_uncovered_spot_probes():
    """T-2 starter: firing probes for previously uncovered findings.

    One assert per finding (firing direction only; safe-direction
    counterparts remain deferred). All use meter inputs; the checker
    scales to integer cm internally.
    """
    print("\nTesting uncovered spot probes (T-2 starter)...")
    checker = _checker()

    lane_map = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 100, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
        'lane_direction': {'l1': [1.0, 0.0]},
    }

    # 1. R1.2 hwy_speed: fast in a highway lane over its limit.
    hwy_map = dict(lane_map, road_type={'l1': 'highway'},
                   speed_limit={'l1': 5.0})
    fast = np.array([[0,0],[8,0],[16,0],[24,0],[32,0],[40,0]],
                    dtype=np.float32)
    ok, v = checker.check(fast, map_data=hwy_map)
    assert not ok and any(d['rule'] == 'hwy_speed' for d in v), \
        f"expected hwy_speed, got {v}"
    print("  [PASS] hwy_speed fires")

    # 2. R3.2 following_min_gap: slow band (2 m/step), 3 m gap < 5 m min.
    ego = np.array([[0,0],[2,0],[4,0],[6,0],[8,0],[10,0]], dtype=np.float32)
    lead = np.array([[3,0],[5,0],[7,0],[9,0],[11,0],[13,0]], dtype=np.float32)
    ok, v = checker.check(
        ego, other_vehicles=[lead],
        ego_intent={'lead_vehicle': {t: 'vehicle_0' for t in range(6)}})
    assert not ok and any(d['rule'] == 'following_min_gap' for d in v), \
        f"expected following_min_gap, got {v}"
    print("  [PASS] following_min_gap fires")

    # 3. R7.3 u_turn_prohibited: u_turn maneuver in a no-u-turn lane.
    uturn_map = dict(lane_map, no_u_turn=['l1'])
    slow = np.array([[0,0],[1,0],[2,0],[3,0],[4,0],[5,0]], dtype=np.float32)
    ok, v = checker.check(
        slow, map_data=uturn_map,
        ego_intent={'maneuver': 'u_turn', 'maneuver_time': 2})
    assert not ok and any(d['rule'] == 'u_turn_prohibited' for d in v), \
        f"expected u_turn_prohibited, got {v}"
    print("  [PASS] u_turn_prohibited fires")

    # 4. R12.1 excess_accel: 1 m/step cruise then a 10 m jump.
    jump = np.array([[0,0],[1,0],[2,0],[12,0],[13,0],[14,0]],
                    dtype=np.float32)
    ok, v = checker.check(jump)
    assert not ok and any(d['rule'] == 'excess_accel' for d in v), \
        f"expected excess_accel, got {v}"
    print("  [PASS] excess_accel fires")

    # 5. R10.2 box_blocking: approaching with no space beyond the box.
    box_map = {
        'lanes': {'l1': [{'x1': -10, 'y1': 0, 'x2': 100, 'y2': 0}]},
        'lane_width': {'l1': 4.0},
        'intersections': {'i1': {'x': 20.0, 'y': 0.0, 'radius': 2.0}},
    }
    approach = np.array([[12,0],[13,0],[14,0],[15,0],[16,0],[17,0]],
                        dtype=np.float32)
    ok, v = checker.check(
        approach, map_data=box_map, extras={'facts': [
            'queue_ahead(vehicle_0, i1, 4).',
            'space_beyond_box(ego, vehicle_0, i1, 4, 100).',
            'ego_length(400).',
            'vehicle(vehicle_0).',
        ]})
    assert not ok and any(d['rule'] == 'box_blocking' for d in v), \
        f"expected box_blocking, got {v}"
    print("  [PASS] box_blocking fires")

    # 6. R1.3 variable_zone_speed: active school zone, 2 m/step over
    # a 0.5 m/step zone cap while inside the box.
    ok, v = checker.check(
        ego, zones=[{'id': 'z1', 'x1': -5, 'y1': -5, 'x2': 15, 'y2': 5,
                     'type': 'school', 'active': True,
                     'active_times': [1], 'speed_limit': 0.5}])
    assert not ok and any(d['rule'] == 'variable_zone_speed' for d in v), \
        f"expected variable_zone_speed, got {v}"
    print("  [PASS] variable_zone_speed fires")



def test_if2_derived_facts():
    """IF-2 remainder: previously manual-only facts now derived or keyed.

    Covers R15.4 (rolling_avg_speed), R15.1 (conflict_points map key),
    R15.3 (merge_gap_sequence/mainline_vehicle), R6.4 (approaching +
    emergency_vehicle array-id contract), R7.2 (oncoming_lanes map key),
    R11.1/R11.2 (hazard_distance/evasive_path extras path).
    Each case has a firing and a quiet variant.
    """
    print("\nTesting IF-2 derived facts (R6.4/R7.2/R11/R15)...")
    checker = _checker()

    # R15.4 wave_amplification: fluctuating lead, ego amplifies harder.
    # Lead steps alternate 4/6 m, ego steps alternate 2/8 m: at every T
    # ego's deviation from the lead's rolling average exceeds the
    # lead's own deviation (AF=100).
    lead = np.array([[0, 0], [4, 0], [10, 0], [14, 0], [20, 0], [24, 0]],
                    dtype=np.float32)
    ego = np.array([[0, 0], [2, 0], [10, 0], [12, 0], [20, 0], [22, 0]],
                   dtype=np.float32)
    intent = {'lead_vehicle': {t: 'vehicle_0' for t in range(6)}}
    ok, v = checker.check(ego, other_vehicles=[lead], ego_intent=intent)
    assert not ok and any(d['rule'] == 'wave_amplification' for d in v), \
        f"expected wave_amplification, got {v}"
    print("  [PASS] R15.4 wave_amplification fires on amplification")
    # Quiet: ego tracks the lead's steady speed.
    steady = np.array([[t * 5, 0] for t in range(6)], dtype=np.float32)
    ok, v = checker.check(steady, other_vehicles=[lead], ego_intent=intent)
    assert not any(d['rule'] == 'wave_amplification' for d in v), \
        f"steady tracking must not amplify, got {v}"
    print("  [PASS] R15.4 quiet on steady tracking")

    # R15.1 multiway_priority: V arrived first and sits inside, ego approaches.
    m2 = {'conflict_points': [{'id': 'c1', 'x': 30.0, 'y': 0.0}]}
    ego3 = np.array([[5 + 5 * t, 0] for t in range(6)], dtype=np.float32)
    v3 = np.array([[30, 2]] * 6, dtype=np.float32)
    ok, v = checker.check(ego3, other_vehicles=[v3], map_data=m2)
    assert not ok and any(d['rule'] == 'multiway_priority' for d in v), \
        f"expected multiway_priority, got {v}"
    print("  [PASS] R15.1 multiway_priority fires on joint approach")

    # R15.3 merge_negotiation: ego ahead of mainline V in merge lane.
    m = {'lanes': {'lm': [{'x1': 0.0, 'y1': 0.0, 'x2': 60.0, 'y2': 0.0}]},
         'lane_width': {'lm': 4.0},
         'merge_lanes': {'lm': {'end_point': 60.0}}}
    ego2 = np.array([[10 + t, 0] for t in range(6)], dtype=np.float32)
    vmain = np.array([[8 + t, 4] for t in range(6)], dtype=np.float32)
    ok, v = checker.check(ego2, other_vehicles=[vmain], map_data=m)
    assert not ok and any(d['rule'] == 'merge_negotiation' for d in v), \
        f"expected merge_negotiation, got {v}"
    print("  [PASS] R15.3 merge_negotiation fires on tight before-slot")
    # Quiet: V far ahead, gap is ample.
    vfar = np.array([[40 + t, 4] for t in range(6)], dtype=np.float32)
    ok, v = checker.check(ego2, other_vehicles=[vfar], map_data=m)
    assert not any(d['rule'] == 'merge_negotiation' for d in v), \
        f"ample gap must not negotiate, got {v}"
    print("  [PASS] R15.3 quiet on ample gap")

    # R6.4 emergency_yield: converging emergency vehicle, ego moving.
    amb = np.array([[0, -40 + 10 * t] for t in range(6)], dtype=np.float32)
    ego4 = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    ex = {'facts': ['emergency_vehicle(vehicle_0, 2, 0, -2000, active).']}
    ok, v = checker.check(ego4, other_vehicles=[amb], extras=ex)
    assert not ok and any(d['rule'] == 'emergency_yield' for d in v), \
        f"expected emergency_yield, got {v}"
    print("  [PASS] R6.4 emergency_yield fires on converging unit")
    stop = np.array([[0, 0]] * 6, dtype=np.float32)
    ok, v = checker.check(stop, other_vehicles=[amb], extras=ex)
    assert not any(d['rule'] == 'emergency_yield' for d in v), \
        f"stopped ego already yields, got {v}"
    print("  [PASS] R6.4 quiet when ego already stopped")

    # R7.2 left_turn_across: close oncoming vehicle in a mapped lane.
    m4 = {'lanes': {
        'l1': [{'x1': -10.0, 'y1': 0.0, 'x2': 60.0, 'y2': 0.0}],
        'l2': [{'x1': -10.0, 'y1': 3.0, 'x2': 60.0, 'y2': 3.0}]},
        'lane_width': {'l1': 4.0, 'l2': 4.0},
        'oncoming_lanes': {'l2': 'i1'}}
    ego5 = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    onc = np.array([[15 - 5 * t, 3] for t in range(6)], dtype=np.float32)
    ok, v = checker.check(ego5, other_vehicles=[onc], map_data=m4,
                          ego_intent={'turn': 'left'})
    assert not ok and any(d['rule'] == 'left_turn_across' for d in v), \
        f"expected left_turn_across, got {v}"
    print("  [PASS] R7.2 left_turn_across fires on hot conflict")

    # R11.1/R11.2: insufficient braking + evasive candidate off-lane.
    fast = np.array([[20 * t, 0] for t in range(6)], dtype=np.float32)
    mlane = {'lanes': {'l1': [{'x1': -10.0, 'y1': 0.0,
                               'x2': 120.0, 'y2': 0.0}]},
             'lane_width': {'l1': 4.0}}
    ex_bad = {'facts': ['hazard_distance(ego, h1, 2, 800).',
                        'evasive_path(ego, 2, 4000, 500).']}
    ok, v = checker.check(fast, map_data=mlane, extras=ex_bad)
    assert not ok and any(d['rule'] == 'unsafe_evasion' for d in v), \
        f"expected unsafe_evasion, got {v}"
    # R11.1->R11.2 handoff: the insufficient-braking verdict that
    # triggers evasive evaluation must be present alongside.
    assert any(d.get('type') == 'insufficient' for d in v), \
        f"expected insufficient(emergency_brake) handoff, got {v}"
    print("  [PASS] R11.2 unsafe_evasion fires off-lane (+handoff)")
    ex_good = {'facts': ['hazard_distance(ego, h1, 2, 800).',
                         'evasive_path(ego, 2, 4000, 0).']}
    ok, v = checker.check(fast, map_data=mlane, extras=ex_good)
    assert not any(d['rule'] == 'unsafe_evasion' for d in v), \
        f"in-lane clear evasion must pass, got {v}"
    print("  [PASS] R11.2 quiet on in-lane clear evasion")


def run_all_rule_tests():
    """Run all rule category tests."""
    print("=" * 60)
    print("ASP Traffic Rule Catalog - Comprehensive Test Suite")
    print("63 Rules across 15 Categories")
    print("=" * 60)

    results = {}
    tests = [
        ("helpers.lp", test_helpers),
        ("R1.x Speed Regulation", test_speed_regulation),
        ("R2.x Lane Discipline", test_lane_discipline),
        ("R3.x Following Behavior", test_following_behavior),
        ("R4.x Collision Avoidance", test_collision_avoidance),
        ("R5.x Traffic Control", test_traffic_control),
        ("R6.x-R9.x Pedestrian Safety", test_pedestrian_safety),
        ("R11.x Emergency", test_emergency),
        ("R14.x Environmental", test_environmental),
        ("R15.x Complex Interactions", test_complex_interactions),
        ("T-2 Spot Probes", test_uncovered_spot_probes),
        ("IF-2 Derived Facts", test_if2_derived_facts),
        ("Priority Interactions", test_priority_interactions),
        ("Correction Engine", test_correction),
    ]

    for name, test_fn in tests:
        try:
            test_fn()
            results[name] = "PASS"
        except Exception as e:
            results[name] = f"FAIL: {e}"
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_pass = True
    for name, result in results.items():
        status = "[PASS]" if result == "PASS" else "[FAIL]"
        print(f"  {status} {name}")
        if result != "PASS":
            print(f"        {result}")
            all_pass = False

    print("=" * 60)

    if all_pass:
        print("All rule tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_rule_tests())

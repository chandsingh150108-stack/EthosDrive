"""T-2 coverage battery + P-9 rule registry.

Firing + quiet pairs for every rule family missing from test_rules.py:
R1.4/R1.5, R2.2/R2.4, R3.4, R4.1/R4.3/R4.4/R4.5, R5.2-R5.5, R6.1-R6.3,
R7.1/R7.3-sightline/R7.5, R8.1-R8.4, R9.1/R9.3/R9.4/R9.5, R10.1/R10.3/R10.4,
R11.4, R12.2/R12.3/excess_decel, R13.1-R13.4, R14.2/R14.3, R12.4.

All coordinates in meters (checker scales x100 internally).
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_TESTS_DIR.parent / "src"))

import numpy as np
from checker import TrajectoryChecker

_RULES_DIR = str(_TESTS_DIR.parent / "rules")


def _checker(**kwargs):
    return TrajectoryChecker(_RULES_DIR, **kwargs)


def _lane(y=0.0, x1=-50.0, x2=50.0, w=4.0, lid='l1'):
    return ({'lanes': {lid: [{'x1': x1, 'y1': y, 'x2': x2, 'y2': y}]},
             'lane_width': {lid: w}}, lid)


def test_r1_limits():
    """R1.4 min highway speed + R1.5 VRU proximity speed."""
    c = _checker()
    m, _ = _lane()
    m['road_type'] = {'l1': 'highway'}
    m['min_speed_limit'] = {'l1': 10.0}  # below default cap: feasible band

    slow = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(np.array([[5 * t, 0] for t in range(6)],
                             dtype=np.float32), map_data=m)
    assert not ok and any(d['rule'] == 'min_hwy_speed' for d in v), \
        f"expected min_hwy_speed, got {v}"
    ok, v = c.check(np.array([[12 * t, 0] for t in range(6)],
                             dtype=np.float32), map_data=m)
    assert not any(d['rule'] == 'min_hwy_speed' for d in v), \
        f"12 m/s must satisfy min 10, got {v}"
    print("  [PASS] R1.4 min_hwy_speed fires + quiet")

    fast = np.array([[15 * t, 0] for t in range(6)], dtype=np.float32)
    ped = [{'id': 'p1', 'positions': {1: (15.0, 3.0)}}]
    ok, v = c.check(fast, pedestrians=ped)
    assert not ok and any(d['rule'] == 'vru_proximity_speed' for d in v), \
        f"expected vru_proximity_speed, got {v}"
    ok, v = c.check(slow, pedestrians=ped)
    assert not any(d['rule'] == 'vru_proximity_speed' for d in v), \
        f"2 m/s near VRU must be capped-quiet, got {v}"
    print("  [PASS] R1.5 vru_proximity_speed fires + quiet")


def test_r2_changes():
    """R2.2 solid/gap + R2.4 HOV/bus restrictions."""
    c = _checker()
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}],
                   'l2': [{'x1': -50.0, 'y1': 4.0, 'x2': 50.0, 'y2': 4.0}]},
         'lane_width': {'l1': 3.5, 'l2': 3.5},
         'lane_boundary': [['l1', 'l2', 'solid']]}
    cross = np.array([[0, 0], [2, 1], [4, 3], [6, 4], [8, 4], [10, 4]],
                     dtype=np.float32)
    ok, v = c.check(cross, map_data=m)
    assert not ok and any(d['rule'] == 'illegal_lane_change' for d in v), \
        f"expected illegal_lane_change, got {v}"
    print("  [PASS] R2.2 illegal_lane_change fires on solid crossing")
    m2 = dict(m)
    m2['lane_boundary'] = [['l1', 'l2', 'dashed']]
    ok, v = c.check(cross, map_data=m2)
    assert not any(d['rule'] == 'illegal_lane_change' for d in v), \
        f"dashed crossing must be legal, got {v}"
    print("  [PASS] R2.2 quiet on dashed crossing")

    # Unsafe gap: vehicle sitting at the T+1 crossing point.
    vclose = np.array([[4, 3]] * 6, dtype=np.float32)
    ok, v = c.check(cross, other_vehicles=[vclose], map_data=m2)
    assert not ok and any(d['rule'] == 'unsafe_lane_change_gap'
                          for d in v), f"expected unsafe_lane_change_gap, got {v}"
    print("  [PASS] R2.2 unsafe_lane_change_gap fires on tight slot")

    hov = {'lanes': {'lh': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}]},
           'lane_width': {'lh': 4.0},
           'lane_restriction': {'lh': ['hov', 2, 'always']}}
    center = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(center, map_data=hov, ego_intent={'occupancy': 1})
    assert not ok and any(d['rule'] == 'restricted_lane' for d in v), \
        f"expected restricted_lane, got {v}"
    ok, v = c.check(center, map_data=hov, ego_intent={'occupancy': 3})
    assert not any(d['rule'] == 'restricted_lane' for d in v), \
        f"3 occupants must satisfy HOV-2, got {v}"
    print("  [PASS] R2.4 restricted_lane fires + quiet (HOV)")


def test_r3_queue():
    """R3.4 caravan oscillation on speed swings behind a slow lead."""
    c = _checker()
    # 5 <-> 1 m/s swings behind a slow (1 m/s) queued lead.
    ego = np.array([[0, 0], [5, 0], [6, 0], [11, 0], [12, 0], [17, 0]],
                   dtype=np.float32)
    lead = np.array([[0, 8], [1, 8], [2, 8], [3, 8], [4, 8], [5, 8]],
                    dtype=np.float32)
    intent = {'lead_vehicle': {t: 'vehicle_0' for t in range(6)},
              'queue_members': ['vehicle_0']}
    ok, v = c.check(ego, other_vehicles=[lead], ego_intent=intent)
    assert not ok and any(d['rule'] == 'queue_oscillation' for d in v), \
        f"expected queue_oscillation, got {v}"
    steady = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(steady, other_vehicles=[lead], ego_intent=intent)
    assert not any(d['rule'] == 'queue_oscillation' for d in v), \
        f"steady following must not oscillate, got {v}"
    print("  [PASS] R3.4 queue_oscillation fires + quiet")


def test_r4_proximity():
    """R4.1 lateral, R4.3 cut-in, R4.4 rear-end, R4.5 sideswipe."""
    c = _checker()
    ego = np.array([[t, 0] for t in range(6)], dtype=np.float32)

    # Lateral clearance: alongside within 1 m.
    side = np.array([[3, 0.5]] * 6, dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[side])
    assert not ok and any(d['rule'] == 'lateral_clearance' for d in v), \
        f"expected lateral_clearance, got {v}"
    far = np.array([[3, 6]] * 6, dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[far])
    assert not any(d['rule'] == 'lateral_clearance' for d in v), \
        f"6 m lateral must be clear, got {v}"
    print("  [PASS] R4.1 lateral_clearance fires + quiet")

    # Cut-in: vehicle swings from l2 into ego's l1 within 8 m.
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}],
                   'l2': [{'x1': -50.0, 'y1': 4.0, 'x2': 50.0, 'y2': 4.0}]},
         'lane_width': {'l1': 3.5, 'l2': 3.5}}
    cut = np.array([[8, 4], [9, 0], [10, 0], [11, 0], [12, 0], [13, 0]],
                   dtype=np.float32)
    ego2 = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(ego2, other_vehicles=[cut], map_data=m)
    assert not ok and any(d['rule'] == 'cut_in' for d in v), \
        f"expected cut_in hazard, got {v}"
    stay = np.array([[8 + t, 4] for t in range(6)], dtype=np.float32)
    ok, v = c.check(ego2, other_vehicles=[stay], map_data=m)
    assert not any(d['rule'] == 'cut_in' for d in v), \
        f"staying in lane must not cut in, got {v}"
    print("  [PASS] R4.3 cut_in fires + quiet")

    # Rear-end risk: hard ego decel with a fast close follower.
    brake = np.array([[0, 0], [9, 0], [10, 0], [11, 0], [12, 0], [13, 0]],
                     dtype=np.float32)
    follow = np.array([[-4 + 8 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(brake, other_vehicles=[follow])
    assert not ok and any(d['rule'] == 'rear_end_risk' for d in v), \
        f"expected rear_end_risk, got {v}"
    gentle = np.array([[5 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(gentle, other_vehicles=[follow])
    assert not any(d['rule'] == 'rear_end_risk' for d in v), \
        f"gentle cruise must not endanger followers, got {v}"
    print("  [PASS] R4.4 rear_end_risk fires + quiet")

    # Sideswipe: ego passes within 1 m of V's predicted spot, converging.
    drift = np.array([[t, 0.3 * t] for t in range(6)], dtype=np.float32)
    intent = {'predicted_positions':
              {'vehicle_0': {t: (4.0, 0.9) for t in range(6)}}}
    ok, v = c.check(drift, ego_intent=intent)
    assert not ok and any(d['rule'] == 'sideswipe_risk' for d in v), \
        f"expected sideswipe_risk, got {v}"
    intent2 = {'predicted_positions':
               {'vehicle_0': {t: (4.0, 8.0) for t in range(6)}}}
    ok, v = c.check(drift, ego_intent=intent2)
    assert not any(d['rule'] == 'sideswipe_risk' for d in v), \
        f"diverging paths must not sideswipe, got {v}"
    print("  [PASS] R4.5 sideswipe_risk fires + quiet")


def test_r5_signals():
    """R5.2 yellow zone, R5.3 green+obstacle, R5.4 flashing, R5.5 arrow."""
    c = _checker()
    road = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 100.0, 'y2': 0.0}]},
            'lane_width': {'l1': 4.0},
            'controls_approach': {'i1': 'l1'},
            'intersections': {'i1': {'x': 80.0, 'y': 0.0, 'radius': 5.0}}}

    # Yellow at T=2, steady 10 m/s cruise 60 m out: can stop, doesn't.
    cruise = np.array([[10 * t, 0] for t in range(6)], dtype=np.float32)
    yel = {'i1': {'color': 'yellow', 'times': [2], 'x': 80.0, 'y': 0.0,
                  'radius': 5.0}}
    ok, v = c.check(cruise, map_data=road, traffic_lights=yel)
    assert not ok and any(d['rule'] == 'unsafe_yellow_response'
                          for d in v), f"expected unsafe_yellow_response, got {v}"
    braking = np.array([[0, 0], [10, 0], [20, 0], [25, 0], [28, 0], [30, 0]],
                       dtype=np.float32)
    ok, v = c.check(braking, map_data=road, traffic_lights=yel)
    assert not any(d['rule'] == 'unsafe_yellow_response' for d in v), \
        f"braking for yellow must be excused, got {v}"
    print("  [PASS] R5.2 unsafe_yellow_response fires + quiet")

    # Green with a stalled vehicle inside the box.
    appr = np.array([[60 + 2 * t, 0] for t in range(6)], dtype=np.float32)
    stall = np.array([[80, 1]] * 6, dtype=np.float32)
    grn = {'i1': {'color': 'green', 'times': list(range(6)), 'x': 80.0,
                  'y': 0.0, 'radius': 5.0}}
    ok, v = c.check(appr, other_vehicles=[stall], map_data=road,
                    traffic_lights=grn)
    assert not ok and any(d['rule'] == 'green_with_obstacle' for d in v), \
        f"expected green_with_obstacle, got {v}"
    ok, v = c.check(appr, map_data=road, traffic_lights=grn)
    assert not any(d['rule'] == 'green_with_obstacle' for d in v), \
        f"clear box on green must pass, got {v}"
    print("  [PASS] R5.3 green_with_obstacle fires + quiet")

    # Flashing red rolled through vs stopped at the line.
    roll = np.array([[70 + 2 * t, 0] for t in range(6)], dtype=np.float32)
    flr = {'i1': {'color': 'flashing_red', 'times': list(range(6)),
                  'x': 80.0, 'y': 0.0, 'radius': 5.0}}
    ok, v = c.check(roll, map_data=road, traffic_lights=flr)
    assert not ok and any(d['rule'] == 'flashing_red_no_stop'
                          for d in v), f"expected flashing_red_no_stop, got {v}"
    hold = np.array([[77.5, 0]] * 6, dtype=np.float32)
    ok, v = c.check(hold, map_data=road, traffic_lights=flr)
    assert not any(d['rule'] == 'flashing_red_no_stop' for d in v), \
        f"full stop at flashing red must pass, got {v}"
    print("  [PASS] R5.4 flashing_red_no_stop fires + quiet")

    # Flashing yellow over the caution cap vs under it.
    fly = {'i1': {'color': 'flashing_yellow', 'times': list(range(6)),
                  'x': 80.0, 'y': 0.0, 'radius': 5.0}}
    near = np.array([[76 + 2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(np.array([[76 + 10 * t, 0] for t in range(6)],
                             dtype=np.float32),
                    map_data=road, traffic_lights=fly)
    assert not ok and any(d['rule'] == 'flashing_yellow_speed'
                          for d in v), f"expected flashing_yellow_speed, got {v}"
    ok, v = c.check(near, map_data=road, traffic_lights=fly)
    assert not any(d['rule'] == 'flashing_yellow_speed' for d in v), \
        f"2 m/s on flashing yellow must pass, got {v}"
    print("  [PASS] R5.4 flashing_yellow_speed fires + quiet")

    # Protected left arrow red: turning left is a violation; with a
    # plain green (no arrow) the arrow rule stands down.
    left = {'turn': 'left'}
    arr = {'i1': {'color': 'red', 'times': list(range(6)), 'x': 80.0,
                  'y': 0.0, 'radius': 5.0, 'arrows': {'left': 'red'}}}
    ok, v = c.check(appr, map_data=road, traffic_lights=arr,
                    ego_intent=left)
    assert not ok and any(d['rule'] == 'unprotected_turn' for d in v), \
        f"expected unprotected_turn, got {v}"
    grn2 = {'i1': {'color': 'green', 'times': list(range(6)), 'x': 80.0,
                   'y': 0.0, 'radius': 5.0}}
    ok, v = c.check(appr, map_data=road, traffic_lights=grn2,
                    ego_intent=left)
    assert not any(d['rule'] == 'unprotected_turn' for d in v), \
        f"green ball must excuse the arrow rule, got {v}"
    print("  [PASS] R5.5 unprotected_turn fires + quiet")


def test_r6_priority():
    """R6.1 arrival order + tie, R6.2 roundabout, R6.3 turn yield."""
    c = _checker()
    imap = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 100.0, 'y2': 0.0}]},
            'lane_width': {'l1': 4.0},
            'intersections': {'i1': {'x': 30.0, 'y': 0.0, 'radius': 5.0}}}

    # Cross traffic arrived first and sits inside; ego approaches.
    ego = np.array([[10 + 4 * t, 0] for t in range(6)], dtype=np.float32)
    cross = np.array([[30, 2]] * 6, dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[cross], map_data=imap)
    assert not ok and any(d['rule'] == 'uncontrolled_priority'
                          for d in v), f"expected uncontrolled_priority, got {v}"
    print("  [PASS] R6.1 uncontrolled_priority fires on late arrival")
    # Quiet: the other vehicle already cleared far away.
    gone = np.array([[200 + t, 0] for t in range(6)], dtype=np.float32)
    gone_c = _checker(max_agent_range_m=1000.0)
    ok, v = gone_c.check(ego, other_vehicles=[gone], map_data=imap)
    assert not any(d['rule'] == 'uncontrolled_priority' for d in v), \
        f"cleared traffic must not block, got {v}"
    print("  [PASS] R6.1 quiet when the other vehicle cleared")

    # Roundabout: circulating vehicle present vs absent.
    ex_circ = {'facts': ['roundabout(r1, 3000, 0, 800).',
                         'in_circulation(vehicle_0, r1, 2).',
                         'vehicle(vehicle_0).']}
    ego_r = np.array([[10 + 2 * t, 0] for t in range(6)], dtype=np.float32)
    circ = np.array([[30, 5]] * 6, dtype=np.float32)
    ok, v = c.check(ego_r, other_vehicles=[circ], extras=ex_circ)
    assert not ok and any(d['rule'] == 'roundabout_entry' for d in v), \
        f"expected roundabout_entry, got {v}"
    ok, v = c.check(ego_r, extras={'facts':
                                   ['roundabout(r1, 3000, 0, 800).']})
    assert not any(d['rule'] == 'roundabout_entry' for d in v), \
        f"empty roundabout must allow entry, got {v}"
    print("  [PASS] R6.2 roundabout_entry fires + quiet")

    # Turn yield: hot oncoming vs distant oncoming (left turn).
    ego_t = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    hot = np.array([[30 - 4 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(ego_t, other_vehicles=[hot],
                    ego_intent={'turn': 'left'})
    assert not ok and any(d['rule'] == 'turn_yield' for d in v), \
        f"expected turn_yield, got {v}"
    cold = np.array([[200 - 4 * t, 0] for t in range(6)], dtype=np.float32)
    cold_c = _checker(max_agent_range_m=1000.0)
    ok, v = cold_c.check(ego_t, other_vehicles=[cold],
                         ego_intent={'turn': 'left'})
    assert not any(d['rule'] == 'turn_yield' for d in v), \
        f"distant oncoming must not block the turn, got {v}"
    print("  [PASS] R6.3 turn_yield fires + quiet")


def test_r7_turns():
    """R7.1 RTOR gates, R7.3 sightline U-turn, R7.5 turning radius."""
    c = _checker()
    road = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 100.0, 'y2': 0.0}]},
            'lane_width': {'l1': 4.0},
            'controls_approach': {'i1': 'l1'},
            'intersections': {'i1': {'x': 80.0, 'y': 0.0, 'radius': 5.0}}}
    red = {'i1': {'color': 'red', 'times': list(range(6)), 'x': 80.0,
                  'y': 0.0, 'radius': 5.0}}
    appr = np.array([[60 + 2 * t, 0] for t in range(6)], dtype=np.float32)
    right = {'turn': 'right'}

    # Sign gate.
    m_sign = dict(road)
    m_sign['no_turn_on_red'] = {'i1': ['right']}
    ok, v = c.check(appr, map_data=m_sign, traffic_lights=red,
                    ego_intent=right)
    assert not ok and any(d['rule'] == 'rtor_no_sign' for d in v), \
        f"expected rtor_no_sign, got {v}"
    print("  [PASS] R7.1 rtor_no_sign fires on prohibited turn")

    # Stop gate: rolling turn with no prior stop.
    ok, v = c.check(appr, map_data=road, traffic_lights=red,
                    ego_intent=right)
    assert not ok and any(d['rule'] == 'rtor_no_stop' for d in v), \
        f"expected rtor_no_stop, got {v}"
    print("  [PASS] R7.1 rtor_no_stop fires on rolling turn")

    # Clear-path gate: stopped, but a pedestrian blocks the box.
    hold = np.array([[74, 0]] * 6, dtype=np.float32)
    ped = [{'id': 'p1', 'positions': {t: (80.0, 0.0) for t in range(6)}}]
    ok, v = c.check(hold, map_data=road, traffic_lights=red,
                    pedestrians=ped, ego_intent=right)
    assert not ok and any(d['rule'] == 'rtor_unclear' for d in v), \
        f"expected rtor_unclear, got {v}"
    print("  [PASS] R7.1 rtor_unclear fires on blocked path")

    # Quiet: stopped, clear, no prohibiting sign.
    ok, v = c.check(hold, map_data=road, traffic_lights=red,
                    ego_intent=right)
    assert not any(d['rule'].startswith('rtor') for d in v), \
        f"legal RTOR must be quiet, got {v}"
    print("  [PASS] R7.1 quiet on legal stop-then-turn")

    # U-turn sightline: 10 m sightline on a 13.4 m/s lane (needs 40 m).
    umap = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}]},
            'lane_width': {'l1': 4.0},
            'speed_limit': {'l1': 13.4},
            'sightline': {'l1': 10.0}}
    utraj = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(utraj, map_data=umap,
                    ego_intent={'maneuver': 'u_turn', 'maneuver_time': 2})
    assert not ok and any(d['rule'] == 'u_turn_sightline' for d in v), \
        f"expected u_turn_sightline, got {v}"
    print("  [PASS] R7.3 u_turn_sightline fires on blind segment")

    # Turning radius: 2.5 m corner vs 5.5 m minimum.
    corner = np.array([[0, 0], [4, 0], [4, 3], [4, 7], [4, 11], [4, 15]],
                      dtype=np.float32)
    ex_r = {'facts': ['min_turning_radius(ego, 550).']}
    ok, v = c.check(corner, extras=ex_r)
    assert not ok and any(d['rule'] == 'turn_radius' for d in v), \
        f"expected turn_radius, got {v}"
    straight = np.array([[4 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(straight, extras=ex_r)
    assert not any(d['rule'] == 'turn_radius' for d in v), \
        f"straight must satisfy any radius, got {v}"
    print("  [PASS] R7.5 turn_radius fires + quiet")


def test_r8_overtake():
    """R8.1 solid/oncoming, R8.2 zones, R8.3 right-side, R8.4 clearance."""
    c = _checker()
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}],
                   'l2': [{'x1': -50.0, 'y1': 4.0, 'x2': 50.0, 'y2': 4.0}]},
         'lane_width': {'l1': 3.5, 'l2': 3.5},
         'lane_boundary': [['l1', 'l2', 'solid']]}
    over = {'maneuver': 'overtake', 'maneuver_time': 1}
    cross = np.array([[0, 0], [2, 1], [4, 3], [6, 4], [8, 4], [10, 4]],
                     dtype=np.float32)
    ok, v = c.check(cross, map_data=m, ego_intent=over)
    assert not ok and any(d['rule'] == 'unsafe_overtake' for d in v), \
        f"expected unsafe_overtake on solid line, got {v}"
    print("  [PASS] R8.1 unsafe_overtake fires on solid crossing")

    # Oncoming conflict inside duration + margin with a dashed line.
    # (Maneuver at T=0, where the oncoming derivation holds.)
    m2 = dict(m)
    m2['lane_boundary'] = [['l1', 'l2', 'dashed']]
    over0 = {'maneuver': 'overtake', 'maneuver_time': 0}
    # Oncoming vehicle inside ego's 5 m lateral corridor at T=0.
    onc = np.array([[10 - 4 * t, 2] for t in range(6)], dtype=np.float32)
    ok, v = c.check(cross, other_vehicles=[onc], map_data=m2,
                    ego_intent=over0)
    assert not ok and any(d['rule'] == 'unsafe_overtake' for d in v), \
        f"expected unsafe_overtake on hot oncoming, got {v}"
    print("  [PASS] R8.1 unsafe_overtake fires on hot oncoming")

    # No-passing zone box + sightline floor.
    m3 = {'lanes': {'l1': [{'x1': 0.0, 'y1': 0.0, 'x2': 60.0, 'y2': 0.0}]},
          'lane_width': {'l1': 4.0},
          'speed_limit': {'l1': 13.4},
          'sightline': {'l1': 10.0},
          'no_passing_zones': [{'id': 'z2', 'x1': 20.0, 'y1': -5.0,
                                 'x2': 40.0, 'y2': 5.0}]}
    straight = np.array([[5 * t, 0] for t in range(6)], dtype=np.float32)
    # Zone box covers x=20..40: maneuver timed when ego is inside (T=4).
    over4 = {'maneuver': 'overtake', 'maneuver_time': 4}
    ok, v = c.check(straight, map_data=m3, ego_intent=over4)
    assert not ok and any(d['rule'] == 'no_passing_zone' for d in v), \
        f"expected no_passing_zone, got {v}"
    print("  [PASS] R8.2 no_passing_zone fires inside the box")
    # Sightline floor is maneuver-time independent; T=1 keeps it simple.
    ok, v = c.check(straight, map_data=m3, ego_intent=over)
    assert not ok and any(d['rule'] == 'no_passing_sightline' for d in v), \
        f"expected no_passing_sightline, got {v}"
    print("  [PASS] R8.2 no_passing_sightline fires on blind segment")

    # Right-side pass on a two-lane road, no left-turn exception.
    m4 = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}]},
          'lane_width': {'l1': 4.0},
          'road_type': {'l1': 'two_lane'}}
    over_r = {'maneuver': 'overtake', 'maneuver_time': 2,
              'overtake_side': {2: 'right'}}
    ok, v = c.check(straight, map_data=m4, ego_intent=over_r)
    assert not ok and any(d['rule'] == 'right_overtake' for d in v), \
        f"expected right_overtake, got {v}"
    over_r2 = dict(over_r)
    over_r2['lead_turning_left'] = ['vehicle_0']
    ok, v = c.check(straight, map_data=m4, ego_intent=over_r2)
    assert not any(d['rule'] == 'right_overtake' for d in v), \
        f"left-turning lead excuses the pass, got {v}"
    print("  [PASS] R8.3 right_overtake fires + excused")

    # Cyclist clearance: 0.6 m vs 0.9 m minimum.
    cyc = np.array([[5 + t, 0.6] for t in range(6)], dtype=np.float32)
    ex_c = {'facts': ['overtaken_type(vehicle_0, cyclist).',
                      'vehicle(vehicle_0).']}
    ok, v = c.check(straight, other_vehicles=[cyc], ego_intent=over,
                    extras=ex_c)
    assert not ok and any(d['rule'] == 'overtake_clearance' for d in v), \
        f"expected overtake_clearance, got {v}"
    wide = np.array([[5 + t, 3.0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(straight, other_vehicles=[wide], ego_intent=over,
                    extras=ex_c)
    assert not any(d['rule'] == 'overtake_clearance' for d in v), \
        f"3 m clearance must pass, got {v}"
    print("  [PASS] R8.4 overtake_clearance fires + quiet")


def test_r9_vru():
    """R9.1 unmarked corners, R9.3 bike lanes, R9.4 blind spots, R9.5 curbs."""
    c = _checker()

    # Unmarked corner: ped near the intersection, ego driving through.
    m = {'intersections': {'i4': {'x': 30.0, 'y': 0.0, 'radius': 4.0}}}
    ped = [{'id': 'p2', 'positions': {t: (30.0, 5.0) for t in range(6)}}]
    thru = np.array([[20 + 2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(thru, map_data=m, pedestrians=ped)
    assert not ok and any(d['rule'] == 'implied_crosswalk_yield'
                          for d in v), f"expected implied_crosswalk_yield, got {v}"
    stop = np.array([[20, 0]] * 6, dtype=np.float32)
    ok, v = c.check(stop, map_data=m, pedestrians=ped)
    assert not any(d['rule'] == 'implied_crosswalk_yield' for d in v), \
        f"stopped ego already yields, got {v}"
    print("  [PASS] R9.1 implied_crosswalk_yield fires + quiet")

    # Bike lane used as a travel lane.
    mb = {'lanes': {'l8': [{'x1': -50.0, 'y1': 2.0, 'x2': 50.0, 'y2': 2.0}]},
          'lane_width': {'l8': 2.0},
          'lane_type': {'l8': 'bicycle'}}
    ride = np.array([[t, 2] for t in range(6)], dtype=np.float32)
    ok, v = c.check(ride, map_data=mb)
    assert not ok and any(d['rule'] == 'bike_lane_encroachment'
                          for d in v), f"expected bike_lane_encroachment, got {v}"
    print("  [PASS] R9.3 bike_lane_encroachment fires")

    # Blind spot: cyclist in the right-rear zone during a right change.
    ped2 = [{'id': 'p3', 'type': 'cyclist',
             'positions': {t: (1.5, -5.0) for t in range(6)}}]
    intent = {'maneuver': 'lane_change', 'maneuver_time': 2,
              'maneuver_direction': 'right',
              'blind_spots': {'right': {'x1': 0.0, 'y1': -8.0,
                                        'x2': 3.0, 'y2': -2.0}}}
    ego = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(ego, pedestrians=ped2, ego_intent=intent)
    assert not ok and any(d['rule'] == 'blind_spot_maneuver' for d in v), \
        f"expected blind_spot_maneuver, got {v}"
    ped3 = [{'id': 'p3', 'type': 'cyclist',
             'positions': {t: (1.5, 5.0) for t in range(6)}}]
    ok, v = c.check(ego, pedestrians=ped3, ego_intent=intent)
    assert not any(d['rule'] == 'blind_spot_maneuver' for d in v), \
        f"clear zone must allow the change, got {v}"
    print("  [PASS] R9.4 blind_spot_maneuver fires + quiet")

    # Sidewalk overshoot vs legitimate driveway crossing.
    ms = {'sidewalks': [{'id': 's1', 'x1': 0.0, 'y1': 12.0,
                         'x2': 20.0, 'y2': 14.0}]}
    over = np.array([[9 + t, 12.5] for t in range(6)], dtype=np.float32)
    ok, v = c.check(over, map_data=ms)
    assert not ok and any(d['rule'] == 'sidewalk_encroachment'
                          for d in v), f"expected sidewalk_encroachment, got {v}"
    ex_d = {'facts': ['driveway_crossing(ego, 2).']}
    ok, v = c.check(over, map_data=ms, extras=ex_d)
    hits = [d for d in v if d['rule'] == 'sidewalk_encroachment'
            and d['time'] == 2]
    assert not hits, f"driveway crossing must excuse T=2, got {v}"
    print("  [PASS] R9.5 sidewalk_encroachment fires + excused")


def test_r10_queue():
    """R10.1 queue cutting, R10.3 protected lane, R10.4 right-turn ped."""
    c = _checker()

    # Red light queue: ego stops alongside the last car, not behind it.
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}]},
         'lane_width': {'l1': 4.0},
         'intersections': {'i5': {'x': 0.0, 'y': 40.0, 'radius': 5.0}}}
    red = {'i5': {'color': 'red', 'times': list(range(6)), 'x': 0.0,
                  'y': 40.0, 'radius': 5.0}}
    queued = np.array([[0, 32]] * 6, dtype=np.float32)
    ex_q = {'facts': ['last_queued_vehicle(vehicle_0, i5, 4).']}
    cut = np.array([[0, 30 + 0.5 * t] for t in range(6)], dtype=np.float32)
    ok, v = c.check(cut, other_vehicles=[queued], map_data=m,
                    traffic_lights=red, extras=ex_q)
    assert not ok and any(d['rule'] == 'queue_cut' for d in v), \
        f"expected queue_cut, got {v}"
    back = np.array([[0, 10 + 0.5 * t] for t in range(6)], dtype=np.float32)
    ok, v = c.check(back, other_vehicles=[queued], map_data=m,
                    traffic_lights=red, extras=ex_q)
    assert not any(d['rule'] == 'queue_cut' for d in v), \
        f"queuing behind must pass, got {v}"
    print("  [PASS] R10.1 queue_cut fires + quiet")

    # Protected-left pocket used for going straight.
    mp = {'lanes': {'l9': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}]},
          'lane_width': {'l9': 4.0},
          'lane_type': {'l9': 'protected_left'}}
    straight = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(straight, map_data=mp,
                    ego_intent={'maneuver': 'straight',
                                'maneuver_time': 2})
    assert not ok and any(d['rule'] == 'protected_lane_misuse'
                          for d in v), f"expected protected_lane_misuse, got {v}"
    ok, v = c.check(straight, map_data=mp,
                    ego_intent={'maneuver': 'left_turn',
                                'maneuver_time': 2})
    assert not any(d['rule'] == 'protected_lane_misuse' for d in v), \
        f"left turn from the pocket must pass, got {v}"
    print("  [PASS] R10.3 protected_lane_misuse fires + quiet")

    # Right turn into an occupied destination crosswalk.
    mc = {'crosswalks': [{'id': 'c3', 'x1': 10.0, 'y1': -3.0,
                          'x2': 14.0, 'y2': 3.0}]}
    ped = [{'id': 'p4', 'positions': {t: (12.0, 0.0) for t in range(6)}}]
    ex_c = {'facts': ['destination_crosswalk(i5, right, c3).',
                      'path_intersects(ego, c3, 3).']}
    turn = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(turn, map_data=mc, pedestrians=ped,
                    ego_intent={'turn': 'right'}, extras=ex_c)
    assert not ok and any(d['rule'] == 'right_turn_ped_yield'
                          for d in v), f"expected right_turn_ped_yield, got {v}"
    print("  [PASS] R10.4 right_turn_ped_yield fires")


def test_r11_construction():
    """R11.4 shifted-lane + worker clearance in an active zone."""
    c = _checker()
    zones = [{'id': 'z3', 'x1': 0.0, 'y1': -10.0, 'x2': 60.0, 'y2': 10.0,
              'type': 'construction', 'active': True,
              'lane_shifts': [{'lane': 'l1', 'path': 'path_shifted'}],
              'workers': [{'id': 'w1', 'x': 20.0, 'y': 1.0,
                           'times': list(range(6))}]}]
    m = {'lanes': {'l1': [{'x1': 0.0, 'y1': 0.0, 'x2': 60.0, 'y2': 0.0}]},
         'lane_width': {'l1': 4.0}}
    old = np.array([[5 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(old, map_data=m, zones=zones)
    rules = set(d['rule'] for d in v)
    assert not ok and 'construction_lane' in rules, f"expected lane hit, got {v}"
    assert 'construction_worker_clearance' in rules, \
        f"expected worker hit, got {v}"
    print("  [PASS] R11.4 construction_lane + worker clearance fire")
    far = np.array([[5 * t, 30] for t in range(6)], dtype=np.float32)
    ok, v = c.check(far, zones=zones)
    assert not any('construction' in d['rule'] for d in v), \
        f"outside the zone must be quiet, got {v}"
    print("  [PASS] R11.4 quiet outside the zone")


def test_r12_comfort_shapes():
    """R12.1 decel, R12.2 lateral accel/jerk, R12.3 reversal."""
    c = _checker()

    # Hard stop 20 -> 5 m/s in one step.
    brake = np.array([[0, 0], [20, 0], [25, 0], [30, 0], [35, 0], [40, 0]],
                     dtype=np.float32)
    ok, v = c.check(brake)
    assert not ok and any(d['rule'] == 'excess_decel' for d in v), \
        f"expected excess_decel, got {v}"
    cruise = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(cruise)
    assert not any(d['rule'] in ('excess_decel', 'excess_accel')
                   for d in v), f"cruise must be comfortable, got {v}"
    print("  [PASS] R12.1 excess_decel fires + quiet")

    # Fast 90-degree kink: lateral accel (6 m steps escape the slow band).
    kink = np.array([[0, 0], [6, 0], [12, 0], [12, 6], [12, 12], [12, 18]],
                    dtype=np.float32)
    ok, v = c.check(kink)
    assert not ok and any(d['rule'] == 'lateral_accel' for d in v), \
        f"expected lateral_accel, got {v}"
    print("  [PASS] R12.2 lateral_accel fires on fast kink")

    # Alternating kink sizes: jerk (change of curvature).
    ess = np.array([[0, 0], [10, 0], [20, 1], [30, -14], [40, -13],
                    [50, -28]], dtype=np.float32)
    ok, v = c.check(ess)
    assert not ok and any(d['rule'] == 'lateral_jerk' for d in v), \
        f"expected lateral_jerk, got {v}"
    print("  [PASS] R12.2 lateral_jerk fires on S-curve")

    # Out-and-back reversal: discontinuous path.
    back = np.array([[0, 0], [5, 0], [10, 0], [5, 0], [0, 0], [-5, 0]],
                    dtype=np.float32)
    ok, v = c.check(back)
    assert not ok and any(d['rule'] == 'discontinuous_path' for d in v), \
        f"expected discontinuous_path, got {v}"
    print("  [PASS] R12.3 discontinuous_path fires on reversal")


def test_r13_predictive():
    """R13.1 collision prediction, R13.2 bands, R13.3 arbitration,
    R13.4 horizon aggregation, R12.4 comfort scoring (all internal)."""
    c = _checker()

    # Predicted paths meet at T=4 inside 5 m.
    ego = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    intent = {'predicted_positions':
              {'vehicle_0': {t: (8.0, 0.0) for t in range(6)}}}
    ok, v = c.check(ego, ego_intent=intent)
    assert not ok and any(d['rule'] == 'predicted_collision' for d in v), \
        f"expected predicted_collision, got {v}"
    intent2 = {'predicted_positions':
               {'vehicle_0': {t: (8.0, 50.0) for t in range(6)}}}
    ok, v = c.check(ego, ego_intent=intent2)
    assert not any(d['rule'] == 'predicted_collision' for d in v), \
        f"distant predictions must not collide, got {v}"
    print("  [PASS] R13.1 predicted_collision fires + quiet")

    # Bands escalate as the gap closes; threat mirrors the max band.
    fast = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    close = np.array([[30 - 3 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(fast, other_vehicles=[close],
                    ego_intent={'lead_vehicle':
                                {t: 'vehicle_0' for t in range(6)}},
                    include_internal=True)
    bands = set((d['band'], d['time']) for d in v if d.get('type') == 'ttc')
    assert bands, f"expected ttc bands, got {v}"
    levels = set((d['level'], d['time']) for d in v
                 if d.get('type') == 'threat')
    assert levels, f"expected threat levels, got {v}"
    # Threat never exceeds two labels at one timestep (arbitration).
    per_t = {}
    for d in v:
        if d.get('type') == 'threat':
            per_t.setdefault(d['time'], set()).add(d['level'])
    assert all(len(s) == 1 for s in per_t.values()), \
        f"arbitration must collapse to one label: {per_t}"
    print(f"  [PASS] R13.2 bands + R13.3 single-label arbitration {sorted(levels)}")

    # Critical band + insufficient braking collapses to imminent.
    ex = {'facts': ['closing_rate(ego, vehicle_0, 3, 400).',
                    'gap_dist(ego, vehicle_0, 3, 700).',
                    'hazard_distance(ego, h1, 3, 100).',
                    'vehicle(vehicle_0).']}
    slow = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(slow, extras=ex, include_internal=True)
    t3 = set(d['level'] for d in v if d.get('type') == 'threat'
             and d['time'] == 3)
    assert t3 == {'imminent'}, f"expected imminent-only @3, got {v}"
    print("  [PASS] R13.3 critical+insufficient resolves to imminent")

    # Horizon aggregation + comfort score ride along real violations.
    ok, v = c.check(np.array([[20 * t, 0] for t in range(6)],
                             dtype=np.float32), include_internal=True)
    assert any(d.get('type') == 'future' for d in v), \
        f"expected future_violation entries, got {v}"
    assert any(d.get('type') == 'any' for d in v), \
        f"expected any_violation, got {v}"
    scores = [d['score'] for d in v if d.get('type') == 'comfort']
    accels = sum(1 for d in v if d['rule'] == 'excess_accel')
    assert scores and scores[0] == accels, \
        f"comfort_score must count excess_accel: {v}"
    print(f"  [PASS] R13.4 aggregation + R12.4 comfort_score={scores[0]}")


def test_r14_night_visibility():
    """R14.2 headlight overdrive + R14.3 low-visibility caps."""
    c = _checker()

    # Night, 50 m headlights, fast band needs 60 m: overdrive.
    fast = np.array([[15 * t, 0] for t in range(6)], dtype=np.float32)
    night = {'time_of_day': {t: 'night' for t in range(6)},
             'headlight_range': 50.0}
    ok, v = c.check(fast, weather=night)
    assert not ok and any(d['rule'] == 'overdriving_headlights'
                          for d in v), f"expected overdriving_headlights, got {v}"
    slow = np.array([[2 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(slow, weather=night)
    assert not any(d['rule'] == 'overdriving_headlights' for d in v), \
        f"slow cruise must fit the beams, got {v}"
    print("  [PASS] R14.2 overdriving_headlights fires + quiet")

    # Degraded perception caps speed below the 15 m/s cruise.
    fog = {'perception_confidence': {t: 'low' for t in range(6)},
           'confidence_speed_caps': {'low': 11.0}}
    ok, v = c.check(fast, weather=fog)
    assert not ok and any(d['rule'] == 'low_visibility_speed'
                          for d in v), f"expected low_visibility_speed, got {v}"
    mid = np.array([[8 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(mid, weather=fog)
    assert not any(d['rule'] == 'low_visibility_speed' for d in v), \
        f"8 m/s under an 11 cap must pass, got {v}"
    print("  [PASS] R14.3 low_visibility_speed fires + quiet")


def test_r2_transition():
    """R2.1 excuses the midpoint bridged between two lanes (P-4/N-1)."""
    c = _checker()
    m = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0, 'x2': 50.0, 'y2': 0.0}],
                   'l2': [{'x1': -50.0, 'y1': 4.0, 'x2': 50.0, 'y2': 4.0}]},
         'lane_width': {'l1': 3.5, 'l2': 3.5}}
    # T=2 straddles the boundary; neighbors sit in l1 then l2.
    cross = np.array([[0, 0], [2, 0], [4, 1.9], [6, 4], [8, 4], [10, 4]],
                     dtype=np.float32)
    ok, v = c.check(cross, map_data=m)
    assert not any(d['rule'] == 'lane_keeping' and d['time'] == 2
                   for d in v), f"transition midpoint must be excused, got {v}"
    # Genuinely off-lane with no lane context still fires.
    off = np.array([[0, 20], [2, 20], [4, 20], [6, 20], [8, 20], [10, 20]],
                   dtype=np.float32)
    ok, v = c.check(off, map_data=m)
    assert not ok and any(d['rule'] == 'lane_keeping' for d in v), \
        f"off-lane must fire, got {v}"
    print("  [PASS] R2.1 transition excused, departure fires")


def test_rule_registry():
    """P-9: every catalog violation/hazard head has a firing test.

    Self-hosting tripwire: adding a rule predicate without a quoted
    assertion anywhere under tests/ fails here. Warning/advisory and
    solver-internal atoms (ttc/threat/comfort/future/any) are covered
    by dedicated tests, not by name match.
    """
    import re
    catalog = set()
    for lp in _TESTS_DIR.parent.glob("rules/*.lp"):
        src = lp.read_text()
        for m in re.finditer(r"(?:violation|hazard|warning)\(\s*([a-z][a-z_0-9]*)",
                             src):
            catalog.add(m.group(1))
    catalog.discard("emergency_brake")  # insufficient/4 payload, not a rule
    asserted = set()
    for tf in _TESTS_DIR.glob("test_*.py"):
        for m in re.finditer(r"['\"]([a-z_]+)['\"]", tf.read_text()):
            if m.group(1) in catalog:
                asserted.add(m.group(1))
    missing = sorted(catalog - asserted)
    assert not missing, f"rules without firing tests: {missing}"
    print(f"  [PASS] Registry: {len(catalog)} findings, all asserted")


if __name__ == "__main__":
    test_r1_limits()
    test_r2_changes()
    test_r2_transition()
    test_r3_queue()
    test_r4_proximity()
    test_r5_signals()
    test_r6_priority()
    test_r7_turns()
    test_r8_overtake()
    test_r9_vru()
    test_r10_queue()
    test_r11_construction()
    test_r12_comfort_shapes()
    test_r13_predictive()
    test_r14_night_visibility()
    test_rule_registry()
    print("T-2 battery complete!")

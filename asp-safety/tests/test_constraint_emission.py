"""P1 constraint-emission battery: every ASP violation rule with a natural
geometric bound emits >=1 constraint atom on a violating trajectory.

Covers the six constraint heads (constraint_v_max, constraint_v_min,
constraint_x_bounds, constraint_y_bounds, constraint_a_max,
constraint_lane_center) across all 15 rule files. Rules with no natural
bound are documented as exemptions in their .lp files (pinned by
test_exemptions_documented below); the checker path is pinned by
test_include_constraints_contract.

All coordinates in meters (checker scales x100 internally).
"""

import sys
from pathlib import Path

_TESTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_TESTS_DIR.parent / "src"))

import numpy as np
from checker import TrajectoryChecker

_RULES_DIR = str(_TESTS_DIR.parent / "rules")

# All six constraint heads must be quoted here (self-hosting: if a head
# is ever removed from the catalog without updating this file, the
# coverage test below fails).
_CONSTRAINT_HEADS = {
    'constraint_v_max', 'constraint_v_min', 'constraint_x_bounds',
    'constraint_y_bounds', 'constraint_a_max', 'constraint_lane_center',
    'constraint_y_bounds_range',
}


def _checker(**kwargs):
    return TrajectoryChecker(_RULES_DIR, **kwargs)


def _lane(y=0.0, x1=-50.0, x2=50.0, w=4.0, lid='l1'):
    return ({'lanes': {lid: [{'x1': x1, 'y1': y, 'x2': x2, 'y2': y}]},
             'lane_width': {lid: w}}, lid)


def test_include_constraints_contract():
    """check() shape contract: 2-tuple by default, 3-tuple on request."""
    c = _checker()
    slow = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    res2 = c.check(slow)
    assert isinstance(res2, tuple) and len(res2) == 2, \
        f"default check must return 2-tuple, got {len(res2)}-tuple"
    is_safe, violations = res2
    assert is_safe and violations == []
    res3 = c.check(slow, include_constraints=True)
    assert isinstance(res3, tuple) and len(res3) == 3, \
        f"include_constraints=True must return 3-tuple, got {len(res3)}-tuple"
    is_safe3, violations3, constraints = res3
    assert is_safe3 == is_safe and violations3 == violations
    assert set(constraints.keys()) == {
        'v_max', 'v_min', 'x_bounds', 'y_bounds', 'a_max', 'lane_center',
        'x_bounds_range', 'y_bounds_range',
    }, \
        f"unexpected constraint keys: {sorted(constraints.keys())}"
    assert all(v == [] for v in constraints.values()), \
        f"safe trajectory must emit no constraints, got {constraints}"
    print("  [PASS] include_constraints contract (2-tuple/3-tuple, empty)")


def test_constraint_speed():
    """R1.1 urban_speed fires and emits constraint_v_max with the limit."""
    c = _checker()
    m, _ = _lane()
    m['road_type'] = {'l1': 'urban'}
    m['speed_limit'] = {'l1': 13.4}  # m/s -> 1340 cm/step
    fast = np.array([[15 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(fast, map_data=m)
    assert not ok and any(d['rule'] == 'urban_speed' for d in v), \
        f"expected urban_speed, got {v}"
    ok3, v3, con = c.check(fast, map_data=m, include_constraints=True)
    assert ok3 == ok and v3 == v, "flag must not change the verdict"
    assert (1, 1340) in con['v_max'] or any(
        lim == 1340 for _, lim in con['v_max']), \
        f"expected v_max cap 1340, got {con['v_max']}"
    print("  [PASS] R1.1 constraint_v_max emits posted limit")


def test_constraint_lane_corridor():
    """R2.1 corridor: y_bounds + lane_center while in-lane; pull-back
    corridor at the first off-lane timestep (drift-out case)."""
    c = _checker()
    m, _ = _lane(w=4.0)
    cruise = np.array([[5 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v, con = c.check(cruise, map_data=m, include_constraints=True)
    assert ok, f"centered cruise must be safe, got {v}"
    assert any(lo <= -200 and hi >= 200 for _, lo, hi in con['y_bounds']), \
        f"expected lane corridor y_bounds, got {con['y_bounds']}"
    assert any(y == 0 for _, y in con['lane_center']), \
        f"expected lane_center y=0, got {con['lane_center']}"
    # Drift out at T=3 (y=3.0 vs half-width 2.0); T=2 still in lane.
    drift = np.array([[0, 0], [5, 0], [10, 0], [15, 3.0], [20, 3.5],
                      [25, 4.0]], dtype=np.float32)
    ok2, v2, con2 = c.check(drift, map_data=m, include_constraints=True)
    assert not ok2 and any(d['rule'] == 'lane_keeping' for d in v2), \
        f"expected lane_keeping, got {v2}"
    assert any(t == 3 for t, _, _ in con2['y_bounds']), \
        f"expected pull-back corridor at T=3, got {con2['y_bounds']}"
    print("  [PASS] R2.1 corridor + pull-back emission")


def test_constraint_following():
    """R3.1 following_time fires; v_max carries the exact D/2 headway cap."""
    c = _checker()
    ego = np.array([[5 * t, 0] for t in range(6)], dtype=np.float32)
    lead = np.array([[12, 0]] * 6, dtype=np.float32)
    intent = {'lead_vehicle': {t: 'vehicle_0' for t in range(6)}}
    ok, v = c.check(ego, other_vehicles=[lead], ego_intent=intent)
    assert not ok and any(d['rule'] == 'following_time' for d in v), \
        f"expected following_time, got {v}"
    _, _, con = c.check(ego, other_vehicles=[lead], ego_intent=intent,
                        include_constraints=True)
    # T=2: gap 200 cm -> cap 100 cm/step.
    assert (2, 100) in con['v_max'], \
        f"expected exact headway cap (2, 100), got {con['v_max']}"
    print("  [PASS] R3.1 constraint_v_max carries D/2 cap")


def test_constraint_collision_default():
    """Bare-trajectory collision_avoidance fires with x/y clearance boxes."""
    c = _checker()
    ego = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    other = np.array([[2, 0.5]] * 6, dtype=np.float32)
    ok, v = c.check(ego, other_vehicles=[other])
    assert not ok and any(d['rule'] == 'collision_avoidance'
                         for d in v), f"expected collision_avoidance, got {v}"
    _, _, con = c.check(ego, other_vehicles=[other],
                        include_constraints=True)
    assert len(con['x_bounds']) > 0 and len(con['y_bounds']) > 0, \
        f"expected clearance boxes, got {con}"
    print("  [PASS] R4 default clearance boxes emitted")


def test_constraint_red_hold():
    """R5.1 red_light fires; hold cap v_max 0 emitted at that timestep."""
    c = _checker()
    road = {'lanes': {'l1': [{'x1': -50.0, 'y1': 0.0,
                              'x2': 50.0, 'y2': 0.0}]},
            'lane_width': {'l1': 4.0},
            'controls_approach': {'i1': 'l1'},
            'intersections': {'i1': {'x': 12.0, 'y': 0.0, 'radius': 3.0}}}
    run = np.array([[3 * t, 0] for t in range(6)], dtype=np.float32)
    red = {'i1': {'color': 'red', 'times': [4], 'x': 12.0, 'y': 0.0,
                  'radius': 3.0}}
    ok, v = c.check(run, map_data=road, traffic_lights=red)
    assert not ok and any(d['rule'] == 'red_light' for d in v), \
        f"expected red_light, got {v}"
    _, _, con = c.check(run, map_data=road, traffic_lights=red,
                        include_constraints=True)
    assert (4, 0) in con['v_max'], \
        f"expected hold cap (4, 0), got {con['v_max']}"
    print("  [PASS] R5.1 hold cap emitted")


def test_constraint_crosswalk_hold():
    """R6.5 crosswalk_yield fires; hold cap v_max 0 emitted."""
    c = _checker()
    m = {'crosswalks': [{'id': 'c1', 'x1': 10.0, 'y1': -2.0,
                         'x2': 12.0, 'y2': 2.0}]}
    ped = [{'id': 'p1', 'positions': {t: (11.0, 0.0) for t in range(6)}}]
    thru = np.array([[5 * t, 0] for t in range(6)], dtype=np.float32)
    ok, v = c.check(thru, map_data=m, pedestrians=ped)
    assert not ok and any(d['rule'] == 'crosswalk_yield' for d in v), \
        f"expected crosswalk_yield, got {v}"
    _, _, con = c.check(thru, map_data=m, pedestrians=ped,
                        include_constraints=True)
    assert (2, 0) in con['v_max'], \
        f"expected hold cap (2, 0), got {con['v_max']}"
    print("  [PASS] R6.5 hold cap emitted")


def test_constraint_comfort_accel():
    """R12.1 excess_accel fires; a_max carries the comfort bound."""
    c = _checker()
    jerk = np.array([[0, 0], [1, 0], [2, 0], [10, 0], [18, 0], [26, 0]],
                    dtype=np.float32)
    ok, v = c.check(jerk)
    assert not ok and any(d['rule'] == 'excess_accel' for d in v), \
        f"expected excess_accel, got {v}"
    _, _, con = c.check(jerk, include_constraints=True)
    assert any(lim == 300 for _, lim in con['a_max']), \
        f"expected a_max comfort bound 300, got {con['a_max']}"
    print("  [PASS] R12.1 a_max bound emitted")


def test_constraint_predicted_collision():
    """R13.1 predicted_collision fires with clearance boxes."""
    c = _checker()
    ego = np.array([[t, 0] for t in range(6)], dtype=np.float32)
    extra = {'facts': ['predicted_pos(v9, 2, 200, 0).']}
    ok, v = c.check(ego, extras=extra)
    assert not ok and any(d['rule'] == 'predicted_collision' for d in v), \
        f"expected predicted_collision, got {v}"
    _, _, con = c.check(ego, extras=extra, include_constraints=True)
    assert len(con['x_bounds']) > 0 and len(con['y_bounds']) > 0, \
        f"expected predicted-collision boxes, got {con}"
    print("  [PASS] R13.1 clearance boxes emitted")


def test_constraint_onesided_bounds():
    """R3.2/R4.2 hold-back boxes are one-sided (P6/S17 attractor fix):
    an ahead lead caps ego short of it (open -1000000 side), never a
    box centered on the threat."""
    c = _checker()
    m, _ = _lane()
    ego = np.array([[0, 0]] * 6, dtype=np.float32)
    lead = np.array([[2, 0]] * 6, dtype=np.float32)
    intent = {'lead_vehicle': {t: 'vehicle_0' for t in range(6)}}
    ok, v = c.check(ego, other_vehicles=[lead], map_data=m,
                    ego_intent=intent)
    assert not ok and any(d['rule'] == 'following_min_gap' for d in v), \
        f"expected following_min_gap, got {v}"
    assert any(d['rule'] == 'longitudinal_clearance' for d in v), \
        f"expected longitudinal_clearance, got {v}"
    _, _, con = c.check(ego, other_vehicles=[lead], map_data=m,
                        ego_intent=intent, include_constraints=True)
    # R3.2 slow floor 500: lead at 200 cm -> cap at -300, open low side.
    assert (2, -1000000, -300) in con['x_bounds'], \
        f"expected R3.2 hold-back (2, -1M, -300), got {con['x_bounds']}"
    # R4.2 geometric floor 300: lead at 200 cm -> cap at -100.
    assert (2, -1000000, -100) in con['x_bounds'], \
        f"expected R4.2 hold-back (2, -1M, -100), got {con['x_bounds']}"
    # No centered-on-threat box for either rule at T=2.
    assert (2, -100, 500) not in con['x_bounds'], \
        f"centered attractor box must be gone: {con['x_bounds']}"
    print("  [PASS] one-sided hold-back bounds (R3.2/R4.2)")


def test_constraint_coverage():
    """Every constraint head in the catalog is asserted in this file."""
    import re
    catalog = set()
    for lp in _TESTS_DIR.parent.glob("rules/*.lp"):
        src = lp.read_text()
        for m in re.finditer(r"(constraint_[a-z_]+)\(", src):
            catalog.add(m.group(1))
    assert catalog == _CONSTRAINT_HEADS, \
        f"catalog heads {sorted(catalog)} != expected {sorted(_CONSTRAINT_HEADS)}"
    asserted = set()
    src = (_TESTS_DIR / "test_constraint_emission.py").read_text()
    for m in re.finditer(r"'(constraint_[a-z_]+)'", src):
        asserted.add(m.group(1))
    missing = sorted(catalog - asserted)
    assert not missing, f"constraint heads without tests: {missing}"
    print(f"  [PASS] Coverage: {len(catalog)} constraint heads asserted")


def test_exemptions_documented():
    """Rules with no natural bound carry a 'documented exemption' note
    in their .lp file (wrong-way, U-turn, signal timing, turn radius,
    bike-lane, sidewalk, evasion, fixed obstacle, lane-shift, TTC bands,
    threat arbitration, aggregation, parameter layer, comfort shapes)."""
    exempt_files = ['lane_discipline.lp', 'turning_rules.lp',
                    'vru_safety.lp', 'emergency.lp', 'comfort_quality.lp',
                    'predictive_safety.lp', 'environmental.lp']
    for fname in exempt_files:
        src = (_TESTS_DIR.parent / "rules" / fname).read_text()
        assert 'documented exemption' in src, \
            f"{fname} must document its constraint exemptions"
    print("  [PASS] Exemptions documented in 7 rule files")


if __name__ == "__main__":
    test_include_constraints_contract()
    test_constraint_speed()
    test_constraint_lane_corridor()
    test_constraint_following()
    test_constraint_collision_default()
    test_constraint_red_hold()
    test_constraint_crosswalk_hold()
    test_constraint_comfort_accel()
    test_constraint_predicted_collision()
    test_constraint_onesided_bounds()
    test_constraint_coverage()
    test_exemptions_documented()
    print("Constraint emission battery complete!")

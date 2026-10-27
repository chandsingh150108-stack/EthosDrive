"""ASP-based trajectory safety checker.

This module provides the TrajectoryChecker class that uses Clingo
to check trajectories against traffic rules.

Supports the full 63-rule ASP traffic rule catalog (R1.1-R15.4).
All coordinates are scaled by SCALE_FACTOR (100 = centimeters) internally.
Clingo only supports integer arithmetic; no floats or sqrt.
"""

import clingo
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

# Scale factor for converting meters to integers for Clingo
# 100 means centimeter precision
SCALE_FACTOR = 100


class _FactBatcher:
    """Buffer fact strings into one Control.add() call (P-8/PR-3).

    Clingo parses every add() fragment separately: ~1500 one-line
    add() calls cost ~30 ms of re-parsing on a 20-agent scene, while
    one joined add() parses once. Builders keep calling .add() with
    the Control signature; check() flushes once before grounding.
    Fact-level semantics are unchanged (independent ground facts).
    """

    def __init__(self, ctrl) -> None:
        self._ctrl = ctrl
        self._buf = []

    def add(self, part, params, program: str) -> None:
        self._buf.append(program)

    def flush(self) -> None:
        if self._buf:
            self._ctrl.add("base", [], "\n".join(self._buf))
            self._buf = []

# Canonical timestep between consecutive trajectory waypoints, seconds
# (P-1). Every ASP speed/gap threshold below is derived for STEP_DT=1.0:
#   50 km/h cap     = 13.8889 m/s -> 1389 cm/step, squared = 1929321
#   2 s headway     = D/S < 2  <=>  D^2 < 4*S2 (R3.1)
#   TTC deciseconds = gap*10/closing_rate (1.0 s = 10 units)
# The real SparseDrive model emits 0.5 s waypoints over 3 s; its output
# MUST be resampled to STEP_DT before check() (see neural_planner and
# the T-5 golden test). Changing this constant requires re-deriving
# every threshold and is pinned by test_step_dt_contract.
STEP_DT = 1.0


class TrajectoryChecker:
    """ASP-based trajectory safety checker.
    
    Loads the 63-rule ASP traffic rule catalog and checks vehicle
    trajectories for violations across 15 rule categories.
    
    Attributes:
        rules_dir: Path to directory containing .lp rule files
        max_agent_range_m: Cull radius for tracked agents
    """
    
    def __init__(self, rules_dir: str = "rules",
                 max_agent_range_m: float = 150.0):
        """Initialize the checker.

        Args:
            rules_dir: Path to directory containing .lp rule files
            max_agent_range_m: Cull radius for tracked agents (GAP-10).
                Agents farther than this from ego at T=0 are skipped
                so High-impact rules never ground distant VxT pairs.

        Raises:
            FileNotFoundError: If rules_dir doesn't exist
            RuntimeError: If no .lp files found in rules_dir
        """
        self.rules_dir = Path(rules_dir)
        self.max_agent_range_m = max_agent_range_m

        if not self.rules_dir.exists():
            raise FileNotFoundError(f"Rules directory not found: {rules_dir}")

        # Pre-concatenate the catalog once: re-resolving 16 #includes
        # per check costs ~70ms of parsing; a single cached blob parses
        # in ~5ms.
        self._program_blob = self._build_blob()
    
    def _build_blob(self) -> Optional[str]:
        """Concatenate the catalog into one program string.

        Include order follows basic.lp's #include directives, then
        basic.lp's own trailing body (total_violations/safe) is
        appended so the blob matches full basic.lp grounding exactly.
        Returns None when basic.lp is absent (per-file fallback used).
        """
        import re
        basic_file = self.rules_dir / "basic.lp"
        if not basic_file.exists():
            return None
        parts = []
        body = []
        for line in basic_file.read_text().splitlines():
            m = re.match(r'\s*#include\s+"([^"]+)"', line)
            if m:
                inc = self.rules_dir / m.group(1)
                if inc.exists():
                    parts.append(inc.read_text())
            else:
                body.append(line)
        parts.append("\n".join(body))
        blob = "\n".join(parts)
        return blob if blob.strip() else None

    def _load_rules_to(self, ctrl):
        """Load all ASP rules into a given Control instance.

        Raises:
            RuntimeError: If no .lp files found
        """
        if self._program_blob is not None:
            ctrl.add("base", [], self._program_blob)
            return

        lp_files = list(self.rules_dir.glob("*.lp"))

        if not lp_files:
            raise RuntimeError(f"No .lp files found in {self.rules_dir}")

        for rule_file in lp_files:
            with open(rule_file, 'r') as f:
                ctrl.add("base", [], f.read())
    
    def check(
        self,
        trajectory: np.ndarray,
        other_vehicles: Optional[List[np.ndarray]] = None,
        map_data: Optional[Dict] = None,
        traffic_lights: Optional[Dict] = None,
        zones: Optional[List[Dict]] = None,
        pedestrians: Optional[List[Dict]] = None,
        weather: Optional[Dict] = None,
        ego_intent: Optional[Dict] = None,
        extras: Optional[Dict] = None,
        include_internal: bool = False,
        include_constraints: bool = False,
    ) -> tuple:
        """Check trajectory against all 63 traffic rules.
        
        Args:
            trajectory: Vehicle trajectory, shape (T, 2), T <= 6, meters
            other_vehicles: List of other vehicle trajectories, each (T, 2)
            map_data: Map data with lanes, intersections, etc.
            traffic_lights: Traffic light states per intersection
            zones: List of zone dicts (school, construction, variable speed)
            pedestrians: List of pedestrian position dicts per timestep
            weather: Weather/environmental context dict
            ego_intent: Ego vehicle intent (turns, maneuvers, lane changes)
            extras: Additional ASP facts (custom per-scenario data)
                
        Returns:
            Tuple of (is_safe, violations) where violations is a list of dicts:
                {'rule': str, 'vehicle': str, 'time': int} for violation/3
                {'rule': str, 'vehicle': str, 'vehicle2': str, 'time': int} for violation/4
            With include_internal=True, solver-internal arbitration atoms
            are appended too (never count toward unsafe; for threshold
            tuning and the P-6 priority pipeline):
                ttc_band -> {'rule': 'ttc_band', 'band': str, type 'ttc'}
                threat_level -> {'rule': 'threat_level', 'level': str,
                                 type 'threat'}
                comfort_score -> {'rule': 'comfort_score', 'score': int,
                                  type 'comfort'}
                future_violation -> {'rule': ..., 'time': int, type 'future'}
                any_violation -> {'rule': 'any_violation', type 'any'}

            With include_constraints=True, ASP-emitted correction bounds
            (P1: constraint_v_max/v_min/x_bounds/y_bounds/a_max/
            lane_center, see rules/helpers.lp #show block) are collected
            into a constraints dict and returned as a third tuple
            element (default False keeps the legacy 2-tuple):
                constraints_dict = {
                    'v_max': [(t, limit_cm_per_step), ...],
                    'v_min': [(t, limit_cm_per_step), ...],
                    'x_bounds': [(t, min_cm, max_cm), ...],
                    'y_bounds': [(t, min_cm, max_cm), ...],
                    'a_max': [(t, limit_cm_per_step2), ...],
                    'lane_center': [(t, y_cm), ...],
                }
            Constraints never count toward unsafe (only
            violation/hazard/insufficient do) and never appear in the
            violations list.
        """
        if trajectory.ndim != 2 or trajectory.shape[1] != 2:
            raise ValueError(
                f"Expected trajectory shape (T, 2), got {trajectory.shape}"
            )
        
        if trajectory.shape[0] > 6:
            raise ValueError(
                f"Maximum 6 timesteps allowed, got {trajectory.shape[0]}"
            )
        
        # GAP-10 + conflict-aware culling (P-8/PR-3): pre-filter
        # tracked agents outside the cull radius (default 150 m at T=0)
        # so High-impact rules never ground the full VxT product for
        # distant agents. Distance-only culling drops fast closers a
        # 160 m agent closing 20 m/s covers 120 m over the horizon --
        # so an agent is also kept when its endpoint-implied closing
        # rate would bring it inside the radius within 6 steps.
        # Malformed/empty entries are skipped (never crash the check).
        if other_vehicles:
            other_vehicles = self._cull_agents(trajectory, other_vehicles)

        # Create a fresh Control instance for each check; fact builders
        # write into a batcher flushed once before grounding (P-8).
        ctrl = clingo.Control(["--warn=none"])
        self._load_rules_to(ctrl)
        facts = _FactBatcher(ctrl)

        # Add ego vehicle trajectory
        self._add_trajectory_to(facts, "ego", trajectory)

        # Add other vehicles
        if other_vehicles:
            for i, veh in enumerate(other_vehicles):
                self._add_trajectory_to(facts, f"vehicle_{i}", veh)

        # Derived proximity facts for R3/R4/R13/R15 (GAP-5).
        # Demand gates (same pattern as want_conflict below): skip fact
        # families no rule can consume without their companion inputs,
        # so the V*T grounding product stays minimal.
        _extra_text = ""
        if extras:
            try:
                _extra_text = " ".join(extras.get('facts', []))
            except (AttributeError, TypeError):
                pass
        self._add_proximity_to(
            facts, trajectory, other_vehicles,
            want_longitudinal=bool(map_data and map_data.get('lanes')),
            want_approaching='emergency_vehicle' in _extra_text,
            want_rolling=bool((ego_intent or {}).get('lead_vehicle')))

        # Derived conflict facts for R6.3/R7.2/R8.1/R8.2/R10.3 (IF-2):
        # oncoming agents, time-to-conflict, maneuver durations,
        # sightlines, lane membership aliases.
        self._add_derived_facts_to(
            facts, trajectory, other_vehicles, map_data, ego_intent, extras)

        # Add map data
        if map_data:
            self._add_map_data_to(facts, map_data)

        # Python-side lane membership, any segment orientation
        # (MC-1 option C). Complements the axis-aligned ASP derivation.
        if map_data and map_data.get('lanes'):
            vehtraj = {'ego': trajectory}
            if other_vehicles:
                for i, veh in enumerate(other_vehicles):
                    vehtraj[f"vehicle_{i}"] = veh
            self._add_lane_membership_to(facts, vehtraj, map_data)

        # Add traffic lights
        if traffic_lights:
            self._add_traffic_lights_to(facts, traffic_lights)

        # Add zones (R1.3, R11.4, R9.2)
        if zones:
            self._add_zones_to(facts, zones)

        # Add pedestrians/VRUs (R6.5, R9.1, R1.5)
        if pedestrians:
            self._add_pedestrians_to(facts, pedestrians)

        # Add weather/environmental context (R14.x)
        if weather:
            self._add_weather_to(facts, weather)

        # Add ego intent (turns, maneuvers)
        if ego_intent:
            self._add_ego_intent_to(facts, ego_intent)

        # Add extras (custom facts)
        if extras:
            self._add_extras_to(facts, extras)
            # ASP lane-geometry fallback (perf): the axis-aligned
            # in_lane/3 clauses ground over every segment x position.
            # Array trajectories are covered by exact in_lane_fact/3,
            # so assert the fallback flag only when extras carry
            # hand-written pos/4 facts that need it.
            if 'pos(' in _extra_text:
                facts.add("base", [], "use_asp_in_lane.")

        # Single parse of all buffered facts, then ground.
        facts.flush()
        ctrl.ground([("base", [])])
        
        # Solve and collect violations (shown atoms only -- see the
        # #show block in helpers.lp; keeps the per-solve Python scan
        # to dozens of atoms instead of thousands of facts).
        violations = []
        constraints: List[Dict] = []

        def on_model(model):
            for atom in model.symbols(shown=True):
                entry = TrajectoryChecker._shown_atom_to_dict(
                    atom, include_internal, include_constraints)
                if entry is not None:
                    if entry.get("type") == "constraint":
                        constraints.append(entry)
                    else:
                        violations.append(entry)

        ctrl.solve(on_model=on_model)

        # Warnings (R2.5 advisory) and internal arbitration atoms are
        # reported but never count toward unsafe; only
        # violation/hazard/insufficient atoms do.
        hard = [v for v in violations
                if v.get("type", "violation") in
                ("violation", "hazard", "insufficient")]
        if include_constraints:
            return len(hard) == 0, violations, TrajectoryChecker._group_constraints(constraints)
        return len(hard) == 0, violations
    
    # === Fact addition methods ===

    @staticmethod
    def _shown_atom_to_dict(atom, include_internal: bool,
                            include_constraints: bool = False) -> Optional[Dict]:
        """Convert one shown answer-set atom to a finding dict (RQ-4).

        Returns None for atoms check() does not collect (internal
        arbitration atoms unless include_internal=True; constraint
        atoms unless include_constraints=True). Dict shapes
        match the legacy inline collector exactly, including the
        arity-safe fallbacks.
        """
        args = atom.arguments
        name = atom.name

        def _num(arg, default=-1):
            return arg.number if hasattr(arg, "number") else default

        if name in ("constraint_v_max", "constraint_v_min",
                    "constraint_a_max"):
            if not include_constraints:
                return None
            if len(args) == 3:
                return {"rule": name, "vehicle": str(args[0]),
                        "time": args[1].number, "limit": args[2].number,
                        "type": "constraint"}
            return None
        if name in ("constraint_x_bounds", "constraint_y_bounds"):
            if not include_constraints:
                return None
            if len(args) == 4:
                return {"rule": name, "vehicle": str(args[0]),
                        "time": args[1].number, "min": args[2].number,
                        "max": args[3].number, "type": "constraint"}
            return None
        if name == "constraint_lane_center":
            if not include_constraints:
                return None
            if len(args) == 3:
                return {"rule": name, "vehicle": str(args[0]),
                        "time": args[1].number, "y": args[2].number,
                        "type": "constraint"}
            return None

        if name == "violation":
            if len(args) == 3:
                return {"rule": str(args[0]), "vehicle": str(args[1]),
                        "time": args[2].number}
            if len(args) == 4:
                return {"rule": str(args[0]), "vehicle": str(args[1]),
                        "vehicle2": str(args[2]), "time": args[3].number}
            return {"rule": str(args[0]) if args else "violation",
                    "vehicle": str(args[1]) if len(args) > 1 else "ego",
                    "time": _num(args[-1]) if args else -1,
                    "type": "violation", "arity": len(args)}
        if name == "hazard":
            if len(args) == 3:
                return {"rule": str(args[0]), "vehicle": str(args[1]),
                        "time": args[2].number, "type": "hazard"}
            if len(args) == 4:
                return {"rule": str(args[0]), "vehicle": str(args[1]),
                        "vehicle2": str(args[2]), "time": args[3].number,
                        "type": "hazard"}
            return {"rule": str(args[0]) if args else "hazard",
                    "vehicle": str(args[1]) if len(args) > 1 else "ego",
                    "time": _num(args[-1]) if args else -1,
                    "type": "hazard", "arity": len(args)}
        if name == "warning":
            if len(args) == 3:
                return {"rule": str(args[0]), "vehicle": str(args[1]),
                        "time": args[2].number, "type": "warning"}
            return {"rule": str(args[0]) if args else "warning",
                    "vehicle": str(args[1]) if len(args) > 1 else "ego",
                    "time": _num(args[-1]) if args else -1,
                    "type": "warning", "arity": len(args)}
        if name == "insufficient":
            if len(args) == 4:
                return {"rule": str(args[0]), "vehicle": str(args[1]),
                        "hazard": str(args[2]), "time": args[3].number,
                        "type": "insufficient"}
            return {"rule": str(args[0]) if args else "insufficient",
                    "vehicle": str(args[1]) if len(args) > 1 else "ego",
                    "hazard": str(args[2]) if len(args) > 2 else "",
                    "time": _num(args[-1]) if args else -1,
                    "type": "insufficient", "arity": len(args)}
        if not include_internal:
            return None
        if name == "ttc_band" and len(args) == 4:
            return {"rule": "ttc_band", "vehicle": str(args[0]),
                    "vehicle2": str(args[1]), "time": args[2].number,
                    "band": str(args[3]), "type": "ttc"}
        if name == "threat_level" and len(args) == 3:
            return {"rule": "threat_level", "vehicle": str(args[0]),
                    "time": args[1].number, "level": str(args[2]),
                    "type": "threat"}
        if name == "comfort_score" and len(args) == 2:
            return {"rule": "comfort_score", "vehicle": str(args[0]),
                    "score": args[1].number, "type": "comfort"}
        if name == "future_violation" and len(args) == 2:
            return {"rule": "future_violation", "vehicle": str(args[0]),
                    "time": args[1].number, "type": "future"}
        if name == "any_violation":
            return {"rule": "any_violation", "vehicle": "ego",
                    "type": "any"}
        return None

    @staticmethod
    def _group_constraints(entries: List[Dict]) -> Dict[str, list]:
        """Group flat constraint dicts by bound family (P1).

        Returns {'v_max': [(t, limit), ...], 'v_min': [...],
        'x_bounds': [(t, min, max), ...], 'y_bounds': [...],
        'x_bounds_range': [(t0, t1, min, max), ...],
        'y_bounds_range': [(t0, t1, min, max), ...],
        'a_max': [...], 'lane_center': [(t, y), ...]} with keys
        always present (possibly empty). Ego-only: non-ego vehicle
        entries are dropped (no rule emits them today).
        """
        grouped: Dict[str, list] = {
            'v_max': [], 'v_min': [], 'x_bounds': [], 'y_bounds': [],
            'x_bounds_range': [], 'y_bounds_range': [],
            'a_max': [], 'lane_center': [],
        }
        for e in entries:
            if e.get("vehicle") != "ego":
                continue
            rule = e.get("rule", "")
            t = e.get("time", -1)
            if rule in ("constraint_v_max", "constraint_v_min",
                        "constraint_a_max"):
                key = rule[len("constraint_"):]
                grouped[key].append((t, e.get("limit", 0)))
            elif rule in ("constraint_x_bounds", "constraint_y_bounds"):
                key = rule[len("constraint_"):]
                grouped[key].append((t, e.get("min", 0), e.get("max", 0)))
            elif rule in ("constraint_x_bounds_range",
                          "constraint_y_bounds_range"):
                key = rule[len("constraint_"):]
                grouped[key].append((e.get("t0", 0), e.get("t1", 0),
                                    e.get("min", 0), e.get("max", 0)))
            elif rule == "constraint_lane_center":
                grouped['lane_center'].append((t, e.get("y", 0)))
        return grouped

    def _cull_agents(
        self,
        trajectory: np.ndarray,
        other_vehicles: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Keep agents within range or closing into range (RQ-4 split)."""
        import math
        kept = []
        for veh in other_vehicles:
            try:
                veh_arr = np.asarray(veh, dtype=float)
                if veh_arr.ndim != 2 or veh_arr.shape[1] != 2 \
                        or veh_arr.shape[0] == 0:
                    continue
                n = min(len(trajectory), len(veh_arr))
                dx = float(trajectory[0, 0] - veh_arr[0, 0])
                dy = float(trajectory[0, 1] - veh_arr[0, 1])
                d0 = math.hypot(dx, dy)
                if d0 <= self.max_agent_range_m:
                    kept.append(veh)
                    continue
                ex = float(trajectory[n - 1, 0] - veh_arr[n - 1, 0])
                ey = float(trajectory[n - 1, 1] - veh_arr[n - 1, 1])
                d_end = math.hypot(ex, ey)
                closing = max(0.0, (d0 - d_end) / max(1, n - 1))
                if closing > 0 and d0 - closing * 6.0 \
                        <= self.max_agent_range_m:
                    kept.append(veh)
            except (IndexError, ValueError, TypeError):
                continue
        return kept

    def _add_trajectory_to(self, ctrl, vehicle_id: str, trajectory: np.ndarray):
        """Add trajectory as pos/4 and vehicle/1 facts."""
        for t, (x, y) in enumerate(trajectory):
            x_scaled = int(round(x * SCALE_FACTOR))
            y_scaled = int(round(y * SCALE_FACTOR))
            ctrl.add("base", [],
                f"pos({vehicle_id}, {t}, {x_scaled}, {y_scaled}).")
        ctrl.add("base", [], f"vehicle({vehicle_id}).")
    
    def _add_map_data_to(self, ctrl, map_data: Dict):
        """Add lane geometry and metadata as ASP facts.
        
        Expected map_data structure:
        {
            'lanes': {
                'lane_id': [
                    {'x1': float, 'y1': float, 'x2': float, 'y2': float}
                ]
            },
            'intersections': {
                'intersection_id': {
                    'x': float, 'y': float, 'radius': float
                }
            },
            'lane_width': {'lane_id': float},           # optional
            'lane_direction': {'lane_id': [dx, dy]},    # optional
            'road_type': {'lane_id': str},               # optional
            'speed_limit': {'lane_id': float},           # optional (m/s)
            'min_speed_limit': {'lane_id': float},       # optional
            'lane_boundary': [[l1, l2, type]],           # optional
            'lane_restriction': {'lane_id': [type, min_occ, hours]}, # optional
            'lane_type': {'lane_id': str},               # optional
            'no_u_turn': ['lane_id'],                    # optional
            'no_turn_on_red': {'intersection_id': [dir]},# optional
            'controls_approach': {'intersection_id': 'lane_id'}, # optional
            'crosswalks': [{'id': str, 'x1': float, ...}], # optional
            'sidewalks': [{'id': str, 'x1': float, ...}],  # optional
            'fixed_obstacles': [{'id': str, 'x1': float, ...}], # optional
            'no_passing_zones': [{'id': str, 'x1': float, ...}], # optional
            'merge_lanes': {'lane_id': {'end_point': float}}, # optional
        }
        """
        # Lane geometry
        for lane_id, segments in map_data.get('lanes', {}).items():
            for seg in segments:
                x1 = int(round(seg['x1'] * SCALE_FACTOR))
                y1 = int(round(seg['y1'] * SCALE_FACTOR))
                x2 = int(round(seg['x2'] * SCALE_FACTOR))
                y2 = int(round(seg['y2'] * SCALE_FACTOR))
                ctrl.add("base", [],
                    f"lane({lane_id}, {x1}, {y1}, {x2}, {y2}).")
        
        # Intersections
        for iid, state in map_data.get('intersections', {}).items():
            x = int(round(state['x'] * SCALE_FACTOR))
            y = int(round(state['y'] * SCALE_FACTOR))
            r = int(round(state['radius'] * SCALE_FACTOR))
            ctrl.add("base", [], f"intersection({iid}, {x}, {y}, {r}).")
        
        # Lane widths
        for lid, w in map_data.get('lane_width', {}).items():
            w_scaled = int(round(w * SCALE_FACTOR))
            ctrl.add("base", [], f"lane_width({lid}, {w_scaled}).")
        
        # Lane directions
        for lid, d in map_data.get('lane_direction', {}).items():
            dx = int(round(d[0] * SCALE_FACTOR))
            dy = int(round(d[1] * SCALE_FACTOR))
            ctrl.add("base", [], f"lane_direction({lid}, {dx}, {dy}).")
        
        # Road types
        for lid, rt in map_data.get('road_type', {}).items():
            ctrl.add("base", [], f"road_type({lid}, {rt}).")
        
        # Speed limits: m/s -> cm/timestep (linear; rules square at
        # comparison time since speed/3 is squared speed).
        for lid, lim in map_data.get('speed_limit', {}).items():
            lim_scaled = int(round(lim * SCALE_FACTOR))
            ctrl.add("base", [], f"speed_limit({lid}, {lim_scaled}).")
        
        # Minimum speed limits (m/s -> cm/timestep, linear)
        for lid, lim in map_data.get('min_speed_limit', {}).items():
            lim_scaled = int(round(lim * SCALE_FACTOR))
            ctrl.add("base", [], f"min_speed_limit({lid}, {lim_scaled}).")
        
        # Lane boundaries (solid/dashed)
        for boundary in map_data.get('lane_boundary', []):
            ctrl.add("base", [],
                f"lane_boundary({boundary[0]}, {boundary[1]}, {boundary[2]}).")
        
        # Lane restrictions (HOV, bus-only)
        for lid, info in map_data.get('lane_restriction', {}).items():
            ctrl.add("base", [],
                f"lane_restriction({lid}, {info[0]}, {info[1]}, {info[2]}).")
        
        # Lane types
        for lid, lt in map_data.get('lane_type', {}).items():
            ctrl.add("base", [], f"lane_type({lid}, {lt}).")
        
        # No U-turn lanes
        for lid in map_data.get('no_u_turn', []):
            ctrl.add("base", [], f"no_u_turn({lid}).")
        
        # No turn on red
        for iid, dirs in map_data.get('no_turn_on_red', {}).items():
            for d in dirs:
                ctrl.add("base", [], f"no_turn_on_red({iid}, {d}).")
        
        # Controls approach
        for iid, lid in map_data.get('controls_approach', {}).items():
            ctrl.add("base", [], f"controls_approach({iid}, {lid}).")
        
        # Crosswalks
        for cw in map_data.get('crosswalks', []):
            x1 = int(round(cw['x1'] * SCALE_FACTOR))
            y1 = int(round(cw['y1'] * SCALE_FACTOR))
            x2 = int(round(cw['x2'] * SCALE_FACTOR))
            y2 = int(round(cw['y2'] * SCALE_FACTOR))
            ctrl.add("base", [],
                f"crosswalk({cw['id']}, {x1}, {y1}, {x2}, {y2}).")
        
        # Sidewalks
        for sw in map_data.get('sidewalks', []):
            x1 = int(round(sw['x1'] * SCALE_FACTOR))
            y1 = int(round(sw['y1'] * SCALE_FACTOR))
            x2 = int(round(sw['x2'] * SCALE_FACTOR))
            y2 = int(round(sw['y2'] * SCALE_FACTOR))
            ctrl.add("base", [],
                f"sidewalk({sw['id']}, {x1}, {y1}, {x2}, {y2}).")
        
        # Fixed obstacles
        for obs in map_data.get('fixed_obstacles', []):
            x1 = int(round(obs['x1'] * SCALE_FACTOR))
            y1 = int(round(obs['y1'] * SCALE_FACTOR))
            x2 = int(round(obs['x2'] * SCALE_FACTOR))
            y2 = int(round(obs['y2'] * SCALE_FACTOR))
            ctrl.add("base", [],
                f"fixed_obstacle({obs['id']}, {x1}, {y1}, {x2}, {y2}).")
        
        # No-passing zones
        for z in map_data.get('no_passing_zones', []):
            x1 = int(round(z['x1'] * SCALE_FACTOR))
            y1 = int(round(z['y1'] * SCALE_FACTOR))
            x2 = int(round(z['x2'] * SCALE_FACTOR))
            y2 = int(round(z['y2'] * SCALE_FACTOR))
            ctrl.add("base", [],
                f"no_passing_zone({z['id']}, {x1}, {y1}, {x2}, {y2}).")
        
        # Merge lanes
        for lid, info in map_data.get('merge_lanes', {}).items():
            ep = int(round(info['end_point'] * SCALE_FACTOR))
            ctrl.add("base", [], f"merge_lane({lid}, {ep}).")

        # Conflict points for multi-way priority (R15.1). Radius comes
        # from helpers (conflict_zone/4 defaults to 1000 cm).
        for cp in map_data.get('conflict_points', []):
            x = int(round(cp['x'] * SCALE_FACTOR))
            y = int(round(cp['y'] * SCALE_FACTOR))
            ctrl.add("base", [],
                f"conflict_point({cp['id']}, {x}, {y}).")

        # Oncoming lanes per intersection for left-turn checks (R7.2).
        for lid, iid in map_data.get('oncoming_lanes', {}).items():
            ctrl.add("base", [], f"oncoming_lane({lid}, {iid}).")
    
    def _add_traffic_lights_to(self, ctrl, lights: Dict):
        """Add traffic light states as ASP facts.

        Each entry supports 'time' (single timestep) and/or 'times'
        (list of timesteps) so a phase can span the whole horizon.
        If neither is given, no light_state fact is emitted (no
        knowledge) — only intersection geometry, if provided. We
        deliberately do NOT default to [0]: inventing a red at T=0
        causes both false positives and false negatives.
        """
        for intersection_id, state in lights.items():
            times = list(state.get('times', []))
            if 'time' in state:
                times.append(state['time'])
            for t in times:
                ctrl.add("base", [],
                    f"light_state({intersection_id}, {state['color']}, {t}).")
            if 'x' in state and 'y' in state and 'radius' in state:
                x = int(round(state['x'] * SCALE_FACTOR))
                y = int(round(state['y'] * SCALE_FACTOR))
                r = int(round(state['radius'] * SCALE_FACTOR))
                ctrl.add("base", [],
                    f"intersection({intersection_id}, {x}, {y}, {r}).")
            # Directional arrow states (one per listed timestep)
            if 'arrows' in state:
                for direction, arrow_color in state['arrows'].items():
                    for t in times:
                        ctrl.add("base", [],
                            f"light_state({intersection_id}, arrow_{arrow_color}, "
                            f"{direction}, {t}).")
    
    def _add_zones_to(self, ctrl, zones: List[Dict]):
        """Add zone facts for school/construction/variable speed zones.
        
        Each zone dict:
        {
            'id': str,
            'x1': float, 'y1': float, 'x2': float, 'y2': float,
            'type': 'school'|'construction'|'variable',
            'active': bool,
            'speed_limit': float (m/s),  # optional
            'active_times': list,         # optional explicit timesteps;
                                          # honored when present, else all 0..5
            'crossing_guard': str,        # optional signal value
            'crossing_guard_times': list, # optional explicit timesteps
            'workers': [{'id': str, 'x': float, 'y': float,
                         't': int, 'times': list}],  # time keys optional
        }
        Temporal rule: explicit times are honored; a worker/guard with no
        time keys is assumed present at all T=0..5 (conservative) rather
        than T=0 only.
        """
        for zone in zones:
            zid = zone['id']
            x1 = int(round(zone['x1'] * SCALE_FACTOR))
            y1 = int(round(zone['y1'] * SCALE_FACTOR))
            x2 = int(round(zone['x2'] * SCALE_FACTOR))
            y2 = int(round(zone['y2'] * SCALE_FACTOR))
            ztype = zone['type']
            ctrl.add("base", [],
                f"zone({zid}, {x1}, {y1}, {x2}, {y2}, {ztype}).")
            if zone.get('active', False):
                active_times = zone.get('active_times', list(range(6)))
                for t in active_times:
                    ctrl.add("base", [], f"zone_active({zid}, {t}).")
            if 'speed_limit' in zone:
                sl = int(round(zone['speed_limit'] * SCALE_FACTOR))
                ctrl.add("base", [], f"zone_speed_limit({zid}, {sl}).")
            # Crossing guard signal
            if 'crossing_guard' in zone:
                guard_times = zone.get('crossing_guard_times', list(range(6)))
                for gt in guard_times:
                    ctrl.add("base", [],
                        f"crossing_guard_signal({zid}, {gt}, {zone['crossing_guard']}).")
            # Construction workers
            for worker in zone.get('workers', []):
                wx = int(round(worker['x'] * SCALE_FACTOR))
                wy = int(round(worker['y'] * SCALE_FACTOR))
                if 'times' in worker:
                    wtimes = list(worker['times'])
                elif 't' in worker:
                    wtimes = [worker['t']]
                else:
                    wtimes = list(range(6))
                for wt in wtimes:
                    ctrl.add("base", [],
                        f"worker({worker['id']}, {wt}, {wx}, {wy}).")
            # Lane shifts for construction
            for shift in zone.get('lane_shifts', []):
                ctrl.add("base", [],
                    f"lane_shift({zid}, {shift['lane']}, {shift['path']}).")
    
    def _add_pedestrians_to(self, ctrl, pedestrians: List[Dict]):
        """Add pedestrian/VRU facts.
        
        Each pedestrian dict:
        {
            'id': str,
            'positions': {timestep: (x, y)},  # positions per timestep in meters
            'type': 'pedestrian'|'cyclist',    # optional, default pedestrian
        }
        """
        for ped in pedestrians:
            pid = ped['id']
            ptype = ped.get('type', 'pedestrian')
            for t, (x, y) in ped.get('positions', {}).items():
                x_scaled = int(round(x * SCALE_FACTOR))
                y_scaled = int(round(y * SCALE_FACTOR))
                ctrl.add("base", [],
                    f"pedestrian({pid}, {t}, {x_scaled}, {y_scaled}).")
                ctrl.add("base", [],
                    f"vru({pid}, {t}, {x_scaled}, {y_scaled}, {ptype}).")
    
    def _add_weather_to(self, ctrl, weather: Dict):
        """Add weather/environmental context facts.
        
        Weather dict:
        {
            'road_condition': {'timestep': 'wet'|'dry'},
            'time_of_day': {'timestep': 'night'|'day'},
            'perception_confidence': {'timestep': 'high'|'medium'|'low'|'severe'},
            'weather_severity': {'timestep': 'mild'|'moderate'|'severe'},
            'wet_speed_factor': int (percentage, e.g. 80 = 80%),
            'headlight_range': float (meters),
        }
        """
        for t, cond in weather.get('road_condition', {}).items():
            ctrl.add("base", [], f"road_condition({t}, {cond}).")
        
        for t, tod in weather.get('time_of_day', {}).items():
            ctrl.add("base", [], f"time_of_day({t}, {tod}).")
        
        for t, level in weather.get('perception_confidence', {}).items():
            ctrl.add("base", [], f"perception_confidence({t}, {level}).")
        
        for t, severity in weather.get('weather_severity', {}).items():
            ctrl.add("base", [], f"weather_severity({t}, {severity}).")
        
        if 'wet_speed_factor' in weather:
            ctrl.add("base", [],
                f"wet_speed_factor({weather['wet_speed_factor']}).")
        
        if 'headlight_range' in weather:
            hr = int(round(weather['headlight_range'] * SCALE_FACTOR))
            ctrl.add("base", [], f"headlight_range(ego, {hr}).")
        
        # Confidence-to-speed-cap lookup (m/s -> cm/timestep, linear)
        caps = weather.get('confidence_speed_caps', {})
        for level, cap in caps.items():
            cap_scaled = int(round(cap * SCALE_FACTOR))
            ctrl.add("base", [], f"confidence_speed_cap({level}, {cap_scaled}).")
        
        # Threshold adjustments
        for level, rules in weather.get('threshold_adjustments', {}).items():
            for rule_family, factor in rules.items():
                ctrl.add("base", [],
                    f"threshold_adjustment({level}, {rule_family}, {factor}).")
    
    def _add_ego_intent_to(self, ctrl, intent: Dict):
        """Add ego vehicle intent facts (turns, maneuvers).
        
        Intent dict:
        {
            'turn': 'left'|'right'|None,
            'maneuver': 'overtake'|'lane_change'|'u_turn'|'stopped'|None,
            'maneuver_direction': 'left'|'right',
            'maneuver_time': int,
            'lead_vehicle': {'timestep': vehicle_id},
            'queue_members': [vehicle_id, ...],
            'predicted_positions': {'vehicle_id': {t: (x,y)}},
            'signal_active': {'direction': bool},
            'platoon': {'id': str, 'members': [vehicle_id, ...], 'comm_active': bool},
            'blind_spots': {'side': {'x1': float, ...}},
            'occupancy': int,            # ego passenger count (R2.4 HOV)
            'current_time': str,         # time-of-day label (R2.4 hours)
        }
        """
        if intent.get('turn'):
            ctrl.add("base", [],
                f"planned_turn(ego, {intent['turn']}).")
        
        if intent.get('maneuver'):
            mt = intent.get('maneuver_time', 0)
            ctrl.add("base", [],
                f"planned_maneuver(ego, {intent['maneuver']}, {mt}).")
        
        if intent.get('maneuver_direction'):
            mt = intent.get('maneuver_time', 0)
            ctrl.add("base", [],
                f"maneuver_direction(ego, {mt}, {intent['maneuver_direction']}).")
        
        for t, vid in intent.get('lead_vehicle', {}).items():
            ctrl.add("base", [], f"lead_vehicle(ego, {vid}, {t}).")
        
        for vid in intent.get('queue_members', []):
            ctrl.add("base", [], f"queue_member({vid}).")
        
        for vid, positions in intent.get('predicted_positions', {}).items():
            for t, (x, y) in positions.items():
                x_scaled = int(round(x * SCALE_FACTOR))
                y_scaled = int(round(y * SCALE_FACTOR))
                ctrl.add("base", [],
                    f"predicted_pos({vid}, {t}, {x_scaled}, {y_scaled}).")
        
        for direction, active in intent.get('signal_active', {}).items():
            if active:
                for t in range(6):
                    ctrl.add("base", [],
                        f"signal_active(ego, {direction}, {t}).")

        # Overtake side per timestep: {'timestep': 'left'|'right'}
        for t, side in intent.get('overtake_side', {}).items():
            ctrl.add("base", [], f"overtake_side(ego, {t}, {side}).")

        # Lead-vehicle turn exception for R8.3: [vehicle_id, ...] turning left
        for vid in intent.get('lead_turning_left', []):
            ctrl.add("base", [], f"lead_vehicle_turning({vid}, left).")
        
        # Platoon
        platoon = intent.get('platoon', {})
        if platoon:
            pid = platoon['id']
            ctrl.add("base", [], f"platoon_member(ego, {pid}).")
            for member in platoon.get('members', []):
                ctrl.add("base", [], f"platoon_member({member}, {pid}).")
            if platoon.get('comm_active', False):
                lead = platoon.get('lead_vehicle')
                if lead:
                    ctrl.add("base", [],
                        f"comm_link_active(ego, {lead}, 0).")
        
        # Blind spot zones
        for side, zone in intent.get('blind_spots', {}).items():
            x1 = int(round(zone['x1'] * SCALE_FACTOR))
            y1 = int(round(zone['y1'] * SCALE_FACTOR))
            x2 = int(round(zone['x2'] * SCALE_FACTOR))
            y2 = int(round(zone['y2'] * SCALE_FACTOR))
            ctrl.add("base", [],
                f"blind_spot_zone(ego, {side}, {x1}, {y1}, {x2}, {y2}).")

        # HOV occupancy + time-of-day for restricted lanes (R2.4).
        if 'occupancy' in intent:
            ctrl.add("base", [], f"occupancy(ego, {int(intent['occupancy'])}).")
        if intent.get('current_time') is not None:
            ctrl.add("base", [], f"current_time({intent['current_time']}).")

    def _add_proximity_to(self, ctrl, ego_traj: np.ndarray,
                          other_vehicles: Optional[List[np.ndarray]],
                          want_longitudinal: bool = True,
                          want_approaching: bool = True,
                          want_rolling: bool = True):
        """Derive gap/closing-rate facts for R3/R4/R13/R15 rules (P-4).

        Emits per other vehicle V and timestep T (integer cm):
          gap_dist(ego, V, T, D)        Euclidean ego->V distance
          following_gap(V, ego, T, D)    same distance, follower perspective
          longitudinal_gap(ego, V, T, D) SIGNED along-track separation:
              projection of (V - ego) onto ego's heading unit vector
              (+ = ahead, - = behind). Falls back to Euclidean D when
              ego is stationary (no along-track direction exists).
          closing_rate(ego, V, T, CR)    max(0, instantaneous closing
              speed): relative velocity (V - ego, central differences)
              projected onto the ego->V line of sight, cm/timestep.
              Smoother than first-differenced distances (no +-1 cm
              quantization stair); 0 when opening or coincident.

        Headings/velocities use central differences with clamped ends
        (same convention as _add_derived_facts_to).
        """
        import math
        if not other_vehicles:
            return

        def _unit_heading(traj, t):
            n = len(traj)
            t2 = min(t + 1, n - 1)
            t0 = max(t - 1, 0)
            hx = float(traj[t2, 0] - traj[t0, 0])
            hy = float(traj[t2, 1] - traj[t0, 1])
            elen = math.hypot(hx, hy)
            if elen < 1e-9:
                return None
            return hx / elen, hy / elen

        def _velocity(traj, t):
            n = len(traj)
            t2 = min(t + 1, n - 1)
            t0 = max(t - 1, 0)
            dt = t2 - t0
            if dt == 0:
                return 0.0, 0.0
            return (float(traj[t2, 0] - traj[t0, 0]) / dt,
                    float(traj[t2, 1] - traj[t0, 1]) / dt)

        for i, veh in enumerate(other_vehicles):
            vid = f"vehicle_{i}"
            n = min(len(ego_traj), len(veh))
            for t in range(n):
                ex, ey = float(ego_traj[t, 0]), float(ego_traj[t, 1])
                vx, vy = float(veh[t, 0]), float(veh[t, 1])
                dx, dy = vx - ex, vy - ey
                dist = math.hypot(dx, dy)
                d = int(round(dist * SCALE_FACTOR))
                ctrl.add("base", [], f"gap_dist(ego, {vid}, {t}, {d}).")
                ctrl.add("base", [], f"following_gap({vid}, ego, {t}, {d}).")

                # Longitudinal (signed along-track) separation.
                uh = _unit_heading(ego_traj, t)
                if uh is None:
                    lon = d  # stationary ego: no along-track axis
                else:
                    lon = int(round((dx * uh[0] + dy * uh[1]) * SCALE_FACTOR))
                if want_longitudinal:
                    ctrl.add("base", [],
                             f"longitudinal_gap(ego, {vid}, {t}, {lon}).")
                    ctrl.add("base", [],
                             f"has_longitudinal_gap(ego, {vid}, {t}).")

                # Instantaneous closing speed via velocity projection.
                if dist < 1e-9:
                    cr = 0
                else:
                    evx, evy = _velocity(ego_traj, t)
                    vvx, vvy = _velocity(veh, t)
                    ux, uy = dx / dist, dy / dist
                    cr = int(round(max(
                        0.0, -((vvx - evx) * ux + (vvy - evy) * uy)
                    ) * SCALE_FACTOR))
                ctrl.add("base", [], f"closing_rate(ego, {vid}, {t}, {cr}).")

                # Converging agent within 50 m (R6.4 emergency yield).
                # Distinct from the zone-oriented approaching_zone/3 in
                # helpers.lp (different argument roles by design; the
                # endpoint-oriented approaching(ego, EndPoint, T) used by
                # R15.3 never collides: its second argument is numeric).
                if want_approaching and cr > 0 and d < 5000:
                    ctrl.add("base", [], f"approaching({vid}, ego, {t}).")

        # Rolling-average squared speed per timestep (R15.4 wave
        # dampening). Avg is in speed2 space (cm/step)^2, averaged over
        # the [T-1, T+1] window of available squared speeds. Gated: the
        # only consumer needs lead_vehicle intent.
        if not want_rolling:
            return
        for vid, traj in [("ego", ego_traj)] + [
                (f"vehicle_{i}", veh)
                for i, veh in enumerate(other_vehicles)]:
            n = len(traj)
            if n < 2:
                continue
            s2 = {}
            for t in range(1, n):
                dx = float(traj[t, 0] - traj[t - 1, 0]) * SCALE_FACTOR
                dy = float(traj[t, 1] - traj[t - 1, 1]) * SCALE_FACTOR
                s2[t] = int(round(dx * dx + dy * dy))
            for t in range(n):
                window = [s2[tt] for tt in (t - 1, t, t + 1)
                          if tt in s2]
                if not window:
                    continue
                avg = int(round(sum(window) / len(window)))
                ctrl.add("base", [],
                         f"rolling_avg_speed({vid}, {t}, {avg}).")
    
    def _add_derived_facts_to(self, ctrl, ego_traj: np.ndarray,
                                other_vehicles: Optional[List[np.ndarray]],
                                map_data: Optional[Dict],
                                ego_intent: Optional[Dict],
                                extras: Optional[Dict] = None):
        """Derive conflict facts for R6.3/R7.2/R8.1/R8.2/R10.3 (IF-2).

        Units: TTC in deciseconds (10 = 1.0s, matching ttc_*_units);
        durations in deciseconds; sightlines in cm. All integer-safe.

        Emits:
          oncoming(V, ego, T)              head-on converging agent
          time_to_conflict_point(V, T, TTC) gap/closing TTC (capped)
          turn_duration(ego, Est)           intent override or 30 (3s)
          overtake_duration(ego, Est)       intent override or 40 (4s)
          sightline_distance(L, SD)         map sightline or 100000 (open)
          required_sightline(Limit, ReqSD)  Limit*30 (3s of travel)
          lane_of(V, L)                     ASP alias of in_lane at T=0
          oncoming_lane(L, I)               map_data['oncoming_lanes'] (R7.2)
          mainline_vehicle(V)               via _add_merge_facts_to (R15.3)
          merge_gap_sequence(ego, V, Order) geometric slot (R15.3)
        """
        import math
        ego_intent = ego_intent or {}
        map_data = map_data or {}

        # Maneuver durations (deciseconds), intent-overridable.
        turn_dur = int(ego_intent.get('turn_duration', 30))
        overtake_dur = int(ego_intent.get('overtake_duration', 40))
        ctrl.add("base", [], f"turn_duration(ego, {turn_dur}).")
        ctrl.add("base", [], f"overtake_duration(ego, {overtake_dur}).")

        # Sightlines: map-provided per lane, else wide open (1000m).
        seen_lanes = set()
        for lid, sds in (map_data.get('sightline') or {}).items():
            try:
                sd = int(round(float(sds) * SCALE_FACTOR))
            except (TypeError, ValueError):
                continue
            ctrl.add("base", [], f"sightline_distance({lid}, {sd}).")
            seen_lanes.add(lid)
        for segments in (map_data.get('lanes') or {}).keys():
            if segments not in seen_lanes:
                ctrl.add("base", [],
                         f"sightline_distance({segments}, 100000).")
        # Required sightline scales with posted limit (3s travel).
        for lim in (map_data.get('speed_limit') or {}).values():
            try:
                lim_i = int(round(float(lim) * SCALE_FACTOR))
            except (TypeError, ValueError):
                continue
            ctrl.add("base", [],
                     f"required_sightline({lim_i}, {lim_i * 3}).")

        if not other_vehicles:
            return
        # Merge-slot assignment for R15.3 (IF-2): while ego travels a
        # mapped merge lane, order each mainline vehicle geometrically.
        # mainline = array vehicle never inside the merge lane;
        # order 'before' (ego merges ahead of V) vs 'after' from the
        # mean signed along-track offset over the horizon. Heuristic
        # (no signal-phase/zipper state); documented, tested, honest.
        self._add_merge_facts_to(ctrl, ego_traj, other_vehicles, map_data)
        # Oncoming/TTC facts feed R6.3/R7.2/R8.1, all of which require
        # intent (planned_turn/planned_maneuver) to fire. Skip the
        # per-agent per-timestep loop when no intent source can provide
        # them (perf: saves ~15 ms on 20-agent checks).
        extra_text = ""
        if extras:
            try:
                extra_text = " ".join(extras.get('facts', []))
            except (AttributeError, TypeError):
                pass
        want_conflict = (
            any(k in ego_intent for k in (
                'turn', 'maneuver', 'maneuver_direction',
                'turn_duration', 'overtake_duration'))
            or 'planned_turn' in extra_text
            or 'planned_maneuver' in extra_text
        )
        if not want_conflict:
            return
        for i, veh in enumerate(other_vehicles):
            vid = f"vehicle_{i}"
            n = min(len(ego_traj), len(veh))
            gaps = []
            for t in range(n):
                ex, ey = float(ego_traj[t, 0]), float(ego_traj[t, 1])
                vx, vy = float(veh[t, 0]), float(veh[t, 1])
                gaps.append(math.hypot(ex - vx, ey - vy))
            for t in range(n):
                ex, ey = float(ego_traj[t, 0]), float(ego_traj[t, 1])
                vx, vy = float(veh[t, 0]), float(veh[t, 1])
                # Headings from consecutive steps (clamped at horizon end).
                t2 = min(t + 1, n - 1)
                t0 = max(t - 1, 0)
                ehx = float(ego_traj[t2, 0] - ego_traj[t0, 0])
                ehy = float(ego_traj[t2, 1] - ego_traj[t0, 1])
                vhx = float(veh[t2, 0] - veh[t0, 0])
                vhy = float(veh[t2, 1] - veh[t0, 1])
                # Oncoming: opposite headings, closing, laterally close.
                # Closing uses lookback for t>0, lookahead at t=0 (where
                # no history exists yet).
                dot = ehx * vhx + ehy * vhy
                gap = gaps[t]
                elen = math.hypot(ehx, ehy)
                if elen > 1e-9:
                    lat = abs(ehy * (vx - ex) - ehx * (vy - ey)) / elen
                else:
                    lat = gap
                if t > 0:
                    cr_m = gaps[t - 1] - gap
                elif n > 1:
                    cr_m = gaps[0] - gaps[1]
                else:
                    cr_m = 0.0
                closing = cr_m > 1e-9
                if dot < 0 and closing and lat * SCALE_FACTOR <= 500:
                    ctrl.add("base", [], f"oncoming({vid}, ego, {t}).")
                # TTC deciseconds = gap*10/closing_rate; capped when
                # stationary or opening.
                if closing:
                    ttc = int(min(10000, gap * 10.0 / cr_m))
                else:
                    ttc = 10000
                ctrl.add("base", [],
                         f"time_to_conflict_point({vid}, {t}, {ttc}).")

    def _add_merge_facts_to(self, ctrl, ego_traj: np.ndarray,
                              other_vehicles: List[np.ndarray],
                              map_data: Dict):
        """Derive merge-slot facts for R15.3 (IF-2). See call-site note."""
        import math
        merge_lanes = map_data.get('merge_lanes', {})
        lanes = map_data.get('lanes', {})
        widths = map_data.get('lane_width', {})
        if not merge_lanes:
            return

        def _in_lane_at_any(traj, lid):
            if lid not in widths:
                return False
            try:
                half = float(widths[lid]) / 2.0
            except (TypeError, ValueError):
                return False
            for seg in lanes.get(lid, []):
                try:
                    x1, y1 = float(seg['x1']), float(seg['y1'])
                    x2, y2 = float(seg['x2']), float(seg['y2'])
                except (KeyError, TypeError, ValueError):
                    continue
                dx, dy = x2 - x1, y2 - y1
                denom = dx * dx + dy * dy
                for p in traj:
                    try:
                        px, py = float(p[0]), float(p[1])
                    except (TypeError, ValueError, IndexError):
                        continue
                    if denom > 1e-12:
                        s = max(0.0, min(1.0,
                                         ((px - x1) * dx + (py - y1) * dy)
                                         / denom))
                        cx, cy = x1 + s * dx, y1 + s * dy
                    else:
                        cx, cy = x1, y1
                    if math.hypot(px - cx, py - cy) <= half:
                        return True
            return False

        ego_merge = next(
            (lid for lid in merge_lanes if _in_lane_at_any(ego_traj, lid)),
            None)
        if ego_merge is None:
            return

        for i, veh in enumerate(other_vehicles):
            vid = f"vehicle_{i}"
            if _in_lane_at_any(veh, ego_merge):
                continue  # fellow merger, not mainline
            ctrl.add("base", [], f"mainline_vehicle({vid}).")
            # Mean signed along-track offset of V ahead of ego.
            n = min(len(ego_traj), len(veh))
            offs = []
            for t in range(n):
                t2 = min(t + 1, n - 1)
                t0 = max(t - 1, 0)
                hx = float(ego_traj[t2, 0] - ego_traj[t0, 0])
                hy = float(ego_traj[t2, 1] - ego_traj[t0, 1])
                elen = math.hypot(hx, hy)
                if elen < 1e-9:
                    continue
                offs.append(((float(veh[t, 0]) - float(ego_traj[t, 0]))
                             * hx / elen
                             + (float(veh[t, 1]) - float(ego_traj[t, 1]))
                             * hy / elen))
            if not offs:
                continue
            order = "after" if sum(offs) / len(offs) > 0 else "before"
            ctrl.add("base", [],
                     f"merge_gap_sequence(ego, {vid}, {order}).")

    def _add_lane_membership_to(self, ctrl, trajectories: Dict[str, np.ndarray],
                                map_data: Dict):
        """Emit in_lane_fact/3 for exact point-segment membership (MC-1).

        The ASP in_lane/3 derivation only handles axis-aligned segments
        (32-bit integer limits); this covers arbitrary orientations with
        proper perpendicular distance in Python. in_lane/3 holds when
        either derivation holds (both are sound sufficient conditions).

        A vehicle is in-lane when its clamped point-segment distance is
        within half the lane width. Lanes without a lane_width entry
        produce no facts (mirrors the ASP rule's requirement).
        """
        import math
        widths = map_data.get('lane_width', {})
        lanes = map_data.get('lanes', {})
        for vid, traj in trajectories.items():
            try:
                pts = [(float(p[0]), float(p[1])) for p in traj]
            except (TypeError, ValueError, IndexError):
                continue
            for lid, segments in lanes.items():
                if lid not in widths:
                    continue
                try:
                    half = float(widths[lid]) / 2.0
                except (TypeError, ValueError):
                    continue
                lane_w = half * 2.0
                for seg in segments:
                    try:
                        x1, y1 = float(seg['x1']), float(seg['y1'])
                        x2, y2 = float(seg['x2']), float(seg['y2'])
                    except (KeyError, TypeError, ValueError):
                        continue
                    dx, dy = x2 - x1, y2 - y1
                    seg_len2 = dx * dx + dy * dy
                    seg_len = math.sqrt(seg_len2) if seg_len2 > 0 else 0.0
                    for t, (px, py) in enumerate(pts):
                        if seg_len2 > 1e-12:
                            s_raw = ((px - x1) * dx + (py - y1) * dy) \
                                / seg_len2
                            fx, fy = x1 + s_raw * dx, y1 + s_raw * dy
                        else:
                            s_raw, fx, fy = 0.0, x1, y1
                        # Perpendicular distance to the segment LINE
                        # (not the clamped endpoint): the end tolerance
                        # below handles past-the-end points.
                        if math.hypot(px - fx, py - fy) > half:
                            continue
                        if 0.0 <= s_raw <= 1.0:
                            emit = True
                        else:
                            # Past-the-end tolerance mirrors the ASP
                            # clause ((X-X1)*(X-X2) <= W*W): a point up
                            # to e meters beyond the end counts when
                            # e*(e+seg_len) <= W*W. Orientation-general.
                            e = (abs(s_raw - 1.0) if s_raw > 1.0
                                 else abs(s_raw)) * seg_len
                            emit = (e * (e + seg_len)
                                    <= lane_w * lane_w)
                        if emit:
                            ctrl.add(
                                "base", [],
                                f"in_lane_fact({vid}, {t}, {lid}).")

    def _add_extras_to(self, ctrl, extras: Dict):
        """Add arbitrary extra ASP facts.
        
        extras structure: {'facts': [str, ...]} where each string is a
        complete ASP fact (e.g. 'speed(ego, 2, 1560).').
        """
        for fact in extras.get('facts', []):
            ctrl.add("base", [], fact)
    
    def get_safe_trajectory(
        self,
        trajectory: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """Return corrected trajectory that satisfies all rules.
        
        Args:
            trajectory: Original trajectory, shape (T, 2)
            **kwargs: Arguments for check() method
            
        Returns:
            Corrected trajectory, shape (T, 2)
            If already safe, returns original trajectory (no copy)
        """
        is_safe, violations = self.check(trajectory, **kwargs)

        if is_safe:
            return trajectory

        # GAP-8: TrajectoryCorrector owns all correction strategies;
        # delegate instead of duplicating logic here.
        try:
            from corrector import TrajectoryCorrector
        except ImportError:  # pragma: no cover
            from src.corrector import TrajectoryCorrector
        corrector = TrajectoryCorrector(self)
        corrected, _ = corrector.correct(trajectory, **kwargs)
        return corrected

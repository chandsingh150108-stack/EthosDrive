"""Trajectory Correction Engine.

This module provides the TrajectoryCorrector class that fixes
invalid trajectories to satisfy ASP traffic rules.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import TrajectoryChecker
try:
    from checker import TrajectoryChecker
except ImportError:
    from src.checker import TrajectoryChecker

# Central rule -> strategy mapping (IF-1). This module must be the
# ONLY place that maps finding names to strategies.
try:
    from correction_strategies import (
        strategy_for, DEFAULT_PARAMS, all_known_rules,
    )
except ImportError:
    from src.correction_strategies import (
        strategy_for, DEFAULT_PARAMS, all_known_rules,
    )

# Continuous optimizer over ASP-emitted bounds (P3/P4). Pure-NumPy
# projected gradient; kept separate from the heuristic fixpoint above.
try:
    from constraint_optimizer import ConstraintOptimizer
except ImportError:
    from src.constraint_optimizer import ConstraintOptimizer


def _is_checker_like(checker) -> bool:
    """Accept TrajectoryChecker by identity or by interface (P-16).

    checker.py is importable as both `checker` and `src.checker`;
    the two module objects make isinstance() fail on a perfectly
    valid checker (dual-identity). Accept either the exact type or
    any object exposing the check()/get_safe_trajectory() interface.
    """
    if isinstance(checker, TrajectoryChecker):
        return True
    return callable(getattr(checker, 'check', None)) and callable(
        getattr(checker, 'get_safe_trajectory', None))


class TrajectoryCorrector:
    """Corrects invalid trajectories to satisfy traffic rules.
    
    This class takes an invalid trajectory and applies corrections
    to make it satisfy all ASP traffic rules.
    
    Attributes:
        checker: TrajectoryChecker instance
        max_iterations: Maximum optimization iterations
    """
    
    def __init__(self, checker: TrajectoryChecker, max_iterations: int = 100,
                 params: Optional[Dict] = None,
                 params_file: Optional[str] = None,
                 on_failure: str = 'raise'):
        """Initialize the corrector.

        Args:
            checker: Initialized TrajectoryChecker instance
            max_iterations: Maximum number of correction iterations
            params: Numeric strategy parameters (overrides file/defaults)
            params_file: Path to YAML params file. Defaults to
                <repo>/configs/correction_strategies.yaml when it exists.
            on_failure: 'raise' (default) raises RuntimeError when the
                fixpoint exhausts max_iterations; 'brake' instead holds
                the trajectory from the earliest remaining violation
                (maximum-braking fallback) and returns it with an
                'emergency_hold_uncorrectable' entry -- the upstream
                uncorrectable-plan contract (flag + hold + takeover).

        Raises:
            TypeError: If checker is not a TrajectoryChecker
            ValueError: If on_failure is neither 'raise' nor 'brake'
        """
        if not _is_checker_like(checker):
            raise TypeError("checker must be a TrajectoryChecker instance")
        if on_failure not in ('raise', 'brake'):
            raise ValueError("on_failure must be 'raise' or 'brake'")

        self.checker = checker
        self.max_iterations = max_iterations
        self.on_failure = on_failure
        self.params = dict(DEFAULT_PARAMS)
        self.params.update(self._load_params(params_file))
        if params:
            self.params.update(params)
    
    @staticmethod
    def _load_params(params_file: Optional[str]) -> Dict:
        """Load numeric params from a YAML file (RQ-1).

        Resolves <repo>/configs/correction_strategies.yaml by default.
        Returns {} when the file is absent or pyyaml is unavailable so
        built-in DEFAULT_PARAMS always apply.
        """
        candidates = []
        if params_file:
            candidates.append(Path(params_file))
        else:
            candidates.append(
                Path(__file__).parent.parent / "configs"
                / "correction_strategies.yaml"
            )
        for path in candidates:
            try:
                if path.exists():
                    import yaml
                    with open(path, 'r') as f:
                        data = yaml.safe_load(f) or {}
                    return {k: v for k, v in data.items()
                            if k in DEFAULT_PARAMS}
            except Exception:
                continue
        return {}
    
    def correct(
        self,
        trajectory: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Correct an invalid trajectory.
        
        Args:
            trajectory: Invalid trajectory, shape (T, 2)
            **kwargs: Arguments for TrajectoryChecker.check()
            
        Returns:
            Tuple of:
                - corrected_trajectory: Shape (T, 2)
                - applied_corrections: List[Dict] with structure:
                    {
                        'rule': str,
                        'timestep': int,
                        'correction_type': str,
                        'original_value': Any,
                        'corrected_value': Any
                    }
                    
        Raises:
            RuntimeError: If correction fails after max_iterations
                (unless on_failure='brake', which holds instead)
        """
        # Check if already valid
        is_safe, violations = self.checker.check(trajectory, **kwargs)

        if is_safe:
            return trajectory.copy(), []

        # Apply corrections iteratively, grouped by timestep. Warnings
        # (advisory) and non-geometric findings (late_signal) are
        # reported but excluded from the fixpoint so the loop terminates.
        corrected = trajectory.copy()
        applied_corrections = []
        group_counts: Dict[str, int] = {}
        group_cap = int(self.params.get('group_cap', 40))
        # Stagnation detection (PR-1): identical violation sets stop
        # improving the plan; break to the fallback early instead of
        # burning max_iterations × full re-grounds on churn.
        stagnant = 0
        last_sig = None

        for iteration in range(self.max_iterations):
            # Check current state
            is_safe, violations = self.checker.check(corrected, **kwargs)

            if is_safe:
                return corrected, applied_corrections

            hard = [v for v in violations
                    if v.get("type", "violation") != "warning"
                    and v["rule"] != "late_signal"]
            if not hard:
                for v in violations:
                    applied_corrections.append({
                        'rule': v['rule'],
                        'timestep': v['time'],
                        'correction_type': 'advisory_no_change',
                        'original_value': corrected[v['time']].tolist(),
                        'corrected_value': corrected[v['time']].tolist(),
                    })
                return corrected, applied_corrections

            sig = frozenset((v['rule'], v.get('vehicle', ''),
                             v.get('vehicle2', ''), v['time'])
                            for v in hard)
            if sig == last_sig:
                stagnant += 1
                if stagnant >= 3:
                    break
            else:
                stagnant = 0
                last_sig = sig

            by_time = {}
            for v in hard:
                by_time.setdefault(v['time'], []).append(v)
            # Stale-snapshot guard (P-16): fixes are computed from one
            # check, but tail-rewriting strategies (hold/speed/boost)
            # invalidate later timesteps in the same pass. Track the
            # rewritten range and skip already-dirty timesteps; they
            # are revisited next iteration with a fresh check.
            # Per-group caps bound single-strategy churn (P-6).
            dirty_from = len(corrected)
            for t in sorted(by_time):
                if t >= dirty_from:
                    continue
                rule = by_time[t][0]['rule']
                group, _ = strategy_for(rule)
                group_key = group or 'advisory'
                if group_counts.get(group_key, 0) >= group_cap:
                    continue
                before = corrected.copy()
                corrected, correction = self._apply_correction(
                    corrected, by_time[t][0], **kwargs
                )
                applied_corrections.append(correction)
                group_counts[group_key] = group_counts.get(group_key, 0) + 1
                if not np.array_equal(before[t:], corrected[t:]):
                    dirty_from = min(dirty_from, t)

        # Final check
        is_safe, violations = self.checker.check(corrected, **kwargs)
        if not is_safe:
            if self.on_failure == 'brake':
                # Maximum-braking fallback: hold from the earliest
                # remaining violation and flag the plan uncorrectable
                # for upstream takeover instead of raising.
                hard = [v for v in violations
                        if v.get("type", "violation") != "warning"
                        and v["rule"] != "late_signal"] or violations
                t0 = max(min(v['time'] for v in hard), 1)
                corrected, _ = self._hold_position(
                    corrected, t0, **kwargs)
                applied_corrections.append({
                    'rule': hard[0]['rule'],
                    'timestep': t0,
                    'correction_type':
                        'emergency_hold_uncorrectable',
                    'original_value': trajectory[
                        min(t0, len(trajectory) - 1)].tolist(),
                    'corrected_value': corrected[
                        min(t0, len(corrected) - 1)].tolist(),
                })
                return corrected, applied_corrections
            raise RuntimeError(
                f"Failed to correct trajectory after {self.max_iterations} iterations"
            )

        return corrected, applied_corrections
    
    def _apply_correction(
        self,
        trajectory: np.ndarray,
        violation: Dict,
        **kwargs
    ) -> Tuple[np.ndarray, Dict]:
        """Apply correction for a specific violation.
        
        Args:
            trajectory: Current trajectory
            violation: Violation dictionary
            
        Returns:
            Tuple of (corrected_trajectory, correction_info)
        """
        rule = violation['rule']
        t = violation['time']

        corrected = trajectory.copy()
        correction_info = {
            'rule': rule,
            'timestep': t,
            'original_value': trajectory[t].tolist()
        }

        # Dispatch via the central rule -> strategy table (IF-1).
        # Every branch changes the trajectory (except advisory
        # late_signal, handled by the caller) so the fixpoint always
        # makes progress.
        group, method_name = strategy_for(rule)

        if group is None:
            # Advisory finding: report, do not touch the trajectory.
            correction_info['correction_type'] = 'advisory_no_change'

        elif group == 'unknown':
            # Unknown future rule: conservative hold guarantees progress.
            corrected, correction_type = self._hold_position(
                corrected, t, **kwargs
            )
            correction_info['correction_type'] = (
                f"hold_unknown_{correction_type}"
            )

        else:
            strategy = getattr(self, method_name)
            corrected, correction_type = strategy(
                corrected, t, **kwargs
            )
            correction_info['correction_type'] = correction_type

        correction_info['corrected_value'] = corrected[t].tolist()

        return corrected, correction_info
    
    def _resolve_speed_target(self, trajectory: np.ndarray, t: int,
                                **kwargs) -> float:
        """Resolve the applicable speed target in m/s (P-16).

        Conservative minimum over: active zone speed limits covering
        the violation point, map speed limits, then the configured
        default. kwargs carries the same map_data/zones passed to
        check(), so no new plumbing is needed.
        """
        candidates = []
        x, y = float(trajectory[t, 0]), float(trajectory[t, 1])
        for zone in kwargs.get('zones') or []:
            try:
                if not zone.get('active', False):
                    continue
                if (zone['x1'] <= x <= zone['x2']
                        and zone['y1'] <= y <= zone['y2']
                        and 'speed_limit' in zone):
                    candidates.append(float(zone['speed_limit']))
            except (KeyError, TypeError):
                continue
        map_data = kwargs.get('map_data') or {}
        for lim in (map_data.get('speed_limit') or {}).values():
            try:
                candidates.append(float(lim))
            except (TypeError, ValueError):
                continue
        candidates.append(float(
            self.params['default_speed_target_ms']))
        return max(0.1, min(candidates))
    
    def _resolve_lane_center(self, trajectory: np.ndarray, t: int,
                             **kwargs) -> float:
        """Resolve the lane-center y in meters for the violation point.

        Uses the nearest map lane centerline covering the point's x
        (axis-aligned segments); falls back to the configured default
        when no map lane matches (P-16: no more hardcoded y=0).
        """
        x = float(trajectory[t, 0])
        best_y = None
        best_dist = None
        map_data = kwargs.get('map_data') or {}
        for segments in (map_data.get('lanes') or {}).values():
            for seg in segments:
                try:
                    x1, y1, x2, y2 = (float(seg['x1']), float(seg['y1']),
                                      float(seg['x2']), float(seg['y2']))
                except (KeyError, TypeError, ValueError):
                    continue
                if y1 == y2 and min(x1, x2) <= x <= max(x1, x2):
                    d = abs(float(trajectory[t, 1]) - y1)
                    if best_dist is None or d < best_dist:
                        best_dist, best_y = d, y1
        if best_y is None:
            return float(self.params['fallback_lane_center_y'])
        return best_y
    
    def _correct_speed_limit(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct speed limit violation.
        
        Strategy: Scale trajectory from the violation point onward so
        velocity stays under the applicable limit (map/zone-aware,
        config default otherwise).
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        if t == 0:
            return trajectory, "no_change"
        
        # Compute current velocity magnitude at violation timestep
        dx = trajectory[t, 0] - trajectory[t-1, 0]
        dy = trajectory[t, 1] - trajectory[t-1, 1]
        current_speed_ms = np.sqrt(dx**2 + dy**2)  # m/s (positions are in meters)
        
        target_speed_ms = self._resolve_speed_target(
            trajectory, t, **kwargs)
        
        if current_speed_ms > target_speed_ms:
            scale = target_speed_ms / current_speed_ms
            # Scale entire trajectory from the violation point onward
            base_x = trajectory[t-1, 0]
            base_y = trajectory[t-1, 1]
            for t2 in range(t, len(trajectory)):
                rel_x = trajectory[t2, 0] - base_x
                rel_y = trajectory[t2, 1] - base_y
                trajectory[t2, 0] = base_x + rel_x * scale
                trajectory[t2, 1] = base_y + rel_y * scale
        
        return trajectory, "velocity_scaled_to_limit"
    
    def _correct_collision(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct collision avoidance violation.

        Strategy: Shift the whole tail laterally (configurable
        distance). A single-point shift manufactures an acceleration
        spike at t+1 that ping-pongs with comfort rules (P-16); rigid
        tail translation preserves every downstream segment shape, so
        only the t-1 -> t entry segment changes.
        """
        shift = float(self.params['lateral_shift_m'])
        trajectory[t:, 1] += shift
        return trajectory, f"lateral_shift_{shift:g}m"
    
    def _correct_lane_keeping(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct lane keeping violation.
        
        Strategy: Project to the nearest map lane center (config
        fallback when no map lane matches).
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        trajectory[t, 1] = self._resolve_lane_center(
            trajectory, t, **kwargs)
        return trajectory, "project_to_lane_center"
    
    def _correct_safe_following(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Correct safe following distance violation.
        
        Strategy: Reduce speed by 30% to increase following distance.
        
        Args:
            trajectory: Current trajectory
            t: Violation timestep
            
        Returns:
            Tuple of (corrected_trajectory, correction_type)
        """
        if t == 0:
            return trajectory, "no_change"
        
        dx = trajectory[t, 0] - trajectory[t-1, 0]
        dy = trajectory[t, 1] - trajectory[t-1, 1]
        
        scale = float(self.params['follow_scale'])
        trajectory[t, 0] = trajectory[t-1, 0] + dx * scale
        trajectory[t, 1] = trajectory[t-1, 1] + dy * scale

        return trajectory, "velocity_reduction_30pct"

    def _hold_position(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Stop-and-hold: zero velocity at t and freeze the tail.

        Used for all signal, yield, right-of-way, emergency and
        maneuver-suppression findings. Freezing the tail (instead of
        only pinning t) avoids manufacturing a new acceleration spike
        at t+1 that would ping-pong with comfort rules.

        At t<=0 there is no earlier point to hold from, and collapsing
        the whole trajectory to the start point destroys information
        without fixing a start-state violation (e.g. starting inside an
        obstacle/box on red). Return unchanged so the caller can try
        other strategies or honestly report uncorrectable.
        """
        if t <= 0:
            return trajectory, "hold_from_start_noop"
        stop_x, stop_y = trajectory[t-1, 0], trajectory[t-1, 1]
        for t2 in range(t, len(trajectory)):
            trajectory[t2, 0] = stop_x
            trajectory[t2, 1] = stop_y
        return trajectory, "hold_position"

    def _smooth_point(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Gently pull a point 25% toward its neighbors' midpoint.

        Used for comfort/quality findings (R12.x, R7.5).
        """
        n = len(trajectory)
        dampen = float(self.params['smooth_edge_dampen'])
        blend = float(self.params['smooth_blend'])
        if n < 3 or t <= 0 or t >= n - 1:
            if t > 0:
                dx = trajectory[t, 0] - trajectory[t-1, 0]
                dy = trajectory[t, 1] - trajectory[t-1, 1]
                trajectory[t, 0] = trajectory[t-1, 0] + dx * dampen
                trajectory[t, 1] = trajectory[t-1, 1] + dy * dampen
            return trajectory, "smooth_edge_dampen"
        mid_x = (trajectory[t-1, 0] + trajectory[t+1, 0]) / 2.0
        mid_y = (trajectory[t-1, 1] + trajectory[t+1, 1]) / 2.0
        trajectory[t, 0] += (mid_x - trajectory[t, 0]) * blend
        trajectory[t, 1] += (mid_y - trajectory[t, 1]) * blend
        return trajectory, "smooth_midpoint_blend"

    def _accelerate_point(
        self,
        trajectory: np.ndarray,
        t: int,
        **kwargs
    ) -> Tuple[np.ndarray, str]:
        """Raise speed toward the posted minimum (R1.4).

        Targeted tail stretch: scale the tail from t onward so the
        slowest downstream segment reaches the strictest applicable
        min_speed_limit (mirrors _resolve_speed_target). A blind fixed
        boost overshoots the speed cap on healthy segments and
        ping-pongs against the scale-down strategy (T-8). From a
        near-stop (min segment < 0.5 m/s) the ratio is unbounded, so
        fall back to one gentle fixed boost per visit.
        """
        if t == 0:
            return trajectory, "no_change"
        segs = [float(np.linalg.norm(trajectory[k] - trajectory[k - 1]))
                for k in range(t, len(trajectory))]
        if not segs:
            return trajectory, "no_change"
        cur_min = min(segs)
        if cur_min < 0.5:
            boost = float(self.params['min_speed_boost'])
        else:
            target = self._resolve_min_target(**kwargs)
            if target <= cur_min:
                return trajectory, "min_target_met"
            boost = min(target / cur_min,
                        float(self.params['min_speed_boost']) * 4)
        base_x, base_y = trajectory[t-1, 0], trajectory[t-1, 1]
        for t2 in range(t, len(trajectory)):
            trajectory[t2, 0] = base_x + (trajectory[t2, 0] - base_x) * boost
            trajectory[t2, 1] = base_y + (trajectory[t2, 1] - base_y) * boost
        return trajectory, "velocity_boost_10pct"

    def _resolve_min_target(self, **kwargs) -> float:
        """Resolve the applicable minimum speed in m/s.

        Strictest (max) map min_speed_limit; 0.0 when none is mapped
        (callers then fall back to a gentle fixed boost).
        """
        best = 0.0
        map_data = kwargs.get('map_data') or {}
        for lim in (map_data.get('min_speed_limit') or {}).values():
            try:
                best = max(best, float(lim))
            except (TypeError, ValueError):
                continue
        return best


class CorrectorQP:
    """ASP-constrained corrector (P4): solver bounds + continuous repair.

    Check with constraints -> projected-gradient solve -> symbolic
    re-validation, instead of the heuristic per-rule fixpoint. Falls
    back to TrajectoryCorrector (same on_failure contract) when the
    constrained solve cannot clear the violations.

    Attributes:
        checker: TrajectoryChecker instance (must support
            check(include_constraints=True) from P1).
        optimizer: ConstraintOptimizer instance.
        max_rounds: solve + re-validate rounds before fallback.
        on_failure: 'raise' or 'brake' (mirrors TrajectoryCorrector).
    """

    def __init__(self, checker, optimizer=None,
                 optimizer_params: Optional[Dict] = None,
                 max_rounds: int = 3,
                 on_failure: str = 'raise',
                 heuristic_fallback: bool = True,
                 bounds: str = 'full'):
        """Initialize the QP corrector.

        Args:
            checker: Initialized TrajectoryChecker instance.
            optimizer: ConstraintOptimizer (built from optimizer_params
                when omitted).
            optimizer_params: Per-solve knobs (lr, steps, tol_cm,
                lane_pull); also forwarded per correct() call.
            max_rounds: Maximum solve + re-validate rounds.
            on_failure: 'raise' (RuntimeError) or 'brake' (heuristic
                corrector's flagged emergency hold).
            heuristic_fallback: If True (default), delegate to
                TrajectoryCorrector when the constrained solve fails;
                if False, apply on_failure directly (raise, or hold
                from the earliest remaining violation).
            bounds: Bound-family ablation switch (P6): 'full' (all
                emitted bounds), 'caps' (v_max/v_min/a_max only),
                'boxes' (x_bounds/y_bounds/lane_center only). Pure
                Python filtering of the constraints dict -- no .lp
                changes -- so subset runs stay comparable.

        Raises:
            TypeError: If checker is not checker-like.
            ValueError: If on_failure is neither 'raise' nor 'brake',
                or bounds is not a known family.
        """
        if not _is_checker_like(checker):
            raise TypeError("checker must be a TrajectoryChecker instance")
        if on_failure not in ('raise', 'brake'):
            raise ValueError("on_failure must be 'raise' or 'brake'")
        if bounds not in ('full', 'caps', 'boxes'):
            raise ValueError("bounds must be 'full', 'caps' or 'boxes'")
        self.checker = checker
        self.optimizer = optimizer or ConstraintOptimizer(
            params=optimizer_params)
        self.optimizer_params = optimizer_params or {}
        self.max_rounds = max(1, int(max_rounds))
        self.on_failure = on_failure
        self.heuristic_fallback = heuristic_fallback
        self.bounds = bounds

    @staticmethod
    def _filter_bounds(constraints: Dict[str, list],
                       bounds: str) -> Dict[str, list]:
        """Keep one bound family for ablation (P6).

        'caps' (limit-type: v_max/v_min/a_max) vs 'boxes'
        (region-type: x_bounds/y_bounds/lane_center). 'full' returns
        the dict unchanged.
        """
        if bounds == 'full':
            return constraints
        keep = {'caps': {'v_max', 'v_min', 'a_max'},
                'boxes': {'x_bounds', 'y_bounds', 'lane_center'}}[bounds]
        return {k: (v if k in keep else []) for k, v in constraints.items()}

    def correct(
        self,
        trajectory: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, List[Dict]]:
        """Correct via constrained optimization with re-validation.

        Args:
            trajectory: Invalid trajectory, shape (T, 2).
            **kwargs: Arguments for TrajectoryChecker.check()
                (include_constraints is managed internally and never
                forwarded, so the heuristic fallback always sees the
                legacy 2-tuple contract).

        Returns:
            Tuple of (corrected_trajectory, applied_corrections) in
            the same 5-key schema as TrajectoryCorrector.correct.

        Raises:
            RuntimeError: If the solve fails and on_failure='raise'.
        """
        kwargs.pop('include_constraints', None)
        kwargs.pop('include_internal', None)
        orig = np.asarray(trajectory, dtype=float)

        is_safe, violations, constraints = self.checker.check(
            orig, include_constraints=True, **kwargs)
        if is_safe:
            return orig.copy(), []
        constraints = self._filter_bounds(constraints, self.bounds)

        current = orig.copy()
        rounds = 0
        last_info: Dict = {}
        # P-23 increment 2: accumulate bounds across rounds. Replacing
        # the constraint set each round let a later round undo an
        # earlier fix (the solved plan moved back toward the original
        # and re-violated round 1's rule) -- the measured whack-a-mole.
        # Merging is sound: the optimizer's indexing keeps the tightest
        # box / minimum cap per timestep, so accumulation only tightens.
        accum = dict(constraints)
        for rounds in range(1, self.max_rounds + 1):
            solved, info = self.optimizer.solve(
                current, accum, params=self.optimizer_params)
            last_info = info
            ok, v2, c2 = self.checker.check(
                solved, include_constraints=True, **kwargs)
            if ok:
                return solved, self._qp_entries(orig, solved, violations,
                                                rounds, info)
            current = solved
            accum = self._merge_constraints(
                accum, self._filter_bounds(c2, self.bounds))

        # Constrained solve could not clear the violations.
        if self.heuristic_fallback:
            heur = TrajectoryCorrector(
                self.checker, on_failure=self.on_failure)
            fixed, entries = heur.correct(orig, **kwargs)
            attempt = {
                'rule': 'qp_optimization',
                'timestep': int(min(v['time'] for v in violations)),
                'correction_type': (
                    f"qp_infeasible_{rounds}r_fallback_"
                    f"{last_info.get('status', 'unknown')}"),
                'original_value': orig[
                    min(int(min(v['time'] for v in violations)),
                        len(orig) - 1)].tolist(),
                'corrected_value': fixed[
                    min(int(min(v['time'] for v in violations)),
                        len(fixed) - 1)].tolist(),
            }
            return fixed, [attempt] + entries
        if self.on_failure == 'brake':
            heur = TrajectoryCorrector(
                self.checker, on_failure='brake')
            fixed, entries = heur.correct(orig, **kwargs)
            return fixed, entries
        raise RuntimeError(
            f"CorrectorQP failed after {rounds} constrained rounds "
            f"({last_info.get('status', 'unknown')})"
        )

    @staticmethod
    def _merge_constraints(old: Dict[str, list],
                           new: Dict[str, list]) -> Dict[str, list]:
        """Union two grouped constraint dicts (P-23 increment 2).

        Concatenates per-family entry lists; the optimizer's indexing
        keeps the tightest box / minimum cap per timestep, so the union
        only ever tightens. Entries are plain tuples, so no aliasing:
        the merged dict shares no mutable state with either input.
        """
        merged: Dict[str, list] = {}
        for key in set(old) | set(new):
            merged[key] = list(old.get(key, [])) + list(new.get(key, []))
        return merged

    @staticmethod
    def _qp_entries(orig: np.ndarray, solved: np.ndarray,
                    violations: List[Dict], rounds: int,
                    info: Dict) -> List[Dict]:
        """One applied entry per moved timestep (1 cm threshold)."""
        entries = []
        moved = [t for t in range(1, len(orig))
                 if float(np.linalg.norm(solved[t] - orig[t])) > 0.01]
        if not moved:
            moved = [int(min(v['time'] for v in violations))]
        for t in moved:
            t = max(1, min(int(t), len(orig) - 1))
            entries.append({
                'rule': 'qp_optimization',
                'timestep': t,
                'correction_type': (
                    f"projected_gradient_{info.get('status', 'unknown')}"
                    f"_{rounds}r_{info.get('iterations', 0)}it"),
                'original_value': orig[t].tolist(),
                'corrected_value': solved[t].tolist(),
            })
        return entries

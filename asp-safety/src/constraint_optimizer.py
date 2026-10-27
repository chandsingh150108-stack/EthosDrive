"""Continuous optimizer over ASP-emitted correction constraints (P3).

MVP: pure-NumPy projected-gradient descent (no new dependencies).
Minimizes L2 deviation from the original trajectory subject to the
box + speed + accel bounds in a checker constraints dict
(see TrajectoryChecker._group_constraints; all values integer cm).

P-23 (horizon-joint solving): each iteration alternates the box /
speed / accel projections until they hold JOINTLY, instead of a
single sequential pass whose accel clamp could push points back out
of tight boxes (the scene-8 whack-a-mole class: a moving CV-projected
threat plus tail coupling made the iteration cycle, so more solve
rounds never converged).

Conventions (docs/conventions.md): positions meters in code, cm in
ASP; T=0 is the anchored ego pose and is never moved.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

_CM = 100.0  # cm per meter

DEFAULT_OPT_PARAMS: Dict[str, float] = {
    'lr': 0.2,          # gradient step toward the original trajectory
    'steps': 50,        # max projection iterations
    'tol_cm': 1.0,      # early-stop when max per-point move < tol
    'lane_pull': 0.05,  # weak spring toward lane_center per iteration
    'joint_iters': 5,   # P-23: inner alternation passes per iteration
    'feas_tol_cm': 1.0,  # P-23: joint-feasibility tolerance (cm)
}


class ConstraintOptimizer:
    """Projected-gradient solver over ASP constraint bounds.

    Attributes:
        params: solver knobs (lr, steps, tol_cm, lane_pull).
    """

    def __init__(self, params: Optional[Dict] = None):
        """Initialize the optimizer.

        Args:
            params: overrides for DEFAULT_OPT_PARAMS.
        """
        self.params = dict(DEFAULT_OPT_PARAMS)
        if params:
            self.params.update(params)

    def solve(
        self,
        trajectory: np.ndarray,
        constraints: Dict[str, list],
        params: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Solve for the feasible trajectory closest to the original.

        Args:
            trajectory: shape (T, 2), meters, finite. T=0 is pinned.
            constraints: grouped dict from check(include_constraints=True).
            params: per-call overrides.

        Returns:
            Tuple of (corrected (T, 2) meters, info dict with status,
            iterations, joint_iters, solve_time_ms, pinned_t0).
            status is 'optimal' when the returned trajectory satisfies
            all constraint families jointly; 'projection_cycle' when
            the inner alternation cap was hit without joint
            feasibility (the caller's re-validation decides).

        Raises:
            ValueError: on bad shape, empty input, or non-finite values.
        """
        traj = np.asarray(trajectory, dtype=float)
        if traj.ndim != 2 or traj.shape[1] != 2 or traj.shape[0] == 0:
            raise ValueError(
                f"Expected trajectory shape (T, 2), got {traj.shape}")
        if traj.shape[0] > 6:
            raise ValueError(
                f"Maximum 6 timesteps allowed, got {traj.shape[0]}")
        if not np.all(np.isfinite(traj)):
            raise ValueError("Trajectory must be finite (no NaN/inf)")

        cfg = dict(self.params)
        if params:
            cfg.update(params)
        lr = float(cfg['lr'])
        steps = int(cfg['steps'])
        tol_m = float(cfg['tol_cm']) / _CM
        lane_pull = float(cfg['lane_pull'])
        joint_max = max(1, int(cfg.get('joint_iters', 5)))
        feas_m = float(cfg.get('feas_tol_cm', cfg['tol_cm'])) / _CM

        orig = traj.copy()
        work = traj.copy()
        n = len(work)

        # Index constraints per timestep (keep tightest overlapping box).
        x_bounds = self._index_box(constraints.get('x_bounds', []))[0]
        y_bounds, moving_boxes = self._index_box(constraints.get('y_bounds', []))
        # Range constraints (P-23 inc3): coupled corridors
        x_ranges = self._index_range(constraints.get('x_bounds_range', []))
        y_ranges = self._index_range(constraints.get('y_bounds_range', []))
        v_max = self._index_cap(constraints.get('v_max', []))
        v_min = self._index_cap(constraints.get('v_min', []))
        a_max = self._index_cap(constraints.get('a_max', []))
        lane_c = self._index_cap(constraints.get('lane_center', []))

        t0 = time.perf_counter()
        it = 0
        joint_iters = 0
        joint_feasible = True
        for it in range(1, steps + 1):
            prev = work.copy()
            # Gradient step toward the original + weak lane-center spring.
            work += lr * (orig - work)
            for t, yc in lane_c.items():
                if 0 < t < n:
                    work[t, 1] += lane_pull * (yc / _CM - work[t, 1])
            # P-23: a single sequential pass (boxes -> speeds -> accel)
            # can push points back out of earlier boxes -- the accel
            # clamp fights tight boxes at a moving threat's timesteps
            # and the iteration cycles instead of converging. Alternate
            # the projections until ALL families hold jointly (or the
            # inner cap is hit and the status says so honestly).
            for joint_iters in range(1, joint_max + 1):
                self._clip_boxes(work, x_bounds, y_bounds)
                self._project_moving_boxes(work, moving_boxes)  # P-23 inc2
                self._project_ranges(work, x_ranges, y_ranges)  # P-23 inc3
                self._project_speeds(work, v_max, v_min)
                self._clamp_accel(work, a_max)
                if self._jointly_feasible(work, x_bounds, y_bounds, y_ranges, x_ranges,
                                          v_max, v_min, a_max, feas_m):
                    break
            work[0] = orig[0]
            if float(np.max(np.abs(work - prev))) < tol_m:
                break
        solve_ms = (time.perf_counter() - t0) * 1000.0

        joint_feasible = self._jointly_feasible(work, x_bounds, y_bounds, y_ranges, x_ranges,
                                                v_max, v_min, a_max,
                                                feas_m)
        info = {
            'status': 'optimal' if joint_feasible else 'projection_cycle',
            'iterations': it,
            'joint_iters': joint_iters,
            'solve_time_ms': solve_ms,
            'pinned_t0': True,
        }
        return work, info

    @staticmethod
    def _index_box(entries: list) -> Tuple[Dict[int, Tuple[float, float]], List[Dict]]:
        """(t, min_cm, max_cm) entries -> ({t: (min_m, max_m)}, moving_boxes).

        Also detects consecutive y_bounds with linearly moving centers
        (constant velocity) as "moving boxes" - these are projected as
        a coupled rigid-body segment instead of independent per-timestep
        boxes (P-23 increment 2).
        """
        out: Dict[int, Tuple[float, float]] = {}
        raw: Dict[int, Tuple[float, float]] = {}
        for e in entries:
            try:
                t, lo, hi = int(e[0]), float(e[1]), float(e[2])
            except (IndexError, TypeError, ValueError):
                continue
            if lo > hi:
                lo, hi = hi, lo
            lo_m, hi_m = lo / _CM, hi / _CM
            raw[t] = (lo_m, hi_m)
            if lo > hi:
                lo, hi = hi, lo
            lo_m, hi_m = lo / _CM, hi / _CM
            if t in out:
                plo, phi = out[t]
                out[t] = (max(plo, lo_m), min(phi, hi_m))
            else:
                out[t] = (lo_m, hi_m)

        # Detect moving boxes: consecutive timesteps with linearly
        # moving centers (constant velocity). Returns list of
        # {t0, t1, center_vel_m} where center = (lo+hi)/2.
        moving_boxes: List[Dict] = []
        if len(raw) >= 3:
            ts = sorted(raw.keys())
            i = 0
            while i < len(ts):
                t0 = ts[i]
                lo0, hi0 = raw[t0]
                center0 = (lo0 + hi0) * 0.5
                j = i + 1
                while j < len(ts):
                    tj = ts[j]
                    if tj != ts[j - 1] + 1:  # must be consecutive
                        break
                    loj, hij = raw[tj]
                    centerj = (loj + hij) * 0.5
                    expected_vel = (centerj - center0) / (tj - t0)
                    # check all intermediate timesteps for linear fit
                    ok = True
                    for k in range(i + 1, j):
                        tk = ts[k]
                        lok, hik = raw[tk]
                        centerk = (lok + hik) * 0.5
                        expected = center0 + expected_vel * (tk - t0)
                        if abs(centerk - expected) > 1e-6:  # exact linear
                            ok = False
                            break
                    if not ok:
                        break
                    j += 1
                if j - i >= 3:  # at least 3 timesteps = moving box
                    t1 = ts[j - 1]
                    center1 = (raw[t1][0] + raw[t1][1]) * 0.5
                    vel = (center1 - center0) / (t1 - t0) if t1 > t0 else 0.0
                    moving_boxes.append({
                        't0': t0, 't1': ts[j - 1],
                        'center0': center0, 'center1': center1,
                        'vel': vel,
                        'half_width': (hi0 - lo0) * 0.5
                    })
                i = j if j > i else i + 1
        return out, moving_boxes

    @staticmethod
    def _index_range(entries: list) -> Dict[int, Tuple[int, int, float, float]]:
        """(t0, t1, min_cm, max_cm) entries -> {t: (t0, t1, min_m, max_m)}.

        Range constraints span a window [t0, t1]. For each timestep t in
        [t0, t1], the tightest range covering t wins.
        """
        out: Dict[int, Tuple[int, int, float, float]] = {}
        for e in entries:
            try:
                t0, t1, lo, hi = int(e[0]), int(e[1]), float(e[2]), float(e[3])
            except (IndexError, TypeError, ValueError):
                continue
            if lo > hi:
                lo, hi = hi, lo
            lo_m, hi_m = lo / _CM, hi / _CM
            for t in range(max(1, t0), t1 + 1):
                if t in out:
                    _, _, plo, phi = out[t]
                    out[t] = (t0, t1, max(plo, lo_m), min(phi, hi_m))
                else:
                    out[t] = (t0, t1, lo_m, hi_m)
        return out

    @staticmethod
    def _index_cap(entries: list) -> Dict[int, float]:
        """(t, value_cm) entries -> {t: value_m}, minimum wins."""
        out: Dict[int, float] = {}
        for e in entries:
            try:
                t, v = int(e[0]), float(e[1])
            except (IndexError, TypeError, ValueError):
                continue
            if t in out:
                out[t] = min(out[t], v / _CM)
            else:
                out[t] = v / _CM
        return out

    @staticmethod
    def _jointly_feasible(work: np.ndarray,
                          x_bounds: Dict[int, Tuple[float, float]],
                          y_bounds: Dict[int, Tuple[float, float]],
                          y_ranges: Dict[int, Tuple[int, int, float, float]],
                          x_ranges: Dict[int, Tuple[int, int, float, float]],
                          v_max: Dict[int, float],
                          v_min: Dict[int, float],
                          a_max: Dict[int, float],
                          tol_m: float) -> bool:
        """True when every constraint family holds at once (P-23).

        Mirrors the projection semantics: v_min is skipped for stopped
        segments (a floor the projections cannot invent), and a_max
        falls back to the previous timestep's cap exactly as
        `_clamp_accel` does.
        """
        for t in range(1, len(work)):
            if t in x_bounds:
                lo, hi = x_bounds[t]
                if work[t, 0] < lo - tol_m or work[t, 0] > hi + tol_m:
                    return False
            if t in y_bounds:
                lo, hi = y_bounds[t]
                if work[t, 1] < lo - tol_m or work[t, 1] > hi + tol_m:
                    return False
            if t in y_ranges:
                t0, t1, lo, hi = y_ranges[t]
                if work[t, 1] < lo - tol_m or work[t, 1] > hi + tol_m:
                    return False
            if t in x_ranges:
                t0, t1, lo, hi = x_ranges[t]
                if work[t, 0] < lo - tol_m or work[t, 0] > hi + tol_m:
                    return False
            seg = float(np.linalg.norm(work[t] - work[t - 1]))
            if t in v_max and v_max[t] >= 0 and seg > v_max[t] + tol_m:
                return False
            if seg > 0 and t in v_min and v_min[t] > 0 \
                    and seg < v_min[t] - tol_m:
                return False
        for t in range(2, len(work)):
            dv = float(np.linalg.norm(
                (work[t] - work[t - 1]) - (work[t - 1] - work[t - 2])))
            cap = a_max.get(t, a_max.get(t - 1, None))
            if cap is not None and cap >= 0 and dv > cap + tol_m:
                return False
        return True

    @staticmethod
    def _clip_boxes(work: np.ndarray,
                    x_bounds: Dict[int, Tuple[float, float]],
                    y_bounds: Dict[int, Tuple[float, float]]) -> None:
        """Clip points into allowed boxes (in place, skips T=0)."""
        for t in range(1, len(work)):
            if t in x_bounds:
                lo, hi = x_bounds[t]
                if lo <= hi:
                    work[t, 0] = min(max(work[t, 0], lo), hi)
            if t in y_bounds:
                lo, hi = y_bounds[t]
                if lo <= hi:
                    work[t, 1] = min(max(work[t, 1], lo), hi)

    @staticmethod
    def _project_speeds(work: np.ndarray, v_max: Dict[int, float],
                        v_min: Dict[int, float]) -> None:
        """Rescale segments to obey per-step speed caps/floors (in place).

        v_max[t] caps |p[t]-p[t-1]|; v_min[t] floors it (skipped when the
        segment has no direction, i.e. a stop the cap cannot invent).
        """
        for t in range(1, len(work)):
            seg = work[t] - work[t - 1]
            dist = float(np.linalg.norm(seg))
            if dist <= 0:
                continue
            if t in v_max and v_max[t] >= 0 and dist > v_max[t]:
                work[t] = work[t - 1] + seg * (v_max[t] / dist)
                seg = work[t] - work[t - 1]
                dist = float(np.linalg.norm(seg))
            if t in v_min and v_min[t] > 0 and dist < v_min[t]:
                work[t] = work[t - 1] + seg * (v_min[t] / dist)

    @staticmethod
    def _project_moving_boxes(work: np.ndarray,
                              moving_boxes: List[Dict]) -> None:
        """P-23 increment 2: project onto moving box corridors.

        For each moving box (consecutive timesteps with linearly moving
        center), shift the entire segment as a rigid body + lateral
        offset to stay within the moving box. This couples the
        independent per-timestep projections and prevents the
        accel-clamp vs box whack-a-mole on moving threats.
        """
        for mb in moving_boxes:
            t0, t1 = mb['t0'], mb['t1']
            if t1 <= t0 or t1 >= len(work):
                continue
            center0, center1 = mb['center0'], mb['center1']
            half_w = mb['half_width']
            # Compute the ideal center at each timestep (linear interpolation)
            centers = np.linspace(center0, center1, t1 - t0 + 1)
            # Current segment y positions
            seg_y = work[t0:t1 + 1, 1]
            # Current offset from ideal center at each timestep
            offsets = seg_y - centers
            # Best constant offset that keeps all points within half_width
            # We want to find a constant offset d such that
            # centers[i] + offsets[i] + d is within [-half_w, half_w] of centers[i]
            # i.e., -half_w <= offsets[i] + d <= half_w for all i
            # So d must be in [max(-half_w - offsets), min(half_w - offsets)]
            lo = max(-half_w - o for o in offsets)
            hi = min(half_w - o for o in offsets)
            if lo <= hi:
                # Apply the midpoint of feasible range to minimize L2 change
                d = (lo + hi) * 0.5
                work[t0:t1 + 1, 1] = centers + offsets + d

    @staticmethod
    def _project_ranges(work: np.ndarray,
                        x_ranges: Dict[int, Tuple[int, int, float, float]],
                        y_ranges: Dict[int, Tuple[int, int, float, float]]) -> None:
        """P-23 increment 3: project onto coupled range corridors.

        For each timestep, apply the tightest range constraint covering it.
        Range constraints couple timesteps by enforcing a single lateral/
        longitudinal offset across the entire window.
        """
        # Y-range corridors (lateral)
        for t in range(1, len(work)):
            if t in y_ranges:
                t0, t1, lo, hi = y_ranges[t]
                lo_shift = max(lo - work[tt, 1] for tt in range(t0, min(t1 + 1, len(work))))
                hi_shift = min(hi - work[tt, 1] for tt in range(t0, min(t1 + 1, len(work))))
                if lo_shift <= hi_shift:
                    d = (lo_shift + hi_shift) * 0.5
                    for tt in range(max(1, t0), min(t1 + 1, len(work))):
                        work[tt, 1] += d

        # X-range corridors (longitudinal)
        for t in range(1, len(work)):
            if t in x_ranges:
                t0, t1, lo, hi = x_ranges[t]
                lo_shift = max(lo - work[tt, 0] for tt in range(t0, min(t1 + 1, len(work))))
                hi_shift = min(hi - work[tt, 0] for tt in range(t0, min(t1 + 1, len(work))))
                if lo_shift <= hi_shift:
                    d = (lo_shift + hi_shift) * 0.5
                    for tt in range(max(1, t0), min(t1 + 1, len(work))):
                        work[tt, 0] += d

    @staticmethod
    def _clamp_accel(work: np.ndarray, a_max: Dict[int, float]) -> None:
        """Clamp per-step velocity change magnitude (forward pass).

        a_max[t] bounds |v[t]-v[t-1]| where v[t]=p[t]-p[t-1]; enforced
        by re-targeting p[t]. Single forward pass per outer iteration;
        converges jointly with the box/speed projections.
        """
        for t in range(2, len(work)):
            v_prev = work[t - 1] - work[t - 2]
            v_cur = work[t] - work[t - 1]
            dv = v_cur - v_prev
            mag = float(np.linalg.norm(dv))
            cap = a_max.get(t, a_max.get(t - 1, None))
            if cap is None or cap < 0 or mag <= cap:
                continue
            v_new = v_prev + dv * (cap / mag)
            work[t] = work[t - 1] + v_new

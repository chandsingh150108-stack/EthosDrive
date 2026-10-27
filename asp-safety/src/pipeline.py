"""End-to-End Safety Layer Pipeline.

This module provides the SafetyLayerPipeline class that connects
all components into a complete processing pipeline.
"""

import time
import numpy as np
from typing import Dict, List, Optional

# Import components
try:
    from neural_planner import SparseDrivePlanner
    from checker import TrajectoryChecker, STEP_DT
    from corrector import TrajectoryCorrector, CorrectorQP
    from data_loader import NuScenesLoader
    from evaluation import MetricsCalculator
except ImportError:
    from src.neural_planner import SparseDrivePlanner
    from src.checker import TrajectoryChecker, STEP_DT
    from src.corrector import TrajectoryCorrector, CorrectorQP
    from src.data_loader import NuScenesLoader
    from src.evaluation import MetricsCalculator


class SafetyLayerPipeline:
    """Complete safety layer pipeline.
    
    This class connects SparseDrive neural planner with ASP safety
    checker to provide trajectory planning with safety guarantees.
    
    Attributes:
        planner: SparseDrivePlanner instance
        checker: TrajectoryChecker instance
        corrector: TrajectoryCorrector instance
    """
    
    def __init__(
        self,
        planner_path: str,
        rules_dir: str = "rules",
        device: str = "cuda",
        on_failure: str = "raise",
        corrector_mode: str = "heuristic",
        optimizer_params: Optional[Dict] = None,
    ):
        """Initialize the pipeline.

        Args:
            planner_path: Path to SparseDrive weights
            rules_dir: Path to ASP rule files
            device: "cuda" or "cpu"
            on_failure: 'raise' (corrector raises when uncorrectable)
                or 'brake' (flagged emergency hold instead)
            corrector_mode: 'heuristic' (default, TrajectoryCorrector
                fixpoint) or 'qp' (CorrectorQP: constrained solve +
                re-validation, falls back to heuristic on failure).
            optimizer_params: Optimizer knobs for 'qp' mode (lr,
                steps, tol_cm, lane_pull); ignored otherwise.
        """
        if corrector_mode not in ('heuristic', 'qp'):
            raise ValueError(
                "corrector_mode must be 'heuristic' or 'qp'")
        self.planner = SparseDrivePlanner(planner_path, device)
        self.checker = TrajectoryChecker(rules_dir)
        self.corrector_mode = corrector_mode
        if corrector_mode == 'qp':
            self.corrector = CorrectorQP(
                self.checker, optimizer_params=optimizer_params,
                on_failure=on_failure)
        else:
            self.corrector = TrajectoryCorrector(self.checker,
                                                 on_failure=on_failure)
    
    def run(
        self,
        images: np.ndarray,
        intrinsics: np.ndarray,
        extrinsics: np.ndarray,
        ego2global: np.ndarray,
        timestamp: float,
        map_data: Optional[Dict] = None,
        traffic_lights: Optional[Dict] = None,
        zones: Optional[List[Dict]] = None,
        pedestrians: Optional[List[Dict]] = None,
        weather: Optional[Dict] = None,
        ego_intent: Optional[Dict] = None,
        extras: Optional[Dict] = None
    ) -> Dict:
        """Run full pipeline on a single frame.

        Args:
            images: Camera images, shape (6, 3, 256, 704)
            intrinsics: Camera intrinsics, shape (6, 3, 3)
            extrinsics: Camera extrinsics, shape (6, 4, 4)
            ego2global: Vehicle-to-global transform, shape (4, 4)
            timestamp: Unix timestamp
            map_data: Optional map data dictionary
            traffic_lights: Optional traffic light states
            zones: Optional school/construction zones (R1.3, R9.2, R11.4)
            pedestrians: Optional VRU list (R1.5, R6.5, R9.x)
            weather: Optional environmental context (R14.x)
            ego_intent: Optional turn/maneuver/lead info (R2.2, R3.x, ...)
            extras: Optional raw ASP facts passthrough
            
        Returns:
            Dict with keys:
                'original_trajectory': np.ndarray, shape (6, 2)
                    Native planner proposal (6x0.5 s, NOT checked).
                'safe_trajectory': np.ndarray, shape (4, 2)
                    Checked/corrected path on the STEP_DT grid
                    (T=0 anchored + true +1/+2/+3 s, i.e. 3 s horizon).
                'horizon_s': int (3)
                'violations': List[Dict]
                'is_safe': bool
                'inference_time_ms': float
                'boxes': Optional[np.ndarray]
                'scores': Optional[np.ndarray]
        """
        start_time = time.time()

        # Step 1: Run SparseDrive inference
        planner_output = self.planner.predict(
            images, intrinsics, extrinsics, ego2global, timestamp
        )
        original_trajectory = planner_output['trajectory']

        # Step 1a: resample the native 6x0.5 s plan onto the checker
        # grid (N-3a, same as calibration): T=0 is the current ego
        # pose, then true +1/+2/+3 s. Checking the native grid reads
        # every speed halved.
        if original_trajectory.shape != (6, 2):
            raise ValueError(
                "planner must emit native (6, 2); got "
                f"{original_trajectory.shape}"
            )
        checked_traj = np.vstack([
            np.zeros((1, 2), dtype=np.float32),
            original_trajectory[1::2][:3].astype(np.float32),
        ])

        # Step 1b: tracked agents from planner detections (P-18). Each
        # detection becomes a constant-velocity trajectory from its
        # estimated (vx, vy) on the SAME grid (dt=STEP_DT); the checker
        # culls distant agents itself. Without this, R3/R4/R13 never see
        # real traffic (the P-5 sweep proved thresholds untunable on
        # agent-free data).
        agent_trajs = self._detections_to_trajectories(
            planner_output.get('boxes'),
            planner_output.get('velocities'),
            planner_output.get('scores'),
            len(checked_traj),
            dt=STEP_DT,
        )
        t_planner = (time.time() - start_time) * 1000.0

        # Step 2: Check with ASP safety layer
        check_kwargs = dict(
            map_data=map_data,
            traffic_lights=traffic_lights,
            zones=zones,
            pedestrians=pedestrians,
            weather=weather,
            ego_intent=ego_intent,
            extras=extras,
        )
        if agent_trajs:
            check_kwargs['other_vehicles'] = agent_trajs
        t0 = time.time()
        is_safe, violations = self.checker.check(
            checked_traj, **check_kwargs
        )
        t_checker = (time.time() - t0) * 1000.0

        # Step 3: Correct if needed
        t0 = time.time()
        if is_safe:
            safe_trajectory = checked_traj
        else:
            safe_trajectory, _ = self.corrector.correct(
                checked_traj, **check_kwargs
            )
        t_corrector = (time.time() - t0) * 1000.0 if not is_safe else 0.0

        inference_time = (time.time() - start_time) * 1000  # ms

        return {
            'original_trajectory': original_trajectory,
            'safe_trajectory': safe_trajectory,
            'horizon_s': 3,
            'corrector_mode': self.corrector_mode,
            'violations': violations,
            'is_safe': is_safe,
            # Full pipeline latency (planner+checker+corrector), not just
            # neural inference. 'pipeline_time_ms' is the canonical name;
            # 'inference_time_ms' kept for backward compatibility.
            # 'timing_ms' splits the stages (P-18).
            'inference_time_ms': inference_time,
            'pipeline_time_ms': inference_time,
            'timing_ms': {
                'planner': t_planner,
                'checker': t_checker,
                'corrector': t_corrector,
            },
            'boxes': planner_output.get('boxes', None),
            'scores': planner_output.get('scores', None)
        }

    @staticmethod
    def _detections_to_trajectories(boxes, velocities, scores,
                                    horizon: int, max_agents: int = 20,
                                    min_score: float = 0.5,
                                    dt: float = 1.0):
        """Project detections into constant-velocity agent trajectories.

        boxes: (N, 7) [x,y,z,w,l,h,yaw] in ego frame; velocities: (N, 3)
        [vx,vy,vz] or None (static); scores: (N,) confidences or None.
        dt is seconds per step (STEP_DT on the checked grid).
        Returns a list of (horizon, 2) float32 arrays. Empty list when
        no detections qualify. Score filter + agent cap bound the
        V*T grounding cross product (the 20-agent perf budget is the
        design point; noise detections on random inputs number in the
        hundreds and would blow up solve time uncorrectably).
        min_score default 0.5 is the P-5 sweep knee: 0.3 keeps
        low-confidence boxes that false-positive on log trajectories
        (GT-FP 8 scenes), 0.7 eats true pred signal (collision 16->4);
        0.5 holds pred hits steady while cutting GT-FP to 5 scenes.
        """
        if boxes is None or len(boxes) == 0 or horizon <= 0:
            return []
        order = range(len(boxes))
        if scores is not None and len(scores) == len(boxes):
            order = [i for i in order if float(scores[i]) >= min_score]
            order = sorted(order, key=lambda i: -float(scores[i]))
        trajs = []
        for i in order[:max_agents]:
            box = boxes[i]
            try:
                x, y = float(box[0]), float(box[1])
                if velocities is not None and len(velocities) > i:
                    vx, vy = float(velocities[i][0]), float(velocities[i][1])
                else:
                    vx, vy = 0.0, 0.0
            except (TypeError, ValueError, IndexError):
                continue
            steps = np.arange(horizon, dtype=np.float32) * dt
            trajs.append(np.column_stack([
                x + vx * steps, y + vy * steps]).astype(np.float32))
        return trajs
    
    def run_batch(
        self,
        data_loader: NuScenesLoader,
        scene_indices: List[int],
        zones: Optional[Dict] = None,
        pedestrians: Optional[Dict] = None,
        weather: Optional[Dict] = None,
        ego_intent: Optional[Dict] = None,
        extras: Optional[Dict] = None,
    ) -> List[Dict]:
        """Run pipeline on multiple scenes.
        
        Args:
            data_loader: Initialized NuScenesLoader
            scene_indices: List of scene indices to process
            zones: Optional per-scene zones {scene_idx: zones-list} or
                a single zones-list applied to all scenes
            pedestrians: Optional per-scene pedestrians (same forms)
            weather: Optional per-scene weather (same forms)
            ego_intent: Optional per-scene intent (same forms)
            extras: Optional per-scene extras (same forms)
            
        Returns:
            List of result dictionaries (same schema as run(), plus
            'scene_idx'; error dicts carry is_safe=False and empty
            violations/trajectories so downstream code keeps its keys)
        """
        results = []
        
        def _per_scene(value, idx):
            if value is None:
                return None
            if isinstance(value, dict):
                # Per-scene mapping {scene_idx: payload}; scenes not
                # listed get None (never the whole mapping).
                return value.get(idx)
            return value

        for idx in scene_indices:
            try:
                # Load scene data
                sample = data_loader.get_scene_sample(idx)
                map_data = data_loader.get_map_data(idx)
                traffic_lights = data_loader.get_traffic_lights(idx)
                
                # Run pipeline
                result = self.run(
                    sample['images'],
                    sample['intrinsics'],
                    sample['extrinsics'],
                    sample['ego2global'],
                    sample['timestamp'],
                    map_data=map_data,
                    traffic_lights=traffic_lights,
                    zones=_per_scene(zones, idx),
                    pedestrians=_per_scene(pedestrians, idx),
                    weather=_per_scene(weather, idx),
                    ego_intent=_per_scene(ego_intent, idx),
                    extras=_per_scene(extras, idx),
                )
                
                result['scene_idx'] = idx
                results.append(result)
                
            except Exception as e:
                print(f"Error processing scene {idx}: {e}")
                # Stable error schema (P-18/T-11): same keys as a
                # success result, Nones where no data exists, so
                # downstream code never branches on key presence.
                results.append({
                    'scene_idx': idx,
                    'error': str(e),
                    'is_safe': False,
                    'violations': [],
                    'original_trajectory': None,
                    'safe_trajectory': None,
                    'corrector_mode': self.corrector_mode,
                    'inference_time_ms': 0.0,
                    'pipeline_time_ms': 0.0,
                    'timing_ms': None,
                    'boxes': None,
                    'scores': None,
                })
        
        return results

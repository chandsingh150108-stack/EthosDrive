"""Evaluation Metrics for ASP Safety Layer.

This module provides the MetricsCalculator class for evaluating
trajectory planning performance.
"""

import numpy as np
from typing import List, Dict, Tuple


class MetricsCalculator:
    """Calculate evaluation metrics for trajectory planning.
    
    This class provides static methods for calculating various
    metrics used to evaluate autonomous driving performance.
    """
    
    @staticmethod
    def collision_rate(
        trajectories: List[np.ndarray],
        gt_boxes: List[np.ndarray],
        pred_boxes: List[np.ndarray]
    ) -> float:
        """Calculate collision rate.
        
        Args:
            trajectories: List of predicted trajectories
                Each shape: (T, 2)
            gt_boxes: List of ground truth bounding boxes
                Each shape: (N, 7) - [x, y, z, w, l, h, yaw]
                (canonical layout, same as SparseDrivePlanner)
            pred_boxes: List of predicted bounding boxes
                Each shape: (M, 7) - [x, y, z, w, l, h, yaw]
                
        Returns:
            float: Collision rate as percentage [0.0, 100.0]

        Formula:
            collision_rate = (num_collisions / total_trajectories) * 100

        Boxes are static single-frame detections (no timestamps);
        trajectory index is the time axis. Moving-agent awareness
        lives in the checker's predicted_pos path, not here.

        Raises:
            ValueError: If the three lists have different lengths (the
                old zip-truncation silently dropped scenes).
        """
        if not trajectories:
            return 0.0

        if not (len(trajectories) == len(gt_boxes) == len(pred_boxes)):
            raise ValueError(
                f"mismatched batch lengths: {len(trajectories)}, "
                f"{len(gt_boxes)}, {len(pred_boxes)}"
            )

        collisions = 0
        for traj, gt, pred in zip(trajectories, gt_boxes, pred_boxes):
            if MetricsCalculator._check_collision(traj, gt, pred):
                collisions += 1

        return collisions / len(trajectories) * 100
    
    @staticmethod
    def l2_error(
        pred_trajectory: np.ndarray,
        gt_trajectory: np.ndarray
    ) -> float:
        """Calculate L2 displacement error.
        
        Args:
            pred_trajectory: Predicted trajectory, shape (T, 2)
            gt_trajectory: Ground truth trajectory, shape (T', 2)
            
        Returns:
            float: Average L2 error in meters
            
        Formula:
            l2_error = mean(sqrt((pred_x - gt_x)^2 + (pred_y - gt_y)^2))

        Raises:
            ValueError: On non-(T, 2) shapes or non-finite values
                (NaN/inf previously propagated silently into means).
        """
        pred_trajectory = np.asarray(pred_trajectory, dtype=float)
        gt_trajectory = np.asarray(gt_trajectory, dtype=float)
        for name, arr in (("pred_trajectory", pred_trajectory),
                          ("gt_trajectory", gt_trajectory)):
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    f"{name} must have shape (T, 2), got {arr.shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN/inf")

        min_len = min(len(pred_trajectory), len(gt_trajectory))

        if min_len == 0:
            return 0.0

        return float(np.mean(np.linalg.norm(
            pred_trajectory[:min_len] - gt_trajectory[:min_len], axis=1
        )))
    
    @staticmethod
    def _matches(true_v: Dict, detected_v: Dict) -> bool:
        """Match two violation dicts (P-17).

        Rule + timestep must agree; vehicle/vehicle2/type must agree
        whenever BOTH dicts carry the key (hand-made legacy dicts
        without them still match by rule+time).
        """
        if (true_v.get('rule') != detected_v.get('rule')
                or true_v.get('time') != detected_v.get('time')):
            return False
        for key in ('vehicle', 'vehicle2', 'type'):
            if key in true_v and key in detected_v:
                if true_v[key] != detected_v[key]:
                    return False
        return True

    @staticmethod
    def violation_detection_rate(
        true_violations: List[Dict],
        detected_violations: List[Dict]
    ) -> float:
        """Calculate true positive rate for violation detection.
        
        Args:
            true_violations: List of actual violations
            detected_violations: List of detected violations
            
        Returns:
            float: Detection rate [0.0, 1.0]
            
        Formula:
            detection_rate = true_positives / total_true_violations
        """
        if not true_violations:
            return 1.0 if not detected_violations else 0.0
        
        detected = 0
        for tv in true_violations:
            for dv in detected_violations:
                if MetricsCalculator._matches(tv, dv):
                    detected += 1
                    break

        return detected / len(true_violations)

    @staticmethod
    def false_discovery_rate(
        true_violations: List[Dict],
        detected_violations: List[Dict]
    ) -> float:
        """Calculate false discovery rate for violation detection.

        FDR = false_positives / total_detected_violations (the
        statistically correct name; false_positive_rate is kept as an
        alias for backward compatibility).

        Args:
            true_violations: List of actual violations
            detected_violations: List of detected violations

        Returns:
            float: False discovery rate [0.0, 1.0]
        """
        if not detected_violations:
            return 0.0

        false_positives = 0
        for dv in detected_violations:
            is_true = False
            for tv in true_violations:
                if MetricsCalculator._matches(tv, dv):
                    is_true = True
                    break
            if not is_true:
                false_positives += 1

        return false_positives / len(detected_violations)

    @staticmethod
    def false_positive_rate(
        true_violations: List[Dict],
        detected_violations: List[Dict]
    ) -> float:
        """Alias of false_discovery_rate (backward compatibility)."""
        return MetricsCalculator.false_discovery_rate(
            true_violations, detected_violations)
    
    @staticmethod
    def average_inference_time(inference_times: List[float]) -> float:
        """Calculate average inference time.
        
        Args:
            inference_times: List of inference times in milliseconds
            
        Returns:
            float: Average inference time in ms
        """
        if not inference_times:
            return 0.0
        return float(np.mean(inference_times))
    
    @staticmethod
    def _check_collision(
        traj: np.ndarray,
        gt_boxes: np.ndarray,
        pred_boxes: np.ndarray
    ) -> bool:
        """Check if trajectory collides with any box.
        
        Args:
            traj: Trajectory (T, 2)
            gt_boxes: Ground truth boxes (N, 7)
            pred_boxes: Predicted boxes (M, 7)
            
        Returns:
            bool: True if collision detected
        """
        for t, point in enumerate(traj):
            for box in gt_boxes:
                if MetricsCalculator._point_in_box(point, box):
                    return True
            for box in pred_boxes:
                if MetricsCalculator._point_in_box(point, box):
                    return True
        return False
    
    @staticmethod
    def _point_in_box(point: np.ndarray, box: np.ndarray) -> bool:
        """Check if 2D point is inside a yawed bounding box (P-17).

        Args:
            point: (x, y) point
            box: Bounding box [x, y, z, w, l, h, yaw]
                Footprint uses l (x-extent) x w (y-extent); h is
                vertical and ignored. Yaw rotates the footprint about
                z (right-hand rule); the point is tested in the box
                frame. Boxes shorter than 7 elements never match.

        Returns:
            bool: True if point is inside box
        """
        if len(box) < 7:
            return False

        import math
        x, y = float(point[0]), float(point[1])
        bx, by = float(box[0]), float(box[1])
        bw, bl = float(box[3]), float(box[4])
        yaw = float(box[6])

        dx, dy = x - bx, y - by
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        lx, ly = dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y

        return abs(lx) <= bl / 2 and abs(ly) <= bw / 2

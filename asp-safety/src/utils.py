"""Utility functions for ASP Safety Layer."""

import numpy as np
from typing import List, Tuple, Dict


def calculate_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two points.
    
    Args:
        p1: (x, y) coordinates of first point
        p2: (x, y) coordinates of second point
        
    Returns:
        Distance between points
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def calculate_velocity(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Calculate velocity from trajectory.
    
    Args:
        trajectory: (T, 2) array of positions
        dt: Time step between positions
        
    Returns:
        (T, 2) array of velocities
    """
    velocities = np.zeros_like(trajectory)
    velocities[1:] = (trajectory[1:] - trajectory[:-1]) / dt
    return velocities


def calculate_speed(trajectory: np.ndarray, dt: float = 1.0) -> np.ndarray:
    """Calculate speed from trajectory.
    
    Args:
        trajectory: (T, 2) array of positions
        dt: Time step between positions
        
    Returns:
        (T,) array of speeds
    """
    velocities = calculate_velocity(trajectory, dt)
    speeds = np.sqrt(velocities[:, 0]**2 + velocities[:, 1]**2)
    return speeds


def check_collision_point_box(
    point: Tuple[float, float],
    box_center: Tuple[float, float],
    box_size: Tuple[float, float]
) -> bool:
    """Check if a point is inside a bounding box.
    
    Args:
        point: (x, y) coordinates
        box_center: (x, y) center of box
        box_size: (width, height) of box
        
    Returns:
        True if point is inside box
    """
    px, py = point
    cx, cy = box_center
    w, h = box_size
    
    return (cx - w/2 <= px <= cx + w/2 and
            cy - h/2 <= py <= cy + h/2)


def interpolate_trajectory(
    trajectory: np.ndarray,
    num_points: int
) -> np.ndarray:
    """Interpolate trajectory to have specified number of points.
    
    Args:
        trajectory: (T, 2) original trajectory
        num_points: Desired number of points
        
    Returns:
        (num_points, 2) interpolated trajectory
    """
    t_original = np.linspace(0, 1, len(trajectory))
    t_interpolated = np.linspace(0, 1, num_points)
    
    x_interp = np.interp(t_interpolated, t_original, trajectory[:, 0])
    y_interp = np.interp(t_interpolated, t_original, trajectory[:, 1])
    
    return np.column_stack([x_interp, y_interp])


def trajectory_to_facts(
    vehicle_id: str,
    trajectory: np.ndarray
) -> List[str]:
    """Convert trajectory to ASP facts.
    
    Args:
        vehicle_id: Identifier for the vehicle
        trajectory: (T, 2) array of positions
        
    Returns:
        List of ASP fact strings
    """
    facts = []
    for t, (x, y) in enumerate(trajectory):
        facts.append(f"pos({vehicle_id}, {t}, {x:.4f}, {y:.4f}).")
    facts.append(f"vehicle({vehicle_id}).")
    return facts


def format_violations(violations: List[Dict]) -> str:
    """Format violations for display.
    
    Args:
        violations: List of violation dictionaries
        
    Returns:
        Formatted string
    """
    if not violations:
        return "No violations detected."
    
    lines = [f"Violations detected ({len(violations)}):"]
    for v in violations:
        lines.append(f"  - {v['rule']} at t={v['time']} for {v['vehicle']}")
    return "\n".join(lines)

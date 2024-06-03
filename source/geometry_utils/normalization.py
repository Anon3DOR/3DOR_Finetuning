import numpy as np


def normalize_points_centroid(points: np.ndarray) -> np.ndarray:
    """Normalize point cloud to its center of mass and unit sphere."""

    if points.shape[1] != 3 or len(points.shape) != 2:
        raise ValueError("Points must be [N, 3].")

    centroid = np.mean(points, axis=0)
    points -= centroid
    furthest_distance = np.sqrt(np.max(np.sum(points**2, axis=-1)))
    points /= furthest_distance
    return points, furthest_distance

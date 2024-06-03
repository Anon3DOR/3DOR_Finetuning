"""
    Test visually in notebook with:

    from source.geometry_utils import augmentation
    from source.geometry_utils import visualizer
    import trimesh
    import numpy as np

    mesh_orig = trimesh.creation.capsule(radius=0.5, height=2.0)
    points_orig = trimesh.sample.sample_surface(mesh_orig, 1024)[0]
    points_orig = np.clip(points_orig, -100, 0.)
    points_scaled = augmentation.stretch_points_normals(points_orig.copy(), (0.28, 2.))
    points_jittered = augmentation.jitter_points(points_orig.copy(), 0.02, 0.1)
    points_shifted = augmentation.shift_point_cloud(points_orig.copy(), 0.3)
    points_flipped = augmentation.flip_point_cloud(points_scaled.copy(), xz=True)
    viz = visualizer.Visualizer()
    viz.add_pointcloud(points_orig)
    viz.add_pointcloud(points_flipped)
    viz.show()
"""

import numpy as np


class Augmentation(object):

    def __init__(self, disable: bool = False):
        self.disable = disable  # Facilitates hyperparam search.

    def __call__(self, points: np.ndarray):

        if self.disable:
            return points
        else:
            return self._augment(points)

    def _augment(self, points: np.ndarray):
        raise NotImplementedError('Augmentation must implement _augment method.')


class Stretch(Augmentation):

    def __init__(self, scale_min: float, scale_max: float, **kwargs):
        super().__init__(**kwargs)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def _augment(self, points: np.ndarray):
        """Stretch points and normals by a given scale in each dimension.
        
        Inputs:
            points: (N, 3) or (N, 6) array of points (and normals).
            scale_range: (min, max) allowed uniform distribution for scale.
        Outputs:
            points: (N, 3) or (N, 6) stretched points (and normals).
        """

        scale = np.random.uniform(self.scale_min, self.scale_max, size=(3,))

        # Stretch points.
        points[:, :3] *= scale

        # Stretch normals and renormalize.
        if points.shape[1] == 6:
            points[:, 3:] *= scale
            points[:, 3:] /= np.linalg.norm(points[:, 3:], axis=1, keepdims=True)

        return points


class Scale(Augmentation):

    def __init__(self, scale_min: float, scale_max: float, **kwargs):
        super().__init__(**kwargs)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def _augment(self, points: np.ndarray):
        "Scale points with a global scale."

        scale = np.random.uniform(self.scale_min, self.scale_max, size=(1,))
        points[:, :3] *= scale
        return points


class Rotate(Augmentation):

    def __init__(self, x: bool = False, y: bool = False, z: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.z = z

    def _augment(self, points: np.ndarray):
        """Rotate points around the x, y, and z axes by a random angle."""

        # Generate random rotation matrix.
        R = np.eye(3)
        if self.x:
            theta_x = np.random.uniform(0, 2 * np.pi)
            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(theta_x), -np.sin(theta_x)],
                [0, np.sin(theta_x), np.cos(theta_x)],
            ])
            R = Rx @ R
        if self.y:
            theta_y = np.random.uniform(0, 2 * np.pi)
            Ry = np.array([
                [np.cos(theta_y), 0, np.sin(theta_y)],
                [0, 1, 0],
                [-np.sin(theta_y), 0, np.cos(theta_y)],
            ])
            R = Ry @ R
        if self.z:
            theta_z = np.random.uniform(0, 2 * np.pi)
            Rz = np.array([
                [np.cos(theta_z), -np.sin(theta_z), 0],
                [np.sin(theta_z), np.cos(theta_z), 0],
                [0, 0, 1],
            ])
            R = Rz @ R

        # Rotate points.
        points[:, :3] = points[:, :3] @ R.T

        # Rotate normals.
        if points.shape[1] == 6:
            points[:, 3:] = points[:, 3:] @ R.T

        return points


class Jitter(Augmentation):

    def __init__(self, sigma: float, clip: float, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.clip = clip

    def _augment(self, points: np.ndarray):

        points[:, :3] += np.clip(self.sigma * np.random.randn(*points[:, :3].shape), -self.clip,
                                 self.clip)
        return points


class Shift(Augmentation):

    def __init__(self, shift_max: float, **kwargs):
        super().__init__(**kwargs)
        self.shift_max = shift_max

    def _augment(self, points: np.ndarray):
        """Shift points by a random amount in each dimension.
        
        Inputs:
            points: (N, 3) or (N, 6) array of points (and normals).
            shift_range: (min, max) allowed uniform distribution for .
        Outputs:
            points: (N, 3) or (N, 6) shifted points (and normals).
        """

        shift = np.random.uniform(-self.shift_max, self.shift_max, size=(3,))
        points[:, :3] += shift
        return points


class Flip(Augmentation):

    def __init__(self, xy: bool = False, yz: bool = False, xz: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.xy = xy
        self.yz = yz
        self.xz = xz

    def _augment(self, points: np.ndarray):

        flipped = False
        if self.xy and np.random.random() > 0.5 and not flipped:
            points[:, 2] *= -1
            flipped = True
            if points.shape[1] == 6:
                points[:, 5] *= -1
        if self.yz and np.random.random() > 0.5 and not flipped:
            points[:, 0] *= -1
            flipped = True
            if points.shape[1] == 6:
                points[:, 3] *= -1
        if self.xz and np.random.random() > 0.5 and not flipped:
            points[:, 1] *= -1
            flipped = True
            if points.shape[1] == 6:
                points[:, 4] *= -1

        return points

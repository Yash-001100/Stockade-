"""Minimal local stand-in for nuscenes.utils.data_classes.

The real nuscenes-devkit==1.0.9 pulls in a pile of unrelated dependencies
(pandas, motmetrics, fire, jupyter, an old Pillow<=6.2.1 pin) just to reach
this one geometry class. coperception's training path only ever uses
Box(center, size, orientation).corners(), so that's all this reimplements,
copied faithfully from nuscenes-devkit's Box class.
"""

import numpy as np
from pyquaternion import Quaternion


class Box:
    def __init__(
        self,
        center,
        size,
        orientation: Quaternion,
        label: int = np.nan,
        score: float = np.nan,
        velocity=(np.nan, np.nan, np.nan),
        name: str = None,
        token: str = None,
    ):
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert type(orientation) == Quaternion

        self.center = np.array(center)
        self.wlh = np.array(size)
        self.orientation = orientation
        self.label = int(label) if not np.isnan(label) else label
        self.score = float(score) if not np.isnan(score) else score
        self.velocity = np.array(velocity)
        self.name = name
        self.token = token

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self.orientation.rotation_matrix

    def translate(self, x) -> None:
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation
        self.velocity = np.dot(quaternion.rotation_matrix, self.velocity)

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        w, l, h = self.wlh * wlh_factor

        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        corners = np.dot(self.orientation.rotation_matrix, corners)

        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

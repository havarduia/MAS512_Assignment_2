# Camera with an intrinsic matrix: back projection of a pixel with depth to the camera frame

import json

import numpy as np


class Camera:
    def __init__(self, name, M):
        self.name = name
        self.M = M  # 3x3 intrinsic matrix

    @classmethod
    def from_json(cls, name, path):
        # intrinsics saved from the L515 in the lab
        intr = json.loads(path.read_text())
        M = np.array([[intr["fx"], 0, intr["ppx"]],
                      [0, intr["fy"], intr["ppy"]],
                      [0, 0, 1]])
        return cls(name, M)

    def back_project(self, u, v, d):
        # pixel (u, v) with depth d -> camera frame, in homogeneous system: p_c = M~^-1 @ [d*u, d*v, d, 1]
        M_h = np.eye(4)
        M_h[:3, :3] = self.M
        p_c = np.linalg.inv(M_h) @ np.array([d * u, d * v, d, 1])
        return p_c[:3] / p_c[3]  # homogeneous -> cartesian

    def back_project_all(self, detections):
        return np.array([self.back_project(det.u, det.v, det.d) for det in detections])

    def field_of_view(self, width=640, height=480):
        fov_h = np.degrees(2 * np.arctan(width / (2 * self.M[0, 0])))
        fov_v = np.degrees(2 * np.arctan(height / (2 * self.M[1, 1])))
        return fov_h, fov_v

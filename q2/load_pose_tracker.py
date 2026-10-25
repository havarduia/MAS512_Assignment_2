# Pose of the load in consecutive lidar frames with KISS-ICP

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP


class LoadPoseTracker:
    def __init__(self, voxel_size=0.02, max_range=5.0):
        config = KISSConfig()
        config.data.deskew = False                # no per-point timestamps in our clouds
        config.data.max_range = max_range         # the load is ~2 m from the lidar
        config.mapping.voxel_size = voxel_size    # default (max_range / 100) is too coarse for a 0.5 m load
        config.adaptive_threshold.initial_threshold = 0.2
        config.adaptive_threshold.min_motion_th = 0.01
        self.icp = KissICP(config)

    def track(self, load_clouds, centre_0):
        # KISS-ICP estimates the pose of the "sensor" relative to the first frame. Here only the load's points
        # are registered and the lidar stands still, so that pose is the inverse of the load's motion M_k.
        # Pose of the load centre in frame k: rotation of M_k, position M_k @ centre in frame 0.
        poses = []
        for points in load_clouds:
            self.icp.register_frame(points, np.zeros(len(points)))
            M = np.linalg.inv(self.icp.last_pose)

            T = np.eye(4)
            T[:3, :3] = M[:3, :3]
            T[:3, 3] = (M @ np.append(centre_0, 1))[:3]
            poses.append(T)
        return np.array(poses)

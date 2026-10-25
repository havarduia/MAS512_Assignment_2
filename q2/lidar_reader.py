# Read the Ouster lidar recording (.pcap + .json metadata) as one point cloud per frame

import numpy as np
from ouster.sdk import core, open_source


class LidarReader:
    def __init__(self, pcap_path, json_path):
        self.source = open_source(str(pcap_path), meta=[str(json_path)])
        self.xyz_lut = core.XYZLut(self.source.sensor_info[0])  # range image -> x, y, z [m]

    def frames(self):
        clouds = []
        for scans in self.source:
            scan = scans[0]
            if scan is None:
                continue
            points = self.xyz_lut(scan.field(core.ChanField.RANGE)).reshape(-1, 3)
            points = points[np.linalg.norm(points, axis=1) > 0]  # 0 = no return
            if len(points) > 0:  # first and last frame of the recording are incomplete
                clouds.append(points)
        return clouds

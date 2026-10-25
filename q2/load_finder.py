# Find the swinging load in a point cloud: crop the area around it, DBSCAN, keep the largest cluster

import numpy as np
import open3d as o3d


class LoadFinder:
    def __init__(self, box_min, box_max, eps=0.1, min_points=20):
        self.box = o3d.geometry.AxisAlignedBoundingBox(box_min, box_max)
        self.eps = eps                # max distance between two neighbouring points of a cluster [m]
        self.min_points = min_points  # min neighbours for a core point

    def clusters(self, points):
        # the points inside the box and their DBSCAN cluster label (-1 = noise)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.crop(self.box)
        labels = np.array(pcd.cluster_dbscan(eps=self.eps, min_points=self.min_points))
        return np.asarray(pcd.points), labels

    def find(self, points):
        # returns the points of the load (N x 3): the largest cluster
        cropped, labels = self.clusters(points)
        largest = np.bincount(labels[labels >= 0]).argmax()
        return cropped[labels == largest]

    @staticmethod
    def centre(load_points):
        return np.mean(load_points, axis=0)  # centroid of the cluster

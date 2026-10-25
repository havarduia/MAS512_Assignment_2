# Centre of the swinging load in every frame: YOLO model from Assignment 1, Q1 + depth from the L515

import json
from typing import NamedTuple

import cv2
import numpy as np
from ultralytics import YOLO


class Detection(NamedTuple):
    frame: int
    u: int        # pixel column of the box centre
    v: int        # pixel row of the box centre
    d: float      # depth [m]
    box: tuple    # x0, y0, x1, y1 [px], for the presentation figures


class LoadDetector:
    def __init__(self, model_path, data_dir):
        self.model = YOLO(str(model_path))
        self.data_dir = data_dir
        self.depth_scale = json.loads((data_dir / "depth_scale_json/depth_scale.json").read_text())["depth_scale"]

    def detect_all(self):
        detections = []
        for fname in sorted((self.data_dir / "rgb_frames").glob("rgb_*.png")):
            det = self.detect(int(fname.stem.split("_")[1]), fname)
            if det is not None:
                detections.append(det)
        return detections

    def detect(self, i, fname):
        boxes = self.model(cv2.imread(str(fname)), verbose=False)[0].boxes
        if len(boxes) == 0:
            return None
        u, v = boxes.xywh[0, :2].int().tolist()  # centre of the bounding box
        box = tuple(boxes.xyxy[0].tolist())

        d = self.depth(i, u, v)
        if d is None:
            return None
        return Detection(i, u, v, d, box)

    def depth(self, i, u, v):
        depth_image = np.load(self.data_dir / f"depth_raw/depth_{i:04d}_raw.npy")
        patch = depth_image[max(v - 5, 0):v + 6, max(u - 5, 0):u + 6]  # 11x11 px around the centre
        patch = patch[patch > 0]  # 0 = no depth reading
        if patch.size == 0:
            return None
        return np.median(patch) * self.depth_scale  # raw -> metres

#Task B(ii): detect people and cars, draw a bounding box and show the lidar distance

from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

DATA = Path(__file__).resolve().parent.parent / "data" / "training"
OUT = Path(__file__).resolve().parent.parent / "output"

model = YOLO("yolo11n.pt")  # pretrained on COCO
random.seed(0)

# get 5 random frames
frames = random.sample([p.stem for p in (DATA / "image_2").glob("*.png")], 5)

for frame in frames:
    img = cv2.imread(str(DATA / "image_2" / f"{frame}.png"))
    lidar = np.fromfile(DATA / "velodyne" / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)

    # read calibration same as in task 1
    calib = {}
    for line in open(DATA / "calib" / f"{frame}.txt"):
        if ":" in line:
            name, numbers = line.split(":", 1)
            calib[name] = np.array(numbers.split(), dtype=float)
    P2 = calib["P2"].reshape(3, 4)
    R0 = np.eye(4); R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
    Tr = np.eye(4); Tr[:3, :4] = calib["Tr_velo_to_cam"].reshape(3, 4)

    # lidar to image (sensor fusion)
    pts = np.hstack((lidar[:, :3], np.ones((len(lidar), 1))))
    cam = R0 @ Tr @ pts.T
    cam = cam[:, cam[2] > 0] # only points in front of the camera
    pix = P2 @ cam
    u, v, depth = pix[0] / pix[2], pix[1] / pix[2], cam[2]
    h, w = img.shape[:2]
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, depth = u[keep], v[keep], depth[keep]

    # detect people class 0 and cars class 2 and higher probability than 0.5
    result = model(img, conf=0.5, classes=[0, 2], verbose=False)[0]

#   plot the results: image, bounding boxes, lidar points inside the boxes and distance to the object
    plt.figure(figsize=(16, 5))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cmap = plt.cm.jet_r
    norm = plt.Normalize(0, 80)              # color scale 0-80 m
    in_any_box = np.zeros(len(u), dtype=bool)  # lidar points that are inside a box

    print("frame", frame)
    #had help from ai to write the following code
    for (x1, y1, x2, y2), cls, conf in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        in_box = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
        in_any_box |= in_box

        # centre of the box, distance from the lidar points in the middle half of the box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = x2 - x1, y2 - y1
        near_centre = (abs(u - cx) <= bw / 4) & (abs(v - cy) <= bh / 4)
        if not near_centre.any():
            continue
        dist = np.median(depth[near_centre])   # distance to the object
        color = cmap(norm(dist))               # box color = distance

        label = f"{model.names[int(cls)]} {conf:.2f} | {dist:.1f} m"
        print("  ", label, f"centre=({cx:.0f}, {cy:.0f})")

        plt.gca().add_patch(plt.Rectangle((x1, y1), bw, bh, fill=False, edgecolor=color, linewidth=2))
        plt.plot(cx, cy, "o", color=color, markeredgecolor="black")   # centre point
        plt.text(x1, y1 - 3, label, fontsize=8, backgroundcolor=color)

    # only draw the lidar points inside the boxes (x dir, y dir, c=color value, cmap=colormap, norm=normalization, s=size of the points)
    plt.scatter(u[in_any_box], v[in_any_box], c=depth[in_any_box], cmap=cmap, norm=norm, s=2)

    plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca(), label="Depth [m]")
    plt.axis("off")
    plt.savefig(OUT / f"{frame}_detection.png", bbox_inches="tight")
    plt.close()
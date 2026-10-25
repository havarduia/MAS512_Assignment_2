#Task B(iii): validate the estimated distance against the ground truth in label_2

from pathlib import Path
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
random.seed(0)

DATA = Path(__file__).resolve().parent.parent / "data" / "training"
OUT = Path(__file__).resolve().parent.parent / "output"
 # pretrained on COCO
model = YOLO("yolo11n.pt") 
# which KITTI classes match the YOLO classes
GT_CLASS = {"car": ["Car"], "person": ["Pedestrian", "Person_sitting"]}

def iou(a, b):
    # overlap between two boxes (x1, y1, x2, y2), 0 = no overlap, 1 = same box
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)

#random.seed(0)  # same 5 frames every time
frames = random.sample(sorted(p.stem for p in (DATA / "image_2").glob("*.png")), 5)

results = []  # (frame, class, estimated, ground truth)

for frame in frames:
    img = cv2.imread(str(DATA / "image_2" / f"{frame}.png"))
    lidar = np.fromfile(DATA / "velodyne" / f"{frame}.bin", dtype=np.float32).reshape(-1, 4)

    # read calibration
    calib = {}
    for line in open(DATA / "calib" / f"{frame}.txt"):
        if ":" in line:
            name, numbers = line.split(":", 1)
            calib[name] = np.array(numbers.split(), dtype=float)
    P2 = calib["P2"].reshape(3, 4)
    R0 = np.eye(4); R0[:3, :3] = calib["R0_rect"].reshape(3, 3)
    Tr = np.eye(4); Tr[:3, :4] = calib["Tr_velo_to_cam"].reshape(3, 4)

    # read ground truth: class, 2D box and distance z
    gt = []
    for line in open(DATA / "label_2" / f"{frame}.txt"):
        p = line.split()
        if p[0] != "DontCare":
            gt.append((p[0], [float(x) for x in p[4:8]], float(p[13])))

    # lidar -> image
    pts = np.hstack((lidar[:, :3], np.ones((len(lidar), 1))))
    cam = R0 @ Tr @ pts.T
    cam = cam[:, cam[2] > 0]
    pix = P2 @ cam
    u, v, depth = pix[0] / pix[2], pix[1] / pix[2], cam[2]
    h, w = img.shape[:2]
    keep = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, depth = u[keep], v[keep], depth[keep]

    # detect people class 0 and cars class 2
    result = model(img, conf=0.5, classes=[0, 2], verbose=False)[0]

    print("frame", frame)
    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        x1, y1, x2, y2 = box
        name = model.names[int(cls)]

        # estimated distance, same as Task2 (middle half of the box)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        near_centre = (abs(u - cx) <= (x2 - x1) / 4) & (abs(v - cy) <= (y2 - y1) / 4)
        if not near_centre.any():
            continue
        est = np.median(depth[near_centre])

        # find the ground truth object of the same class with the most overlap
        same_class = [g for g in gt if g[0] in GT_CLASS[name]]
        if not same_class:
            print(f"   {name:6s} est={est:5.1f} m   no ground truth")
            continue
        best = max(same_class, key=lambda g: iou(box, g[1]))
        if iou(box, best[1]) < 0.5:
            print(f"   {name:6s} est={est:5.1f} m   no ground truth")
            continue

        true_z = best[2]
        results.append((frame, name, est, true_z))
        print(f"   {name:6s} est={est:5.1f} m   gt={true_z:5.1f} m   error={est - true_z:+5.1f} m")

# KPIs over all matched objects
est = np.array([r[2] for r in results])
true = np.array([r[3] for r in results])
err = est - true

print(f"\nmatched objects: {len(results)}")
print(f"MAE  (mean absolute error): {np.mean(abs(err)):.2f} m")
print(f"RMSE:                       {np.sqrt(np.mean(err ** 2)):.2f} m")
print(f"bias (mean error):          {np.mean(err):+.2f} m")
print(f"mean relative error:        {np.mean(abs(err) / true) * 100:.1f} %")

# estimated vs ground truth
plt.figure(figsize=(6, 6))
plt.scatter(true, est)
lim = max(true.max(), est.max()) + 5
plt.plot([0, lim], [0, lim], "k--", label="perfect (y = x)")
plt.xlabel("Ground truth distance [m]")
plt.ylabel("Estimated distance [m]")
plt.title("Estimated vs ground truth distance")
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(OUT / "validation.png", bbox_inches="tight")
plt.close()
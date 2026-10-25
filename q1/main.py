# Assignment 2, Q1 - Camera
# 3D coordinates of the centre of the swinging load with M_int and M_d, ML mapping o1 -> o2*, Euclidean error

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import results
from camera import Camera
from load_detector import LoadDetector
from mapping_model import MappingModel

HERE = Path(__file__).resolve().parent
DATA = HERE / "camera_data"
RESULT = HERE / "result"
SLIDES = HERE.parent / "slides.mplstyle"  # shared slide style: 16:9, large text, thick lines


def style_3d(ax):
    # fewer ticks and more space for the labels, so a 3D axis stays readable on a slide
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(4))
        axis.labelpad = 18
    ax.tick_params(labelsize=14)
    ax.set_box_aspect(None, zoom=0.82)  # leave room for the axis labels inside the figure


def plot_trajectories(o1, o2):
    # camera frame: x right, y down, z forward -> plot x, z, -y so the load hangs down
    P_all = np.vstack([o1, o2])
    lims = [(P_all[:, 0].min(), P_all[:, 0].max()), (P_all[:, 2].min(), P_all[:, 2].max()),
            (-P_all[:, 1].max(), -P_all[:, 1].min())]  # same limits on both -> sizes can be compared

    fig = plt.figure()
    for k, (P, color, title) in enumerate([(o1, "C0", "(a) $o_1$ with $M_{int}$"), (o2, "C1", "(b) $o_2$ with $M_d$")]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        ax.plot(P[:, 0], P[:, 2], -P[:, 1], ".-", color=color, lw=1.5, ms=5)
        ax.set(xlim=lims[0], ylim=lims[1], zlim=lims[2], xlabel="x [m]", ylabel="z [m]", zlabel="−y [m]")
        ax.set_title(title, pad=0)
        style_3d(ax)
    fig.savefig(RESULT / "q1_trajectories.png")


def plot_training(model):
    # train and validation loss per epoch: both should flatten out and stay close (no over- or underfitting)
    fig, ax = plt.subplots()
    ax.semilogy(model.history["loss"], label="train")
    ax.semilogy(model.history["val_loss"], "--", label="validation")
    ax.set(title="(d) Training of $f_\\theta$: loss per epoch", xlabel="epoch", ylabel="MSE loss [m²]")
    ax.legend(loc="upper right")
    fig.savefig(RESULT / "q1_training.png")


def plot_kpis(model):
    # predicted vs true o2 on the test set, one panel per axis with its KPIs
    pred = model.predict(model.o1_test)
    kpis = model.kpis(multioutput="raw_values")
    fig, axes = plt.subplots(1, 3)
    for k, (ax, axis) in enumerate(zip(axes, "xyz")):
        true = model.o2_test[:, k]
        ax.plot([true.min(), true.max()], [true.min(), true.max()], "--", color="#52514e", lw=2, label="ideal")
        ax.plot(true, pred[:, k], "o", ms=8, label="test frame")
        ax.set(title=axis, xlabel=f"${axis}_2$ true [m]", ylabel=f"${axis}_2^*$ predicted [m]")
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.text(0.04, 0.96, f"RMSE {kpis['RMSE [mm]'][k]:.3f} mm\nMAE  {kpis['MAE [mm]'][k]:.3f} mm\n"
                            f"R²   {kpis['R2'][k]:.4f}",
                transform=ax.transAxes, va="top", fontsize=16, family="monospace",
                bbox=dict(facecolor="#fcfcfb", edgecolor="#d9d8d4"))
    axes[2].legend(loc="lower right")
    fig.suptitle("(d) $f_\\theta(o_1) = o_2^*$ on the test set")
    fig.savefig(RESULT / "q1_kpis.png")


def plot_errors(frames, e1, e2):
    fig, ax = plt.subplots()
    ax.plot(frames, 1e3 * e1, label="(i) $\\|o_1 - o_2\\|$")
    ax.plot(frames, 1e3 * e2, "--", label="(ii) $\\|o_1 - o_2^*\\|$")
    ax.set(title="(e) Euclidean error per frame", xlabel="frame", ylabel="error [mm]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    fig.savefig(RESULT / "q1_error.png")


def main():
    plt.style.use(SLIDES)
    cam_int = Camera.from_json("M_int", DATA / "depth_scale_json/color_intrinsics_own.json")  # M_int read from our L515 (fetch_intrinsics.py)
    cam_d = Camera("M_d", np.array([[1342, 0, 325],
                                    [0, 1342, 235],
                                    [0, 0, 1]], float))  # default L515 matrix from the assignment

    detections = LoadDetector(HERE / "camera_tracking_model.pt", DATA).detect_all()
    frames = np.array([det.frame for det in detections])

    # (a) o1 with M_int and (b) o2 with M_d
    o1 = cam_int.back_project_all(detections)
    o2 = cam_d.back_project_all(detections)
    plot_trajectories(o1, o2)

    # (c) which matrix is accurate: compare the field of view of each matrix with the L515 datasheet
    for cam in (cam_int, cam_d):
        fov_h, fov_v = cam.field_of_view()
        print(f"(c) {cam.name}: field of view {fov_h:.1f} x {fov_v:.1f} deg")
    print("    L515 datasheet RGB field of view (4:3): ~55 x 43 deg")

    # (d) ML model f_theta(o1) = o2*
    model = MappingModel()
    model.fit(o1, o2)
    o2_star = model.predict(o1)
    W, b = model.parameters()
    print(f"\n(d) one-layer perceptron, split {model.n_train} / {model.n_val} / {len(model.o1_test)} frames "
          f"(train / val / test), {model.epochs} epochs")
    print("    final loss (MSE): train {:.2e}, val {:.2e} m²".format(model.history["loss"][-1],
                                                                   model.history["val_loss"][-1]))
    print("    learned parameters theta: W =\n", W.round(4), "\n    b =", b.round(5))
    print("    KPIs on test set:", ", ".join(f"{k} = {v:.6f}" for k, v in model.kpis().items()))
    plot_training(model)
    plot_kpis(model)

    # (e) Euclidean error (i) o1 - o2 and (ii) o1 - o2*
    e1 = np.linalg.norm(o1 - o2, axis=1)
    e2 = np.linalg.norm(o1 - o2_star, axis=1)
    print(f"\n(e) (i)  |o1 - o2|:  mean {1e3 * e1.mean():.2f} mm, max {1e3 * e1.max():.2f} mm")
    print(f"    (ii) |o1 - o2*|: mean {1e3 * e2.mean():.2f} mm, max {1e3 * e2.max():.2f} mm")
    plot_errors(frames, e1, e2)

    # everything the presentation figure scripts need, so they don't rerun YOLO and the training
    normalization, dense = model.model.layers
    kpis = model.kpis()
    results.save(frames=frames, u=[d.u for d in detections], v=[d.v for d in detections],
                 depth=[d.d for d in detections], boxes=[d.box for d in detections], o1=o1, o2=o2, o2_star=o2_star,
                 W=W, b=b, loss=model.history["loss"], val_loss=model.history["val_loss"], n_train=model.n_train,
                 n_val=model.n_val, epochs=model.epochs, n_trainable=dense.count_params(),
                 n_fixed=normalization.count_params(), rmse_mm=kpis["RMSE [mm]"], mae_mm=kpis["MAE [mm]"],
                 r2=kpis["R2"])
    plt.show()


if __name__ == "__main__":
    main()

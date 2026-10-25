# Q3 (b)(i) : Lidar-camera fusion on KITTI
# projecting the lidar points onto the camera image for 5 random frames

#comand to install reqired packages
#pip install numpy opencv-python-headless matplotlib

import os
import random
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no screen on the server so we save the plots as images
import matplotlib.pyplot as plt


# folders
script_folder = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_folder, "..", "data", "training")
output_folder = os.path.join(script_folder, "..", "output")




# function to read the calibration fil
def read_calib(filename):
    calib = {}
    f = open(filename)
    for line in f:
        # split the line into name and numbers, convert the numbers to a numpy array"
        if ":" in line:
            name, numbers = line.split(":", 1)
            values = []
            for n in numbers.split():
                values.append(float(n))
            calib[name.strip()] = np.array(values)
    f.close()

    # P2 is the projection matrix for the color camera 
    P2 = calib["P2"].reshape(3, 4)
    # R0_rect is 3x3, we put it in a 4x4 identity matrix
    R0 = np.eye(4)
    R0[0:3, 0:3] = calib["R0_rect"].reshape(3, 3)
    # Tr_velo_to_cam is 3x4, also put it in a 4x4 identity matrix
    Tr = np.eye(4)
    Tr[0:3, 0:4] = calib["Tr_velo_to_cam"].reshape(3, 4)

    return P2, R0, Tr


# get all the frame names from the image folder
frame_list = []
for file in sorted(os.listdir(os.path.join(data_folder, "image_2"))):
    if file.endswith(".png"):
        frame_list.append(file.replace(".png", ""))

# pick 5 random frames
chosen_frames = random.sample(frame_list, 5)
print("frames chosen:", chosen_frames)


for frame in chosen_frames:

    # load the image, opencv uses BGR so we change it to RGB
    image = cv2.imread(os.path.join(data_folder, "image_2", frame + ".png"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height = image.shape[0]
    width = image.shape[1]

    # load the lidar points, 4 numbers per point (x, y, z, reflectance)
    lidar = np.fromfile(os.path.join(data_folder, "velodyne", frame + ".bin"), dtype=np.float32)
    lidar = lidar.reshape(-1, 4)

    # load the calibration for this frame
    P2, R0, Tr = read_calib(os.path.join(data_folder, "calib", frame + ".txt"))

    # take x, y, z and add a column of ones -> [x, y, z, 1]
    xyz = lidar[:, 0:3]
    ones = np.ones((xyz.shape[0], 1))
    points = np.hstack((xyz, ones))

    # lidar frame -> camera frame
    points_cam = R0 @ Tr @ points.T

    # remove points behind the camera (z < 0)
    # if we dont do this they get flipped and show up in the image
    points_cam = points_cam[:, points_cam[2] > 0]

    # camera frame to image plane
    pixels = P2 @ points_cam
    u = pixels[0] / pixels[2]
    v = pixels[1] / pixels[2]

    # depth is the z value in the camera frame 
    depth = points_cam[2]

    # only keep the points that are inside the image since lidar is 360 degrees
    # and the camera is only 90 degrees
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u = u[inside]
    v = v[inside]
    depth = depth[inside]

    print("frame", frame, ":", len(lidar), "lidar points,", len(u), "inside the image")

    # plot the image with the lidar points on top
    # jet_r makes close points red and far points blue
    plt.figure(figsize=(16, 5))
    plt.imshow(image)
    plt.scatter(u, v, c=depth, cmap="jet_r", s=1)
    plt.colorbar(label="Depth [m]", fraction=0.015, pad=0.01)  # thin colour bar close to the image
    plt.title("Lidar projected onto image - frame " + frame)
    plt.axis("off")
    plt.savefig(os.path.join(output_folder, frame + "_projection.png"))
    plt.close()


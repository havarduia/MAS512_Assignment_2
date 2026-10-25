# Read M_int from our own L515: the calibration stored in the camera, for the colour stream we recorded with
# pyrealsense2 dropped L515 support after 2.50, and 2.50 has no wheels past Python 3.9, so run it with:
#   uv run --no-project --python 3.9 --with "pyrealsense2==2.50.0.3812" python fetch_intrinsics.py

import json
from pathlib import Path

import pyrealsense2 as rs

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "camera_data/depth_scale_json/color_intrinsics_own.json"   # same format as color_intrinsics.json
WIDTH, HEIGHT, FPS = 640, 480, 30                                          # same as the recording


def main():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    try:
        device = profile.get_device()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    finally:
        pipeline.stop()

    data = {
        "width": intr.width, "height": intr.height,
        "ppx": intr.ppx, "ppy": intr.ppy,
        "fx": intr.fx, "fy": intr.fy,
        "model": str(intr.model).split(".")[-1],
        "coeffs": list(intr.coeffs),
        "serial_number": device.get_info(rs.camera_info.serial_number),  # which camera this matrix belongs to
    }
    OUTPUT.write_text(json.dumps(data, indent=2))

    print(f"L515 serial {data['serial_number']}, colour {intr.width}x{intr.height}")
    print(f"M_int = [[{intr.fx:.2f}, 0, {intr.ppx:.2f}],\n         [0, {intr.fy:.2f}, {intr.ppy:.2f}],\n         [0, 0, 1]]")
    print(f"Saved {OUTPUT.relative_to(HERE)}")


if __name__ == "__main__":
    main()

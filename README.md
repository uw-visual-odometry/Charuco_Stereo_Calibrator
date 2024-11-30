# Overview

The `CharucoStereoCalibrator`, `StereoCalibrator` and `Rectifier` are Python tools for stereo camera calibration and rectification. They compute intrinsic and stereo parameters, rectify stereo images, and visualize epipolar geometry, essential for 3D vision tasks like depth estimation and 3D reconstruction.

---

## Features

#### StereoCalibrator
- Automatic detection of chessboard corners in stereo images.
- Computes intrinsic parameters (camera matrix, distortion coefficients).
- Computes stereo parameters (rotation, translation, rectification, projection, and Q matrix).
- Saves calibration data to `stereoMap.xml`.
- Visualizes calibration errors and highlights outliers.

#### CharucoStereoCalibrator
- Support same feature with `StereoCalibrator` but using Charucoboard.

#### CharucoCalibrator
- One camera calibrator using Charucoboard.
- Feel free to use `generate_charuco.py` to generate your own Charucoboard. It's designed for A4 printing.

#### Rectifier
- Loads stereo calibration data from `stereoMap.xml`.
- Rectifies stereo image pairs for horizontal alignment.
- Computes the **Fundamental Matrix (F)**.
- Visualizes epipolar geometry with epilines and matched points.
- Saves rectified images and visualization outputs.

---

## Quick Start

#### 0. Installation

```
conda create -n csc python=3.12 -y
conda activate csc
pip install -e .
```

`CharucoCalibrator` is sensitive to opencv version, because there's big update since `4.6`

#### 1. **Calibration**
```python
from csc.charuco_stereo_calibrator import CharucoStereoCalibrator
import glob

# Specify image paths
images_left = glob.glob("input/charuco/left/*.jpg")
images_right = glob.glob("input/charuco/right/*.jpg")

# Specify number of column, row of checkerboard
chessboard_size = (11, 8) 

# Specify image sizes
frame_size_h = 1296 
frame_size_w = 2304 

# [Optional] if you don't know camera spec, then algorithm figure this out.
f_in_mm = 4.74 # or None
pixel_size_mm = 1.4e-3 # or None
debug = False # or False

stereo_calibrator = CharucoStereoCalibrator(
    chessboard_size=chessboard_size,
    frame_size_h=frame_size_h,
    frame_size_w=frame_size_w,
    f_in_mm=f_in_mm,
    pixel_size_mm=pixel_size_mm,
    debug=debug,
)

# Specify samples to show epipolar geometry qualitatively to evaluate calibration
left_show = "demo/samples/left_sample.jpg"
right_show = "demo/samples/right_sample.jpg"

stereo_calibrator.perform_calibration(images_left, images_right)
stereo_calibrator.save_rectified_images(images_left, images_right)
stereo_calibrator.visualize_epipolar(left_show, right_show, save=debug)
stereo_calibrator.print_results()
stereo_calibrator.measure_outlier()
```

**Note:** To explore additional examples, please refer to the `__main__` section within each function.

---

## Outputs
- **Calibration Data**: `stereoMap.xml` (camera matrices, rectification maps, etc.).
- **Rectified Images**: Horizontally aligned stereo pairs.
- **Epipolar Visualizations**: Saved epipolar images highlighting epilines and point matches.

---

## Key Methods
- `StereoCalibrator.perform_calibration()` - Calibrate cameras.
- `Rectifier.rectify_image()` - Rectify stereo image pairs.
- `Rectifier.visualize_epipolar()` - Draw epilines and visualize epipolar geometry.

---

## Notes
- Ensure chessboard size matches the setup (`chessboard_size`).
- Use `visualize_epipolar()` to verify and debug calibration.
- Unlike Checker board, Charuco board is strict for the order of `col`, `row`
  - Checker board considers the number of conjunctions.
  - Charuco board considers the number of rows and columns of blocks.

---

## License

This project is licensed under the MIT License.
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
- Support same feature above but using Charucoboard.
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
`CharucoCalibrator` is sensitive to opencv version, because there's big update since `4.7`
```
pip3 install opencv-python==4.10.0.84
```

#### 1. **Stereo Calibration**
```python
images_left = glob.glob("images/left/*.jpg")
images_right = glob.glob("images/right/*.jpg")

# Perform calibration
stereo_calibrator = StereoCalibrator(
    chessboard_size=(7, 10), frame_size_h=1296, frame_size_w=2304, size_of_chessboard_squares_mm=23
)
stereo_calibrator.perform_calibration(images_left, images_right)
stereo_calibrator.print_results()
```
- Calibration outputs are saved in `stereoMap.xml`.

#### 2. **Stereo Rectification and Epipolar Visualization**
```python
rectifier = Rectifier(calibration_file="stereoMap.xml", visualization_dims=(960, 540))

# Rectify images
left, right = rectifier.rectify_image("left.jpg", "right.jpg")
cv.imwrite("rectified_left.jpg", left)
cv.imwrite("rectified_right.jpg", right)

# Visualize epipolar geometry
rectifier.visualize_epipolar("left.jpg", "right.jpg", save=True)
```

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
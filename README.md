# StereoCalibrator README

## Overview

The `StereoCalibrator` class is a Python-based tool designed to facilitate the calibration of stereo camera systems. This script processes images from stereo cameras, extracts chessboard features, and computes calibration parameters crucial for applications involving 3D reconstruction, depth estimation, and stereo vision.

## Features

- **Image Processing**: Automatically sorts and processes input images to detect chessboard corners.
- **Camera Calibration**: Computes intrinsic parameters like the camera matrix and distortion coefficients for both cameras.
- **Stereo Calibration**: Determines the rotation, translation, rectification, and projection matrices between two cameras.
- **Calibration Results**: Saves all computed matrices and maps for future use and prints them in a readable format.


## Usage

1. **Image Preparation**: Capture chessboard images from both left and right cameras and place them into separate directories, e.g., `downloaded_images/left` and `downloaded_images/right`.

2. **Script Execution**: Adjust the image path patterns if necessary and run the script:

```python
images_left = glob.glob("downloaded_images/left/*.jpg")
images_right = glob.glob("downloaded_images/right/*.jpg")

stereo_calibrator = StereoCalibrator()
stereo_calibrator.perform_calibration(images_left, images_right)
stereo_calibrator.print_results()
```

3. **Results**: Calibration results will be printed to the console and saved to `stereoMap.xml`, which includes intrinsic camera parameters, rectification matrices, projection matrices, and the Q matrix for disparity-to-depth mapping.

## Methods

- **`process_images(images_left, images_right)`**: Processes stereo images and finds chessboard corners.
- **`calibrate_camera(imgpoints)`**: Performs camera calibration using image points.
- **`perform_calibration(images_left, images_right)`**: Main function to start the calibration process.
- **`stereo_calibration(camera_matrix_L, dist_L, camera_matrix_R, dist_R)`**: Conducts stereo calibration and rectification.
- **`save_matrices(camera_matrix_L, dist_L, camera_matrix_R, dist_R)`**: Saves the calibration results to an XML file.
- **`print_results()`**: Outputs the focal length, rotation matrix, translation vector, and baseline distance.

## Parameters

- `chessboard_size`: Tuple specifying the number of inner corners per a chessboard row and column (default: (10, 7)).
- `frame_size_h` and `frame_size_w`: Dimensions of the images to be processed.
- `size_of_chessboard_squares_mm`: Real dimension of the chessboard squares in millimeters (default: 23mm).
- `f_in_mm`: Focal length of the camera in millimeters (optional).
- `pixel_size_mm`: Physical size of a pixel in millimeters (optional).

## License

This project is open-source and freely redistributable with credit to its original authors.

**Note**: Ensure that the chessboard pattern used corresponds to the `chessboard_size` specified.
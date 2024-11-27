import numpy as np
import cv2 as cv
import glob
import re
import os
import matplotlib.pyplot as plt
import datetime


def log_message(message, level="INFO"):
    """Helper function to log messages with color and timestamps."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    level_colors = {
        "INFO": "\033[94m",  # Blue
        "WARNING": "\033[93m",  # Yellow
        "SUCCESS": "\033[92m",  # Green
        "ERROR": "\033[91m",  # Red
    }
    reset_color = "\033[0m"
    color = level_colors.get(level.upper(), "\033[94m")  # Default to Blue for INFO
    print(f"{color}[{level.upper()}] {now} - {message}{reset_color}")


def numerical_sort(value):
    """Helper function to extract numbers from a file name for sorting."""
    numbers = re.findall(r"\d+", value)
    return list(map(int, numbers))


class CharucoStereoCalibrator:

    def __init__(
        self,
        chessboard_size=(10, 7),
        frame_size_h=2592,
        frame_size_w=4608,
        f_in_mm=2.75,
        pixel_size_mm=1.4e-3,
        debug=False,
    ):
        self.chessboard_size = chessboard_size
        self.frame_size_h = frame_size_h
        self.frame_size_w = frame_size_w
        self.debug = debug

        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Initialize lists to store object points and image points (for both cameras)
        self.objpointsL = []  # 3d point in real-world space
        self.objpointsR = []  # 3d point in real-world space
        self.imgpointsL = []  # 2d point in image plane
        self.imgpointsR = []  # 2d point in image plane
        self.objpoints_common = []  # 3d point in real-world space
        self.imgpointsL_common = []  # 2d point in image plane
        self.imgpointsR_common = []  # 2d point in image plane
        self.idL = []  # 2d points in left camera image plane.
        self.idR = []  # 2d points in right camera image plane.

        # Intrinsic parameters
        self.f_in_mm = f_in_mm
        self.pixel_size_mm = pixel_size_mm
        self.board = None

        if self.f_in_mm is not None and self.pixel_size_mm is not None:
            f_in_pixels = f_in_mm / pixel_size_mm
            cx_in_pixel = frame_size_w // 2
            cy_in_pixel = frame_size_h // 2

            # Note: if sensor pixel is not square, it needs fx and fy.
            self.known_camera_matrix = np.array(
                [
                    [f_in_pixels, 0, cx_in_pixel],
                    [0, f_in_pixels, cy_in_pixel],
                    [0, 0, 1],
                ],
                dtype=np.float64,
            )
        else:
            self.known_camera_matrix = None

        log_message("Calibration started for stereo images.", level="INFO")

    def measure_outlier(
        self, outlier_threshold_single: int = 1.5, outlier_threshold_stereo: int = 2.5
    ):
        """Identify and visualize outliers based on re-projection and stereo errors."""
        errorsL = []
        errorsR = []
        stereo_errors = []

        # Calculate re-projection errors for each image set
        for (
            objpoints,
            imgpointsL,
            imgpointsR,
            idL,
            idR,
            rvecL,
            tvecL,
            rvecR,
            tvecR,
        ) in zip(
            self.objpoints_common,
            self.imgpointsL_common,
            self.imgpointsR_common,
            self.idL,
            self.idR,
            self.rvecsL,
            self.tvecsL,
            self.rvecsR,
            self.tvecsR,
        ):
            projected_pointsL, _ = cv.projectPoints(
                objpoints, rvecL, tvecL, self.cameraMatrixL, self.distL
            )
            projected_pointsR, _ = cv.projectPoints(
                objpoints, rvecR, tvecR, self.cameraMatrixR, self.distR
            )

            errorL = cv.norm(imgpointsL, projected_pointsL, cv.NORM_L2) / len(
                projected_pointsL
            )
            errorR = cv.norm(imgpointsR, projected_pointsR, cv.NORM_L2) / len(
                projected_pointsR
            )

            # Calculate stereo error using the epipolar constraint
            pointsL_h = cv.convertPointsToHomogeneous(imgpointsL).reshape(-1, 3)
            pointsR_h = cv.convertPointsToHomogeneous(imgpointsR).reshape(-1, 3)
            Fund_mat, _ = cv.findFundamentalMat(pointsL_h, pointsR_h, cv.FM_8POINT)

            stereo_error = 0
            for pl, pr in zip(pointsL_h, pointsR_h):
                # Epipolar constraint: pl' * F * pr = 0
                err = np.abs(pl @ Fund_mat @ pr.T)
                stereo_error += err

            stereo_error /= len(objpoints)

            errorsL.append(errorL)
            errorsR.append(errorR)
            stereo_errors.append(stereo_error)

        # Plot errors to visualize potential outliers
        plt.figure(figsize=(10, 8))
        plt.plot(errorsL, label="Left Camera Re-projection Error")
        plt.plot(errorsR, label="Right Camera Re-projection Error")
        plt.plot(stereo_errors, label="Stereo Error")
        plt.xlabel("Image Index")
        plt.ylabel("Error")
        plt.title("Errors for Calibration Images")
        plt.legend()
        plt.grid(True)
        plt.show()

        # Determine and print outliers based on a threshold
        reproject_threshold = max(
            np.mean(errorsL) + outlier_threshold_single * np.std(errorsL),
            np.mean(errorsR) + outlier_threshold_single * np.std(errorsR),
        )
        stereo_threshold = np.mean(stereo_errors) + outlier_threshold_stereo * np.std(
            stereo_errors
        )

        outlier_indices = [
            i
            for i, (eL, eR, st_e) in enumerate(zip(errorsL, errorsR, stereo_errors))
            if eL > reproject_threshold
            or eR > reproject_threshold
            or st_e > stereo_threshold
        ]

        if outlier_indices:
            for idx in outlier_indices:
                left_file_name = os.path.basename(images_left[idx])
                right_file_name = os.path.basename(images_right[idx])
                eL = errorsL[idx]
                eR = errorsR[idx]
                st_e = stereo_errors[idx]
                print(
                    f"Outlier at index {idx}: Left Image: {left_file_name}, Right Image: {right_file_name}, "
                    f"Left Error: {eL:.3f}, Right Error: {eR:.3f}, Stereo Error: {st_e:.3f}"
                )
        else:
            print("No significant outliers detected.")

    def save_rectified_images(self, images_left, images_right):
        """Save both fully rectified images and ROI-cropped images."""
        images_left.sort(key=numerical_sort)
        images_right.sort(key=numerical_sort)

        # Create directories for saving the rectified images
        full_dir = "./rectified/full"
        roi_dir = "./rectified/only_roi"
        os.makedirs(full_dir, exist_ok=True)
        os.makedirs(roi_dir, exist_ok=True)

        for img_left_path, img_right_path in zip(images_left, images_right):
            imgL = cv.imread(img_left_path)
            imgR = cv.imread(img_right_path)

            # Apply rectification maps
            rectifiedL = cv.remap(
                imgL, self.stereoMapL[0], self.stereoMapL[1], cv.INTER_LANCZOS4
            )
            rectifiedR = cv.remap(
                imgR, self.stereoMapR[0], self.stereoMapR[1], cv.INTER_LANCZOS4
            )

            # Save full rectified images
            imgL_filename = os.path.basename(img_left_path)
            imgR_filename = os.path.basename(img_right_path)
            rectifiedL_filename = f"{os.path.splitext(imgL_filename)[0]}_rectified.jpg"
            rectifiedR_filename = f"{os.path.splitext(imgR_filename)[0]}_rectified.jpg"
            rectifiedL_path = os.path.join(full_dir, "full", rectifiedL_filename)
            rectifiedR_path = os.path.join(full_dir, "full", rectifiedR_filename)

            cv.imwrite(rectifiedL_path, rectifiedL)
            cv.imwrite(rectifiedR_path, rectifiedR)

            # Crop to ROI and save
            xL, yL, wL, hL = self.rect_roi_L
            roi_rectifiedL = rectifiedL[yL : yL + hL, xL : xL + wL]
            roi_rectifiedL_filename = (
                f"{os.path.splitext(imgL_filename)[0]}_rectified_roi.jpg"
            )
            roi_rectifiedL_path = os.path.join(roi_dir, roi_rectifiedL_filename)
            cv.imwrite(roi_rectifiedL_path, roi_rectifiedL)

            xR, yR, wR, hR = self.rect_roi_R
            roi_rectifiedR = rectifiedR[yR : yR + hR, xR : xR + wR]
            roi_rectifiedR_filename = (
                f"{os.path.splitext(imgR_filename)[0]}_rectified_roi.jpg"
            )
            roi_rectifiedR_path = os.path.join(roi_dir, roi_rectifiedR_filename)
            cv.imwrite(roi_rectifiedR_path, roi_rectifiedR)

    def process_images(self, images_left, images_right):
        """Process stereo images to find Charuco corners."""

        # Sort images to maintain consistent pairing
        images_left.sort(key=numerical_sort)
        images_right.sort(key=numerical_sort)

        # Parameters for ArUco detection
        aruco_dict = cv.aruco.getPredefinedDictionary(
            cv.aruco.DICT_4X4_250
        )  # what is 1000?
        arucoParams = cv.aruco.DetectorParameters()

        self.board = cv.aruco.CharucoBoard(
            self.chessboard_size, 20, 15, aruco_dict
        )  # self.square_length, marker_length

        for img_left_path, img_right_path in zip(images_left, images_right):
            img_left = cv.imread(img_left_path)
            img_right = cv.imread(img_right_path)
            gray_left = cv.cvtColor(img_left, cv.COLOR_BGR2GRAY)
            gray_right = cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

            # Check frame size
            if gray_left.shape != (
                self.frame_size_h,
                self.frame_size_w,
            ) or gray_right.shape != (self.frame_size_h, self.frame_size_w):
                raise ValueError(
                    f"File size and frame size do not match. file: {img_left_path}"
                )

            # Detect markers and interpolate Charuco corners
            corners_left, ids_left, rejected_left = cv.aruco.detectMarkers(
                gray_left, aruco_dict, parameters=arucoParams
            )

            corners_right, ids_right, rejected_right = cv.aruco.detectMarkers(
                gray_right, aruco_dict, parameters=arucoParams
            )

            # Validate the refined corner IDs before proceeding
            if (
                ids_left is not None
                and len(ids_left) > 0
                and ids_right is not None
                and len(ids_right) > 0
            ):

                corners_left2, ids_left2, rejected_left, ids_recovered_left = (
                    cv.aruco.refineDetectedMarkers(
                        gray_left,
                        self.board,
                        corners_left,
                        ids_left,
                        rejected_left,
                        cameraMatrix=self.known_camera_matrix,
                        distCoeffs=None,
                    )
                )

                corners_right2, ids_right2, rejected_right, ids_recovered_right = (
                    cv.aruco.refineDetectedMarkers(
                        gray_right,
                        self.board,
                        corners_right,
                        ids_right,
                        rejected_right,
                        cameraMatrix=self.known_camera_matrix,
                        distCoeffs=None,
                    )
                )

                # Convert the tuple of arrays into a single 3D NumPy array due to OpenCV error
                corners_L_array = np.array(
                    [corner[0] for corner in corners_left2], dtype=np.float32
                )
                corners_R_array = np.array(
                    [corner[0] for corner in corners_right2], dtype=np.float32
                )

                # Ensure the structure is as expected: (n, 4, 2)
                assert corners_L_array.shape[1] == 4
                assert corners_R_array.shape[1] == 4

                # Process the data to extract each i-th point across all corner sets
                temp_fix_corners_left = [
                    tuple(corners_L_array[:, i].reshape(-1, 1, 2)) for i in range(4)
                ]
                temp_fix_corners_right = [
                    tuple(corners_R_array[:, i].reshape(-1, 1, 2)) for i in range(4)
                ]

                for temp_fix_corner_left in temp_fix_corners_left:
                    (obj_points_L, img_points_L) = self.board.matchImagePoints(
                        detectedCorners=temp_fix_corner_left, detectedIds=ids_left2
                    )
                    self.imgpointsL.append(img_points_L)
                    self.objpointsL.append(obj_points_L)

                for temp_fix_corner_right in temp_fix_corners_right:
                    (obj_points_R, img_points_R) = self.board.matchImagePoints(
                        detectedCorners=temp_fix_corner_right, detectedIds=ids_right2
                    )
                    self.imgpointsR.append(img_points_R)
                    self.objpointsR.append(obj_points_R)

                self.idL.append(ids_left2)
                self.idR.append(ids_right2)

                if self.debug:

                    cv.aruco.drawDetectedMarkers(
                        img_left, corners_left2, ids_left2, (255, 0, 0)
                    )
                    cv.aruco.drawDetectedMarkers(
                        img_right, corners_right2, ids_right2, (255, 0, 0)
                    )
                    img_left_resized = cv.resize(img_left, (1920, 1080))
                    img_right_resized = cv.resize(img_right, (1920, 1080))

                    # Combine the images side by side by concatenating them horizontally
                    combined_image = cv.hconcat([img_left_resized, img_right_resized])

                    # Create a resizable window for visualization
                    cv.namedWindow("Calibration Debug", cv.WINDOW_NORMAL)

                    # Resize the window if needed
                    cv.resizeWindow(
                        "Calibration Debug", 960 * 2, 540
                    )  # Adjust the size as needed

                    # Display the combined image
                    cv.imshow("Calibration Debug", combined_image)
                    # Wait until 'c' is pressed
                    while True:
                        key = cv.waitKey(1) & 0xFF  # Wait for a key press
                        if key == ord("c"):
                            break

                cv.destroyAllWindows()  # Destroy the window after the key press
            else:
                log_message(
                    f"Charuco board couldn't be detected. Image pair: {img_left_path} and {img_right_path}",
                    level="ERROR",
                )

        if self.debug:
            cv.destroyAllWindows()

    def draw_thicker_markers(self, img, corners, thickness=8, radius=25):
        """Draw thicker circles at detected corners and connect them with lines using row-based colors."""
        cols, rows = self.chessboard_size
        num_corners = len(corners)

        # Define colors for alternating rows
        colors = [
            (255, 0, 0),  # Red
            (0, 255, 0),  # Green
            (0, 0, 255),  # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (0, 128, 128),  # Teal
            (128, 128, 0),  # Olive
            (128, 128, 128),  # Gray
            (255, 128, 0),  # Orange
            (255, 192, 203),  # Pink
            (128, 0, 0),  # Maroon
            (192, 192, 192),  # Silver
            (0, 128, 0),  # Dark Green
        ]

        # Draw the circles and lines following the color for each row
        for row in range(rows):
            color = colors[row % len(colors)]  # Alternate colors based on row index
            for col in range(cols):
                idx = row * cols + col
                if idx < num_corners:
                    center = tuple(map(int, corners[idx].ravel()))
                    cv.circle(img, center, radius, color, thickness)

                    # Draw horizontal line to the next column if within bounds
                    if col < cols - 1:
                        next_idx = idx + 1
                        if next_idx < num_corners:
                            next_center = tuple(map(int, corners[next_idx].ravel()))
                            cv.line(img, center, next_center, color, int(thickness / 2))

    def visualize_and_save_corners(
        self,
        img_left,
        corners2_left,
        ret_left,
        img_right,
        corners2_right,
        ret_right,
        img_left_path,
        img_right_path,
    ):
        # Create a debug directory if it doesn't exist
        debug_dir = "debug"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        """Visualize and save chessboard corners if debug is enabled."""
        cv.drawChessboardCorners(
            img_left, self.chessboard_size, corners2_left, ret_left
        )
        cv.drawChessboardCorners(
            img_right, self.chessboard_size, corners2_right, ret_right
        )

        # Optionally, overlay thicker markers on each corner
        self.draw_thicker_markers(img_left, corners2_left)
        self.draw_thicker_markers(img_right, corners2_right)

        # Concatenate images horizontally
        combined_image = np.hstack((img_left, img_right))

        # Create a resizable window for visualization
        cv.namedWindow("Stereo Calibration Debug", cv.WINDOW_NORMAL)

        # Resize the window if needed
        cv.resizeWindow(
            "Stereo Calibration Debug", 960 * 2, 540
        )  # Adjust the size as needed

        # Display the combined image
        cv.imshow("Stereo Calibration Debug", combined_image)

        left_debug_path = os.path.join(
            debug_dir, f"checker_{os.path.basename(img_left_path)}"
        )
        right_debug_path = os.path.join(
            debug_dir, f"checker_{os.path.basename(img_right_path)}"
        )
        cv.imwrite(left_debug_path, img_left)
        cv.imwrite(right_debug_path, img_right)
        cv.waitKey(2000)  # Wait for 2000 milliseconds
        cv.destroyAllWindows()  # Close windows after visualization

    def calibrate_camera(self, objpoints, imgpoints):
        """Calibrate the camera using the provided image points."""
        if self.known_camera_matrix is not None:
            given_camera_matrix = self.known_camera_matrix.copy()

            ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
                objpoints,
                imgpoints,
                (self.frame_size_w, self.frame_size_h),
                given_camera_matrix,  # this matrix will be updated in openCV
                distCoeffs=None,
                flags=(
                    cv.CALIB_USE_INTRINSIC_GUESS
                    + cv.CALIB_FIX_FOCAL_LENGTH
                    + cv.CALIB_FIX_PRINCIPAL_POINT
                ),
            )

        else:
            ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
                objpoints,
                imgpoints,
                (self.frame_size_w, self.frame_size_h),
                None,
                None,
            )
        return ret, camera_matrix, dist, rvecs, tvecs

    def perform_calibration(self, images_left, images_right):
        """Main function to perform stereo calibration."""
        log_message("Starting stereo image processing...", "INFO")
        self.process_images(images_left, images_right)

        self.print_pretty_matrix("✅Known Camera Intrinsics", self.known_camera_matrix)

        log_message("Calibrating the left camera...", "INFO")
        retL, cameraMatrixL, distL, rvecsL, tvecsL = self.calibrate_camera(
            self.objpointsL, self.imgpointsL
        )
        log_message(f"🎥 Left Camera Calibration RMS Error: {retL:.4f}", "SUCCESS")
        self.print_pretty_matrix("Left Camera Matrix", cameraMatrixL)

        log_message("Calibrating the right camera...", "INFO")
        retR, cameraMatrixR, distR, rvecsR, tvecsR = self.calibrate_camera(
            self.objpointsR, self.imgpointsR
        )
        log_message(f"🎥 Right Camera Calibration RMS Error: {retR:.4f}", "SUCCESS")
        self.print_pretty_matrix("Right Camera Matrix", cameraMatrixR)

        log_message("Performing stereo calibration...", "INFO")
        self.stereo_calibration(cameraMatrixL, distL, cameraMatrixR, distR)

        log_message("Saving calibration matrices and parameters...", "INFO")
        self.save_matrices(cameraMatrixL, distL, cameraMatrixR, distR)

        log_message("Stereo calibration completed successfully!", "SUCCESS")

        # Save results for class attributes
        self.rvecsL = rvecsL
        self.tvecsL = tvecsL
        self.rvecsR = rvecsR
        self.tvecsR = tvecsR
        self.cameraMatrixL = cameraMatrixL
        self.distL = distL
        self.cameraMatrixR = cameraMatrixR
        self.distR = distR

    def stereo_calibration(self, camera_matrix_L, dist_L, camera_matrix_R, dist_R):
        """Perform stereo calibration."""
        flags = cv.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # For stereo calibration, matched pairs are only used.
        for i in range(min(len(self.idL), len(self.idR))):
            # Ensure matching ids in both left and right images

            common_ids = np.intersect1d(self.idL[i], self.idR[i])
            if len(common_ids) > 0:
                indices_left = np.isin(self.idL[i], common_ids).flatten()
                indices_right = np.isin(self.idR[i], common_ids).flatten()

                self.objpoints_common.append(self.objpointsL[i * 4][indices_left])
                self.objpoints_common.append(self.objpointsL[i * 4 + 1][indices_left])
                self.objpoints_common.append(self.objpointsL[i * 4 + 2][indices_left])
                self.objpoints_common.append(self.objpointsL[i * 4 + 3][indices_left])
                # since we use only common_id, objpointsL represents all.
                self.imgpointsL_common.append(self.imgpointsL[i * 4][indices_left])
                self.imgpointsL_common.append(self.imgpointsL[i * 4 + 1][indices_left])
                self.imgpointsL_common.append(self.imgpointsL[i * 4 + 2][indices_left])
                self.imgpointsL_common.append(self.imgpointsL[i * 4 + 3][indices_left])
                self.imgpointsR_common.append(self.imgpointsR[i * 4][indices_right])
                self.imgpointsR_common.append(self.imgpointsR[i * 4 + 1][indices_right])
                self.imgpointsR_common.append(self.imgpointsR[i * 4 + 2][indices_right])
                self.imgpointsR_common.append(self.imgpointsR[i * 4 + 3][indices_right])

        (
            retStereo,
            newCameraMatrixL,
            distL,
            newCameraMatrixR,
            distR,
            rot,
            trans,
            E,
            F,
        ) = cv.stereoCalibrate(
            self.objpoints_common,
            self.imgpointsL_common,
            self.imgpointsR_common,
            camera_matrix_L,
            dist_L,
            camera_matrix_R,
            dist_R,
            (self.frame_size_w, self.frame_size_h),
            criteria_stereo,
            flags,
        )
        log_message(
            f"🎥 Stereo Camera Calibration RMS Error: {retStereo:.4f}", "SUCCESS"
        )

        # Assign the Fundamental matrix to self.F
        self.F = F

        # Stereo Rectification
        rectify_scale = 1
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
            newCameraMatrixL,
            distL,
            newCameraMatrixR,
            distR,
            (self.frame_size_w, self.frame_size_h),
            rot,
            trans,
            rectify_scale,
            (0, 0),
        )

        self.rectL = rectL
        self.rectR = rectR
        self.rect_roi_L = roi_L
        self.rect_roi_R = roi_R

        stereoMapL = cv.initUndistortRectifyMap(
            newCameraMatrixL,
            distL,
            rectL,
            projMatrixL,
            (self.frame_size_w, self.frame_size_h),
            cv.CV_16SC2,
        )
        stereoMapR = cv.initUndistortRectifyMap(
            newCameraMatrixR,
            distR,
            rectR,
            projMatrixR,
            (self.frame_size_w, self.frame_size_h),
            cv.CV_16SC2,
        )

        self.projMatrixL = projMatrixL
        self.projMatrixR = projMatrixR
        self.Q = Q
        self.rot = rot
        self.trans = trans
        self.stereoMapL = stereoMapL
        self.stereoMapR = stereoMapR

    def save_matrices(self, camera_matrix_L, dist_L, camera_matrix_R, dist_R):
        """Save calibration matrices and rectification maps to a file."""
        print("Saving parameters!")
        cv_file = cv.FileStorage("stereoMap.xml", cv.FILE_STORAGE_WRITE)

        cv_file.write("stereoMapL_x", self.stereoMapL[0])
        cv_file.write("stereoMapL_y", self.stereoMapL[1])
        cv_file.write("stereoMapR_x", self.stereoMapR[0])
        cv_file.write("stereoMapR_y", self.stereoMapR[1])

        # Store intrinsic parameters and distortion coefficients
        cv_file.write("cameraMatrixL", camera_matrix_L)
        cv_file.write("distL", dist_L)
        cv_file.write("cameraMatrixR", camera_matrix_R)
        cv_file.write("distR", dist_R)

        # Store rotation and translation between cameras
        cv_file.write("R", self.rot)
        cv_file.write("T", self.trans)

        # Store rectification transforms (rectification matrices)
        cv_file.write("rectL", self.rectL)
        cv_file.write("rectR", self.rectR)

        # Store projection matrices for both cameras
        cv_file.write("projMatrixL", self.projMatrixL)
        cv_file.write("projMatrixR", self.projMatrixR)

        # Store the Q matrix for disparity-to-depth mapping
        cv_file.write("Q", self.Q)

        cv_file.release()
        print("All parameters saved successfully!")

    def print_pretty_matrix(self, name, matrix):
        """Utility function to print matrices in a readable format with separators."""
        divider = "=" * 50
        print(f"\n{divider}\n{name.upper()}:\n{divider}\n")
        print(np.array2string(matrix, formatter={"float_kind": lambda x: f"{x:0.4f}"}))
        print(f"\n{divider}")

    def print_results(self):
        """Print calibration results."""
        divider = "=" * 70

        print(f"\n{divider}")
        print(f"{'CALIBRATION RESULTS':^70}")
        print(f"{divider}\n")

        focal_length_px = self.projMatrixL[0, 0]
        print(f"🎯 Focal Length in Pixels: {focal_length_px:.4f} px")

        self.print_pretty_matrix("Rotation Matrix (rot)", self.rot)
        self.print_pretty_matrix("Translation Vector (trans)", self.trans)

        baseline_distance = np.linalg.norm(self.trans)
        print(f"🔄 Baseline Distance (Camera Separation): {baseline_distance:.4f} mm")

    def visualize_epipolar(
        self, left_images, right_images, save: bool = False, num_lines: int = 10
    ):
        """
        Visualize epipolar geometry after stereo calibration by drawing epipolar lines
        on the rectified left and right images.

        Args:
            save (bool): Whether to save the visualization images to disk.
            num_lines (int): Number of random epipolar lines to draw for visualization.
        """

        # If the inputs are single file paths (strings), convert them to lists
        if isinstance(left_images, str):
            left_images = [left_images]
        if isinstance(right_images, str):
            right_images = [right_images]

        # Rectification must have been performed
        if not hasattr(self, "stereoMapL") or not hasattr(self, "stereoMapR"):
            raise ValueError(
                "Stereo calibration has not been performed or rectification data is missing!"
            )

        # Load left and right images
        left_images.sort(key=numerical_sort)
        right_images.sort(key=numerical_sort)

        # Define a target visualization size
        target_width = 960 * 2  # Adjust as needed
        target_height = 540  # Adjust as needed

        for idx, (left_img_path, right_img_path) in enumerate(
            zip(left_images, right_images)
        ):
            print(
                f"Visualizing epipolar geometry for {left_img_path} and {right_img_path}..."
            )

            # Read the images
            img_left = cv.imread(left_img_path)
            img_right = cv.imread(right_img_path)

            # Rectify the images using the stereo map
            imgL_rectified = cv.remap(
                img_left, self.stereoMapL[0], self.stereoMapL[1], cv.INTER_LINEAR
            )
            imgR_rectified = cv.remap(
                img_right, self.stereoMapR[0], self.stereoMapR[1], cv.INTER_LINEAR
            )

            # Generate random sample points within the image dimensions
            height, width, _ = imgL_rectified.shape

            y_coords = np.linspace(0, height - 1, num=num_lines)
            x_coords = np.random.randint(0, width, size=num_lines)

            # Combine `x` and `y` into points
            points = np.column_stack((x_coords, y_coords)).reshape(-1, 1, 2)
            points = points.astype(np.float32).reshape(-1, 1, 2)

            # Compute epilines for points in the left image and map them to the right image
            epilinesR = cv.computeCorrespondEpilines(points, 1, self.F).reshape(-1, 3)
            epilinesL = cv.computeCorrespondEpilines(points, 2, self.F).reshape(-1, 3)

            imgL_with_epilines = self.draw_epilines_on_image(
                imgL_rectified, epilinesR, points
            )
            imgR_with_epilines = self.draw_epilines_on_image(
                imgR_rectified, epilinesL, points
            )

            # Combine rectified left and right images with epipolar lines
            combined_output = np.hstack((imgL_with_epilines, imgR_with_epilines))

            # Resize the combined image to fit the screen
            scale = max(
                target_width / combined_output.shape[1],
                target_height / combined_output.shape[0],
            )
            resized_output = cv.resize(
                combined_output, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA
            )

            # Create a resizable window
            cv.namedWindow(
                f"Epipolar Geometry Visualization - Pair {idx + 1}", cv.WINDOW_NORMAL
            )
            cv.imshow(
                f"Epipolar Geometry Visualization - Pair {idx + 1}", resized_output
            )

            # Display the result for a specified duration or until a key is pressed
            cv.waitKey(7000)  # Display for 3 seconds or adjust as needed

            # Save the visualization if required
            if save:
                output_folder = "epipolar_geometry_visualizations"
                os.makedirs(output_folder, exist_ok=True)
                output_path = os.path.join(
                    output_folder, f"epipolar_image_pair_{idx + 1}.jpg"
                )
                cv.imwrite(output_path, combined_output)

        # Clean up OpenCV windows
        cv.destroyAllWindows()

    def draw_epilines_on_image(
        self, img, epilines, points, color=(0, 255, 0), thickness=2
    ):
        """
        Helper function to draw epilines on an image.

        Args:
            img (numpy.ndarray): The input image on which to draw epilines.
            epilines (numpy.ndarray): The epilines computed for the points.
            points (list of numpy.ndarray): Points corresponding to the epilines.
            color (tuple): The color of the epilines.
            thickness (int): The thickness of the epilines.

        Returns:
            numpy.ndarray: The image with epilines drawn.
        """
        img_with_lines = img.copy()

        for line, point in zip(epilines, points):
            a, b, c = line  # Epiline coefficients: ax + by + c = 0
            x0, y0 = 0, int(-c / b) if b != 0 else 0  # Line at the left image boundary
            x1, y1 = img.shape[1], (
                int(-(c + a * img.shape[1]) / b) if b != 0 else 0
            )  # Line at the right boundary

            # Draw the epiline on the image
            cv.line(img_with_lines, (x0, y0), (x1, y1), color, thickness)

            # Draw the corresponding point in the image
            pt = tuple(int(x) for x in point.ravel())
            cv.circle(
                img_with_lines, pt, radius=5, color=(255, 0, 0), thickness=-1
            )  # Mark the point

        return img_with_lines


if __name__ == "__main__":

    # Example usage
    images_left = glob.glob("downloaded_images/charuco/left/*.jpg")
    images_right = glob.glob("downloaded_images/charuco/right/*.jpg")

    chessboard_size = (8, 11)
    frame_size_h = 2592 // 2
    frame_size_w = 4608 // 2

    # if below is None, then algorithm figure this out.
    f_in_mm = 4.74
    pixel_size_mm = 1.4e-3 * 2  # binning

    stereo_calibrator = CharucoStereoCalibrator(
        chessboard_size=chessboard_size,
        frame_size_h=frame_size_h,
        frame_size_w=frame_size_w,
        f_in_mm=f_in_mm,
        pixel_size_mm=pixel_size_mm,
        debug=False,
    )

    left_show = "test_12mp_nonwide/1732617733_left.jpg"
    right_show = "test_12mp_nonwide/1732617733_right.jpg"

    stereo_calibrator.perform_calibration(images_left, images_right)
    # stereo_calibrator.save_rectified_images(images_left, images_right)
    # stereo_calibrator.visualize_epipolar(left_show, right_show, save=True)
    # stereo_calibrator.print_results()
    stereo_calibrator.measure_outlier()

import numpy as np
import cv2 as cv
import glob
import re
import os


def numerical_sort(value):
    """Helper function to extract numbers from a file name for sorting."""
    numbers = re.findall(r"\d+", value)
    return list(map(int, numbers))


class StereoCalibrator:

    def __init__(
        self,
        chessboard_size=(10, 7),
        frame_size_h=2592,
        frame_size_w=4608,
        size_of_chessboard_squares_mm=23,
        f_in_mm=2.75,
        pixel_size_mm=1.4e-3,
        debug=False,
    ):
        self.chessboard_size = chessboard_size
        self.frame_size_h = frame_size_h
        self.frame_size_w = frame_size_w
        self.size_of_chessboard_squares_mm = size_of_chessboard_squares_mm
        self.debug = debug

        # termination criteria
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # Prepare object points
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0 : chessboard_size[0], 0 : chessboard_size[1]
        ].T.reshape(-1, 2)
        self.objp = objp * size_of_chessboard_squares_mm

        # Initialize lists to store object points and image points (for both cameras)
        self.objpoints = []  # 3d point in real-world space
        self.imgpointsL = []  # 2d points in left camera image plane.
        self.imgpointsR = []  # 2d points in right camera image plane.

        # Intrinsic parameters
        self.f_in_mm = f_in_mm
        self.pixel_size_mm = pixel_size_mm

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

    def process_images(self, images_left, images_right):
        """Process stereo images to find chessboard corners."""
        images_left.sort(key=numerical_sort)
        images_right.sort(key=numerical_sort)

        for img_left_path, img_right_path in zip(images_left, images_right):
            img_left, img_right = cv.imread(img_left_path), cv.imread(img_right_path)
            gray_left, gray_right = cv.cvtColor(
                img_left, cv.COLOR_BGR2GRAY
            ), cv.cvtColor(img_right, cv.COLOR_BGR2GRAY)

            if gray_left.shape != (
                self.frame_size_h,
                self.frame_size_w,
            ) or gray_right.shape != (self.frame_size_h, self.frame_size_w):
                raise ValueError(
                    f"File size and frame size do not match. file: {img_left_path}"
                )

            ret_left, corners_left = cv.findChessboardCorners(
                gray_left, self.chessboard_size, None
            )
            ret_right, corners_right = cv.findChessboardCorners(
                gray_right, self.chessboard_size, None
            )

            if ret_left and ret_right:
                self.objpoints.append(self.objp)

                corners2_left = cv.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1), self.criteria
                )
                self.imgpointsL.append(corners2_left)

                corners2_right = cv.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1), self.criteria
                )
                self.imgpointsR.append(corners2_right)

            if self.debug:
                self.visualize_and_save_corners(
                    img_left,
                    corners2_left,
                    ret_left,
                    img_right,
                    corners2_right,
                    ret_right,
                    img_left_path,
                    img_right_path,
                )

    def draw_thicker_markers(self, img, corners, thickness=10, radius=35):
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

    def calibrate_camera(self, imgpoints):
        """Calibrate the camera using the provided image points."""
        if self.known_camera_matrix is not None:
            ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
                self.objpoints,
                imgpoints,
                (self.frame_size_w, self.frame_size_h),
                self.known_camera_matrix,
                (
                    cv.CALIB_USE_INTRINSIC_GUESS
                    + cv.CALIB_FIX_FOCAL_LENGTH
                    + cv.CALIB_FIX_PRINCIPAL_POINT
                ),
            )
        else:
            ret, camera_matrix, dist, rvecs, tvecs = cv.calibrateCamera(
                self.objpoints,
                imgpoints,
                (self.frame_size_w, self.frame_size_h),
                None,
                None,
            )
        return ret, camera_matrix, dist

    def perform_calibration(self, images_left, images_right):
        """Main function to perform stereo calibration."""
        self.process_images(images_left, images_right)

        print("Known Camera Matrix:")
        print(self.known_camera_matrix)

        retL, cameraMatrixL, distL = self.calibrate_camera(self.imgpointsL)
        print(f"Left Camera Calibration RMS Error: {retL}")

        retR, cameraMatrixR, distR = self.calibrate_camera(self.imgpointsR)
        print(f"Right Camera Calibration RMS Error: {retR}")

        self.stereo_calibration(cameraMatrixL, distL, cameraMatrixR, distR)
        self.save_matrices(cameraMatrixL, distL, cameraMatrixR, distR)

    def stereo_calibration(self, camera_matrix_L, dist_L, camera_matrix_R, dist_R):
        """Perform stereo calibration."""
        flags = cv.CALIB_FIX_INTRINSIC
        criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

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
            self.objpoints,
            self.imgpointsL,
            self.imgpointsR,
            camera_matrix_L,
            dist_L,
            camera_matrix_R,
            dist_R,
            (self.frame_size_w, self.frame_size_h),
            criteria_stereo,
            flags,
        )
        print(f"Stereo Calibration RMS Error: {retStereo}")

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
        """Utility function to print matrices in a readable format."""
        print(f"\n{name}:\n")
        print(np.array2string(matrix, formatter={"float_kind": lambda x: f"{x:0.4f}"}))

    def print_results(self):
        """Print calibration results."""
        focal_length_px = self.projMatrixL[0, 0]
        print("focal_length_px:", focal_length_px)

        self.print_pretty_matrix("Rotation Matrix (rot)", self.rot)
        self.print_pretty_matrix("Translation Vector (trans)", self.trans)

        baseline_distance = np.linalg.norm(self.trans)
        print(f"Baseline Distance: {baseline_distance:.4f} mm")


if __name__ == "__main__":

    # Example usage
    images_left = glob.glob("downloaded_images/left/*.jpg")
    images_right = glob.glob("downloaded_images/right/*.jpg")

    chessboard_size = (10, 7)
    frame_size_h = 2592
    frame_size_w = 4608
    size_of_chessboard_squares_mm = 23
    f_in_mm = 2.75
    pixel_size_mm = 1.4e-3

    stereo_calibrator = StereoCalibrator(
        chessboard_size=chessboard_size,
        frame_size_h=frame_size_h,
        frame_size_w=frame_size_w,
        size_of_chessboard_squares_mm=size_of_chessboard_squares_mm,
        f_in_mm=f_in_mm,
        pixel_size_mm=pixel_size_mm,
        debug=True,
    )
    stereo_calibrator.perform_calibration(images_left, images_right)
    stereo_calibrator.print_results()

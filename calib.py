from csc.charuco_stereo_calibrator import CharucoStereoCalibrator
import glob

# Specify image paths
# images_left = glob.glob("./images/AC_split/l/left_half*.jpg")
# images_right = glob.glob("./images/AC_split/r/right_half*.jpg")
images_left = glob.glob("./images/sa2/left/*.png")
images_right = glob.glob("./images/sa2/right/*.png")

images_left.sort()
images_right.sort()


# Specify number of column, row of checkerboard
chessboard_size = (11, 8) 

# Specify image sizes
frame_size_h = 1088
frame_size_w = 1440

# [Optional] if you don't know camera spec, then algorithm figure this out.
f_in_mm = 6 # or None
pixel_size_mm = 3.45e-3 # or None
debug = False

stereo_calibrator = CharucoStereoCalibrator(
    chessboard_size=chessboard_size,
    frame_size_h=frame_size_h,
    frame_size_w=frame_size_w,
    f_in_mm=f_in_mm,
    pixel_size_mm=pixel_size_mm,
    debug=debug,
)

# Specify samples to show epipolar geometry qualitatively to evaluate calibration
left_show = "./images/AC_split/l/*.jpg"
right_show = "./images/AC_split/r/*.jpg"

stereo_calibrator.perform_calibration(images_left, images_right)
stereo_calibrator.save_rectified_images(images_left, images_right)
stereo_calibrator.visualize_epipolar(left_show, right_show, save=debug)
stereo_calibrator.print_results()
stereo_calibrator.measure_outlier()

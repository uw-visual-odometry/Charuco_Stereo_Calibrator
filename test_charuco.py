import cv2 as cv
import numpy as np
import argparse
import os

def detect_charuco_board(image_path, output_path=None, debug=False):

    img = cv.imread(image_path)
    if img is None:
        print(f"Error, no image, {image_path}")
        return None, None, False

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)
    board = cv.aruco.CharucoBoard((8, 11), 0.0, 0.18, aruco_dict)

    params = cv.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.minMarkerPerimeterRate = 0.03
    params.polygonalApproxAccuracyRate = 0.05
    params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_CONTOUR

    detector = cv.aruco.CharucoDetector(board)
    detector.setDetectorParameters(params)

    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    print(charuco_corners)
    success = charuco_corners is not None and len(charuco_corners) > 0

    if success:
        print(f"Charuco board detected! {len(charuco_corners)} corners total.")

        result_img = img.copy()

        if charuco_corners is not None and charuco_ids is not None:
            cv.aruco.drawDetectedCornersCharuco(result_img, charuco_corners, charuco_ids)

        if marker_corners is not None and marker_ids is not None:
            cv.aruco.drawDetectedMarkers(result_img, marker_corners, marker_ids)

        if debug:
            cv.namedWindow("Charuco Board Detection", cv.WINDOW_NORMAL)
            cv.imshow("Charuco Board Detection", result_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        if output_path:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            cv.imwrite(output_path, result_img)
            print(f"result saved, {output_path}")

        if charuco_corners is not None and len(charuco_corners) > 0:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            refined_corners = cv.cornerSubPix(gray, charuco_corners, (3, 3), (-1, -1), criteria)
            return refined_corners, charuco_ids, True
    else:
        print(f"No Charuco board detected, {image_path}")

    return charuco_corners, charuco_ids, success


def main():
    image = './output/left/left_frame_000014.jpg'
    output = "./"
    debug = False

    corners, ids, success = detect_charuco_board(image, output, debug)

    if success:
        print(f"Succeed! {len(corners)} corners in total，IDs: {ids.ravel()}")
    else:
        print("Fail!")


if __name__ == "__main__":
    main()
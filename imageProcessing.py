import cv2 as cv
import os
import glob

"""
    Extracts frames from a video and saves them as PNG images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the output folder where PNG images will be saved.
        frame_interval (int): Interval of frame to capture
"""
def video_to_png(video_path, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video_capture = cv.VideoCapture(video_path)
    frame_count = 0
    image_count = 0

    while frame_count < 200: # True:
        success, frame = video_capture.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            output_file = os.path.join(output_folder, f"frame_{image_count:04d}.png")
            cv.imwrite(output_file, frame)
            image_count += 1
            print(frame_count)
        frame_count += 1

    video_capture.release()
    print(f"Extracted {image_count} frames to {output_folder}")

"""
    Extracts left and right PNG images from a full image and saves as grayscale.

    Args:
        image_path (str): Path of image
        output_folder (str): Path to the output folder where PNG images will be saved.
        idx (int): Number of image to correlate the left and right image to
"""
def splitLR(image_path, output_folder, idx):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    if not os.path.exists(output_folder + '/l'):
        os.makedirs(output_folder + '/l')
    
    if not os.path.exists(output_folder + '/r'):
        os.makedirs(output_folder + '/r')
    
    img = cv.imread(image_path)
    #GRAYSCALE
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    #height, width, channel = img.shape
    height, width = img.shape
    
    midpoint = width // 2
    
    left_half = img[:, :midpoint]
    right_half = img[:, midpoint:]
    
    cv.imwrite(output_folder + '/l/left_half' + str(idx) + '.jpg', left_half)
    cv.imwrite(output_folder + '/r/right_half' + str(idx) + '.jpg', right_half)

"""
    Extracts frames after a certain time period from a video and saves them as PNG images.

    Args:
        video_path (str): Path to the input video file.
        time (int): Time in seconds to wait before starting to capture frames
        output_folder (str): Path to the output folder where PNG images will be saved.
        frame_interval (int): Interval of frame to capture
"""
def waitCapture(video_path, time, output_folder, frame_interval):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # Skip the first 10 seconds (skip_time is in seconds)
    skip_time = time  
    fps = cap.get(cv.CAP_PROP_FPS)  # Frames per second of the video
    frame_to_skip = int(skip_time * fps)

    # Set the current frame position to the frame after the skip
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_to_skip)
    
    frame_count = 0
    image_count = 0

    while frame_count < 1000: # True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            output_file = os.path.join(output_folder, f"frame_{image_count:04d}.png")
            cv.imwrite(output_file, frame)
            image_count += 1
            print(frame_count)
        frame_count += 1

    cap.release()
    print(f"Extracted {image_count} frames to {output_folder}")
    
    
    

# #Example usage:
# video_path = "/home/sysop/bag/aaron_video/2025_04_07_hh101_calibration.mp4"
# output_folder = "./images/Aaron-Cal/"
# waitCapture(video_path, 6, output_folder, 6)
#
# images = glob.glob('./images/Aaron-Cal/*.png')
# output_folder = "./images/AC_split/"
# i = 0
# for image_path in images:
#     print(image_path)
#     splitLR(image_path, output_folder, i)
#     i += 1


#Example usage:
video_path = "./calib1_left.mp4"
output_folder = "./images/sa2/left"
waitCapture(video_path, 0, output_folder, 6)


video_path = "./calib1_right.mp4"
output_folder = "./images/sa2/right"
waitCapture(video_path, 0, output_folder, 6)


'''
video_path = "./underwater.mp4"
output_folder = "./U_frames/"
video_to_png(video_path, output_folder, 6)

images = glob.glob('../TG1_frames/*.png')
output_folder = "../TG1_split/"
i = 0
for image_path in images:
    splitLR(image_path, output_folder, i)
    i += 1
'''

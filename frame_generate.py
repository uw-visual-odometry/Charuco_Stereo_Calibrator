import cv2
import os
import argparse


def extract_frames(left_video_path, right_video_path, output_folder, frame_interval=10):
    """
    从左右两个摄像头的视频中每隔指定帧数提取图像

    参数:
    left_video_path: 左摄像头视频路径
    right_video_path: 右摄像头视频路径
    output_folder: 输出图像保存的文件夹
    frame_interval: 提取图像的帧间隔，默认为10
    """

    # 创建输出文件夹
    left_output_folder = os.path.join(output_folder, "left")
    right_output_folder = os.path.join(output_folder, "right")

    os.makedirs(left_output_folder, exist_ok=True)
    os.makedirs(right_output_folder, exist_ok=True)

    # 打开左摄像头视频
    left_cap = cv2.VideoCapture(left_video_path)
    if not left_cap.isOpened():
        print(f"错误: 无法打开左摄像头视频 {left_video_path}")
        return

    # 打开右摄像头视频
    right_cap = cv2.VideoCapture(right_video_path)
    if not right_cap.isOpened():
        print(f"错误: 无法打开右摄像头视频 {right_video_path}")
        left_cap.release()
        return

    # 获取视频帧数
    left_total_frames = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    right_total_frames = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"左摄像头视频总帧数: {left_total_frames}")
    print(f"右摄像头视频总帧数: {right_total_frames}")

    # 提取帧
    frame_idx = 0
    left_frame_count = 0
    right_frame_count = 0

    while True:
        # 读取左摄像头帧
        left_ret, left_frame = left_cap.read()

        # 读取右摄像头帧
        right_ret, right_frame = right_cap.read()

        # 如果两个视频都已读完，则退出循环
        if not left_ret and not right_ret:
            break

        # 如果当前帧索引是frame_interval的倍数，则保存图像
        if frame_idx % frame_interval == 0:
            show_frame = cv2.resize(left_frame, (1080, 720))
            cv2.imshow("", show_frame)
            if cv2.waitKey(0) != ord('s'):
                cv2.destroyAllWindows()
                continue
            cv2.destroyAllWindows()

            if left_ret:
                left_output_path = os.path.join(left_output_folder, f"left_frame_{left_frame_count:06d}.jpg")
                cv2.imwrite(left_output_path, left_frame)
                left_frame_count += 1

            if right_ret:
                right_output_path = os.path.join(right_output_folder, f"right_frame_{right_frame_count:06d}.jpg")
                cv2.imwrite(right_output_path, right_frame)
                right_frame_count += 1

            print(f"已保存帧 {frame_idx} (左: {left_frame_count - 1}, 右: {right_frame_count - 1})")

        frame_idx += 1

    # 释放资源
    left_cap.release()
    right_cap.release()

    print(f"完成! 共从左摄像头提取了 {left_frame_count} 帧，从右摄像头提取了 {right_frame_count} 帧")


if __name__ == "__main__":
    left = '/home/sysop/bag/mp4/calib_close_left.mp4'
    right = '/home/sysop/bag/mp4/calib_close_right.mp4'
    output = './output1/'
    interval = 3

    extract_frames(left, right, output, interval)

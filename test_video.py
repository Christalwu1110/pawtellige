# test_video.py
import cv2
cap = cv2.VideoCapture('/home/yeling/dog_project/dog_video.mp4')
if cap.isOpened():
    print("视频文件成功打开")
    print(f"帧数: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}, FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    cap.release()
else:
    print("无法打开视频文件，请检查格式或编码")
    
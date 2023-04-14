import os
import numpy as np
from glob import glob
import cv2


videoDirPath="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/EGTEA/video_clips"
videoDirs=sorted(glob(videoDirPath+"/*/*"))
# print(videoDirs)
for videoDir in videoDirs:
    # print(videoDir)
    videos=sorted(glob(videoDir+"/*"))
    # print(videos)
    for videoPath in videos:
        cap=cv2.VideoCapture(videoPath)
        num_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # if "P05-R01-PastaSalad-1087378-1093795-F026082-F026267" in videoPath:
        for i in range(num_frame):
            saveDir="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_rgb/"+os.path.basename(videoPath)[:-4]
            os.makedirs(saveDir,exist_ok=True)
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                resultPath="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_rgb/"+os.path.basename(videoPath)[:-4]+"/"+str(i+1).zfill(4)+".jpg"
                cv2.imwrite(resultPath, frame)
                print(resultPath)

# dataset/images_rgb/P05-R01-PastaSalad-1087378-1093795-F026082-F026267/0001.jpg
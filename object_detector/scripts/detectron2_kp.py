from detectron2_core import *

import os
import numpy as np
import torch
import cv2
from glob import glob

"""
model type
OD: object detection
IS: instance segmentation
LVIS: LVinstance segmentation
PS: panoptic segmentation
KP: keypoint detection
"""

sources_path="/home/hayashide/catkin_ws/src/object_detector/images/sources"
sources=sorted(glob(sources_path+"/*"))[37:]
print(sources)

results_path="/home/hayashide/catkin_ws/src/object_detector/images/results"

for source in sources:
    pic_path=os.path.basename(source)
    detectron2_img=results_path+"/detectron2/"+pic_path
    remap_img=results_path+"/remap/"+pic_path

    detector=Detector(model_type="KP")

    pred_keypoints=detector.onImage(source,detectron2_img)

    np_pred_keypoints=pred_keypoints.to(torch.device('cpu')).detach().clone().numpy()[0]
    print(np_pred_keypoints)

    img=cv2.imread(source)

    for keypoint in np_pred_keypoints:
        cv2.circle(img,
            center=(int(keypoint[0]), int(keypoint[1])),
            radius=10,
            color=(0, 255*float(keypoint[2]), 0),
            thickness=3,
            lineType=cv2.LINE_4,
            shift=0)
        cv2.putText(img,
                text=str(round(float(keypoint[2]),3)),
                org=(int(keypoint[0])+5, int(keypoint[1])+5),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 0, 255*float(keypoint[2])),
                thickness=2,
                lineType=cv2.LINE_4)

    cv2.imwrite(remap_img,img)
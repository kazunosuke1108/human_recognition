from detectron2_core import *
"""
model type
OD: object detection
IS: instance segmentation
LVIS: LVinstance segmentation
PS: panoptic segmentation
KP: keypoint detection
"""
detector=Detector(model_type="KP")

detector.onImage("/home/hayashide/catkin_ws/src/object_detector/images/01_no_lost.jpg")


# detector.onVideo("images/GX010708.MP4")
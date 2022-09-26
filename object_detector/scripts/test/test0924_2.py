from test0924_1 import *
"""
model type
OD: object detection
IS: instance segmentation
LVIS: LVinstance segmentation
PS: panoptic segmentation
KP: keypoint detection
"""
detector=Detector(model_type="KP")

detector.onImage("images/GOPR0757.JPG")

# detector.onVideo("images/GX010708.MP4")
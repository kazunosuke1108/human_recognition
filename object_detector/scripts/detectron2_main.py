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

# detector.onImage("/home/hayashide/catkin_ws/src/object_detector/images/01_no_lost.jpg")


detector.onVideo(videoPath="chair_occlusion.mp4",savePath="/home/hayashide/catkin_ws/src/object_detector/images/results/detectron2/41_chair_occlusion45.mp4",)#csvPath=f'/home/hayashide/catkin_ws/src/object_detector/csv/1004_detectron2.csv')
from detectron2_core import *
"""
model type
OD: object detection
IS: instance segmentation
LVIS: LVinstance segmentation
PS: panoptic segmentation
KP: keypoint detection
"""
detector=Detector(model_type="OD")

results=detector.onImage(imagePath="/home/hayashide/catkin_ws/src/object_detector/images/sources/00_no_lost.jpeg")
# print(list(detector.onImage(imagePath="/home/hayashide/catkin_ws/src/object_detector/images/sources/00_no_lost.jpeg")))#[0].numpy())
print(results)

# detector.onVideo(videoPath="chair_occlusion.mp4",csvPath=f'/home/hayashide/catkin_ws/src/object_detector/csv/1004_detectron2.csv')
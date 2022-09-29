import os
import matplotlib.pyplot as plt
import torch
import cv2
from torchvision import transforms
import numpy as np
from reference.yolov7.utils.datasets import letterbox
from reference.yolov7.utils.general import non_max_suppression_kpt
from reference.yolov7.utils.plots import output_to_keypoint, plot_skeleton_kpts


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# weigths = torch.load('/home/hayashide/catkin_ws/src/object_detector/config/yolov7/yolov7-w6-pose.pt', map_location=device)
weights = torch.hub.load("/usr/local/lib/python3.8/dist-packages/yolov7", 'custom', path=os.environ['HOME']+'/catkin_ws/src/object_detector/config/yolov7/yolovv7-w6-pose.pt',source='local')

model = weigths['model']
_ = model.float().eval()

if torch.cuda.is_available():
    model.half().to(device)
image = cv2.imread('home/hayashide/catkin_ws/src/object_detector/images/sources/00_no_lost.jpeg')
image = letterbox(image, 960, stride=64, auto=True)[0]
image_ = image.copy()
image = transforms.ToTensor()(image)
image = torch.tensor(np.array([image.numpy()]))

if torch.cuda.is_available():
    image = image.half().to(device)   
output, _ = model(image)

output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
with torch.no_grad():
    output = output_to_keypoint(output)
nimg = image[0].permute(1, 2, 0) * 255
nimg = nimg.cpu().numpy().astype(np.uint8)
nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
for idx in range(output.shape[0]):
    plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)

plt.figure(figsize=(8,8))
plt.axis('off')
plt.imsave("~/catkin_ws/src/object_detector/images/results/yolov7/00_no_lost.jpeg",nimg)
print("program finished somehow")
# plt.show()
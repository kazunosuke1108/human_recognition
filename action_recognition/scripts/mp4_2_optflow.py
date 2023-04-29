import os
import shutil
import numpy as np
from glob import glob
import cv2


videoDirPath="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/EGTEA/video_clips/cropped_clips"
videos=sorted(glob(videoDirPath+"/*/*"))
print(len(videos))
# for videoDir in videoDirs:
#     # print(videoDir)
#     videos=sorted(glob(videoDir+"/*"))
#     # print(videos)
for idx,videoPath in enumerate(videos):
    print(str(idx)+" / "+str(len(videos))+"    "+os.path.basename(videoPath))
    cap=cv2.VideoCapture(videoPath)
    ret, frame1 = cap.read()
    prvsImg = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    num_frame=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # if "P05-R01-PastaSalad-1087378-1093795-F026082-F026267" in videoPath:
    saveDir="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_rgb/"+os.path.basename(videoPath)[:-4]
    saveDir_f="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_flow/"+os.path.basename(videoPath)[:-4]
    saveDir_u="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_flow/"+os.path.basename(videoPath)[:-4]+"/u"
    saveDir_v="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_flow/"+os.path.basename(videoPath)[:-4]+"/v"
    # shutil.rmtree(saveDir)
    os.makedirs(saveDir,exist_ok=True)
    os.makedirs(saveDir_f,exist_ok=True)
    os.makedirs(saveDir_u,exist_ok=True)
    os.makedirs(saveDir_v,exist_ok=True)
    for i in range(num_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame2 = cap.read()
        nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvsImg,nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # fimgu=cv2.normalize(flow[:,:,0],None,0,1e-7,cv2.NORM_MINMAX)
        # fimgv=cv2.normalize(flow[:,:,1],None,0,1e-7,cv2.NORM_MINMAX)
        fimgu=flow[:,:,0]*255/flow[:,:,0].max()
        fimgv=flow[:,:,1]*255/flow[:,:,1].max()
        # print(flow[:,:,0])
        # ret, frame = cap.read()
        # if ret:
        # frame数2倍
        for j in [-1,0]:
            resultPath="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_rgb/"+os.path.basename(videoPath)[:-4]+"/"+str(int(2*(i+1)+j)).zfill(4)+".jpg"
            resultPath_u="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_flow/"+os.path.basename(videoPath)[:-4]+"/u/"+str(int(2*(i+1)+j)).zfill(4)+".jpg"
            resultPath_v="/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_flow/"+os.path.basename(videoPath)[:-4]+"/v/"+str(int(2*(i+1)+j)).zfill(4)+".jpg"
            cv2.imwrite(resultPath, frame2)
            cv2.imwrite(resultPath_u, fimgu)
            cv2.imwrite(resultPath_v, fimgv)
        prvsImg = nextImg

                # cv2.imwrite(resultPath, frame)
                # print(resultPath)


# KAZU HR hayashide:~/catkin_ws/src/third_party/Gaze-Attention$ python3 main.py --mode test
# exp_name:      test_i3d_iga_best1_base
# datasplit:     1
# weight:        weights/i3d_iga_best1_base.pt
# mode:          test
# test_sparse:   False
# loading weight file: weights/i3d_iga_best1_base.pt
# loading weight file: weights/i3d_iga_best1_gaze.pt
# loading weight file: weights/i3d_iga_best1_attn.pt
# run on cuda
# [ WARN:0@1.960] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P04-R06-GreekSalad-69970-77130-F001662-F001869/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.960] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P04-R06-GreekSalad-682170-683940-F016368-F016419/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.960] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P04-R06-GreekSalad-767250-769130-F018410-F018464/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.960] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P04-R06-GreekSalad-861590-864300-F020672-F020750/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.962] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P04-R06-GreekSalad-971840-974170-F023319-F023386/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.962] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P04-R06-GreekSalad-967770-971840-F023216-F023335/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.962] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P05-R01-PastaSalad-101915-123786-F002393-F003023/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.963] global loadsave.cpp:244 findDecoder imread_('dataset/images_rgb/P05-R01-PastaSalad-1178767-1181856-F028283-F028372/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.968] global loadsave.cpp:244 findDecoder imread_('dataset/images_flow/P05-R01-PastaSalad-1087378-1093795-F026082-F026267/u/0001.jpg'): can't open/read file: check file path/integrity
# [ WARN:0@1.968] global loadsave.cpp:244 findDecoder imread_('dataset/images_flow/P05-R01-PastaSalad-1087378-1093795-F026082-F026267/v/0001.jpg'): can't open/read file: check file path/integrity
# Traceback (most recent call last):
#   File "main.py", line 256, in <module>
#     main()
#   File "main.py", line 85, in main
#     test(test_loader, model_base, model_gaze, model_attn, num_action)
#   File "main.py", line 222, in test
#     for i, (rgb, flow, label) in enumerate(test_loader, 1):
#   File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py", line 681, in __next__
#     data = self._next_data()
#   File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py", line 1376, in _next_data
#     return self._process_data(data)
#   File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/dataloader.py", line 1402, in _process_data
#     data.reraise()
#   File "/usr/local/lib/python3.8/dist-packages/torch/_utils.py", line 461, in reraise
#     raise exception
# TypeError: Caught TypeError in DataLoader worker process 0.
# Original Traceback (most recent call last):
#   File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/_utils/worker.py", line 302, in _worker_loop
#     data = fetcher.fetch(index)
#   File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/_utils/fetch.py", line 49, in fetch
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "/usr/local/lib/python3.8/dist-packages/torch/utils/data/_utils/fetch.py", line 49, in <listcomp>
#     data = [self.dataset[idx] for idx in possibly_batched_index]
#   File "/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset.py", line 63, in __getitem__
#     rimg = rimg[..., ::-1]
# TypeError: 'NoneType' object is not subscriptable

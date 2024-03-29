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

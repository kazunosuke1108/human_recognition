import os
from glob import glob
with open("/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/EGTEA/test_split1.txt",'r') as f:
    lines=f.readlines()
    cnames, labels=[],[]
    for l in lines:
        cn,label,label2,label3=l.split(' ')
        cnames.append(cn) 
        labels.append(int(label)-1)
        if os.path.exists("/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_rgb/"+cn):
            with open("/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/EGTEA/test_split1_v2.txt",'a') as a:
                a.write(cn+" "+label+" "+label2+" "+label3)
        else:
            suggest=glob("/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/images_rgb/"+cn[:-16]+"*")
            print(len(suggest))
            with open("/home/hayashide/catkin_ws/src/third_party/Gaze-Attention/dataset/EGTEA/test_split1_v2.txt",'a') as a:
                a.write(os.path.basename(suggest[0])+" "+label+" "+label2+" "+label3)
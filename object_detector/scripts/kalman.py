import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dt=1/30
A=np.array([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]
],dtype=np.float64)
P=np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
],dtype=np.float64)
W=np.array([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
],dtype=np.float64)
V=np.array([
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
],dtype=np.float64)

for i in range(1,3):
    print(f"###### SEQUENCE {i-2} #####")
    P=np.dot(np.dot(A,P),np.linalg.pinv(A))-np.dot(np.dot(np.dot(np.dot(A,P),np.linalg.pinv(P+V)),P),A.T)+W
    H=np.dot(np.dot(A,P),np.linalg.pinv(P+V))
    print("P",P)
    print("H",H)

csv_path="/home/hayashide/kazu_ws/human_recognition/object_detector/csv/1004_detectron2.csv"
csv_data=np.loadtxt(csv_path,delimiter=",")
csv_data_pd=pd.DataFrame(csv_data).to_numpy()
print(csv_data_pd)

data_old=np.zeros_like(csv_data_pd[0])
comp_flg=False
history=[]
for i,data in enumerate(csv_data_pd):
    try:
        data_next=csv_data_pd[i+1]
    except IndexError:
        break
    
    if comp_flg:
        comp_flg=False
        pass
    else:
        X=np.array([
                data[1],
                data[2],
                (data[1]-data_old[1])/dt,
                (data[2]-data_old[2])/dt,
        ],dtype=np.float64).T
    
    if i==0:
        Xhat=X
    
    if np.isnan(data_next[1]) or np.isnan(data[1]) or np.isnan(data_old[1]):
        print(f"######SEQUENCE {i} X[{i}] -> X[{i+1}] #####")
        P=np.dot(np.dot(A,P),np.linalg.pinv(A))-np.dot(np.dot(np.dot(np.dot(A,P),np.linalg.pinv(P+V)),P),A.T)+W
        H=np.dot(np.dot(A,P),np.linalg.pinv(P+V))
        print("P",P)
        print("H",H)
        print("X",X)
        Xhat=np.dot(A,Xhat)+np.dot(H,X-Xhat)
        print("Xhat",Xhat)
        X=Xhat
        comp_flg=True


        data_old=data
    history.append([X[0],X[1]])
    pass
history=np.array(history)
print(np.array(history)[:,0])
plt.plot(history[:,0],history[:,1],label="kalman")
plt.plot(csv_data_pd[:,1],csv_data_pd[:,2],label="raw")
plt.legend()
plt.show()
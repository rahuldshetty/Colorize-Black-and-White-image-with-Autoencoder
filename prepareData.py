import numpy as np
import pandas as  pd
import os
import cv2


FILES="data/opencountry/"
clip_limit=3

data=pd.DataFrame()

def createBW(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return gray
        

def merge(l,a,b):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l)    
    merged_channels = cv2.merge((cl, a, b))
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return final_image

xtrain=[]
ytrain=[]
i=0
for file in os.listdir(FILES):
    if file.endswith("jpg")!=True:continue
    print(i,file)
    img=cv2.imread(FILES+file,1)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    xtrain.append(np.array(gray))
    ytrain.append(np.array(img))
    i+=1


data['x']=xtrain
data['y']=ytrain
data.to_pickle('data.pkl')
    

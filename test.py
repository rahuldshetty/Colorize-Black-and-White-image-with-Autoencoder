from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

autoencoder = load_model('finalmodel.h5')
'''
data=pd.read_pickle('data.pkl')
print("model loaded...")
x=data['x']
y=data['y']
'''
unseenClr=cv2.imread('unseen.jpg',1)
unseenBW=cv2.cvtColor(unseenClr,cv2.COLOR_BGR2GRAY)

unseenClr=unseenClr.astype('float32')/255
unseenBW1=unseenBW.astype('float32')/255
unseenBW=unseenBW1.copy()
unseenBW=unseenBW.reshape(1,256,256,1)

unseenBW=np.asarray(unseenBW)

cv2.imshow('B&W input',unseenBW1)
cv2.imshow('Real Image',unseenClr)

predicted=autoencoder.predict(unseenBW)
cv2.imshow('Predicted',predicted.reshape(256,256,3))

cv2.waitKey(0)
'''
xyy=[]
yyx=[]
for i in range(len(x)):
    x[i]=x[i].astype('float32')/255
    y[i]=y[i].astype('float32')/255
    x[i]=x[i].reshape(256,256,1)
    y[i]=y[i].reshape(256,256,3)
    xyy.append(x[i])
    yyx.append(y[i])
    
x=np.asarray(x)
y=np.asarray(y)
x=np.array(xyy)
y=np.array(yyx)
print("Predicting..")
while True:
    i=int(input("Enter id:"))
    img=x[i]
    img=img.reshape(1,256,256,1)
    predicted=autoencoder.predict(img)
    cv2.imshow('Input B&W',img.reshape(256,256,1))
    cv2.imshow('Real ',y[i])
    cv2.imshow('Output from the model',predicted.reshape(256,256,3))
    cv2.waitKey(0)
    s=input("Type to continue...")
    cv2.destroyAllWindows()
'''

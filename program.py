from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import plot_model
from keras import backend as K


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0.114])


data=pd.read_pickle('data.pkl')

x=data['x']
y=data['y']

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


img_rows = len(x)
img_cols = len(x[0])
channels = 3

LIM=int(0.80*len(x))
xtrain=x[0:LIM]
ytrain=y[0:LIM]
xtest=x[LIM:]
ytest=y[LIM:]


print(xtrain.shape)


input_shape=(256,256,1)
batch_size=32
kernel_size=3
latent_dim=256
layer_filters=[64,128,256]

inputs=Input(shape=input_shape,name='encoder_input')
x=inputs
for filters in layer_filters:
    x=Conv2D( filters=filters, kernel_size=kernel_size,strides=2,activation='relu',padding='same' )(x)
shape=K.int_shape(x)

x=Flatten()(x)
latent=Dense(latent_dim,name='latent_vector')(x)

encoder=Model(inputs,latent,name='encoder')
encoder.summary()

latent_inputs = Input( shape=(latent_dim,),name='decoder_input' )
x=Dense(shape[1]*shape[2]*shape[3])(latent_inputs)
x=Reshape((shape[1],shape[2],shape[3]))(x)

for filt in layer_filters[::-1]:
    x=Conv2DTranspose( filters=filt, kernel_size=kernel_size,strides=2,activation='relu',padding='same' )(x)
outputs=Conv2DTranspose(filters=channels,kernel_size=kernel_size,activation='sigmoid',padding='same',name='decoder_output')(x)
decoder=Model(latent_inputs,outputs,name='decoder')
decoder.summary()


autoencoder=Model(inputs,decoder(encoder(inputs)),name='autoencoder')
autoencoder.summary()
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),cooldown=0,patience=5,verbose=1,min_lr=0.5e-6)
model_name = 'colorized_ae_model.{epoch:03d}.h5'
filepath="saved/"
checkpoint = ModelCheckpoint(filepath=filepath+model_name,monitor='val_loss',verbose=1,save_best_only=True)
autoencoder.compile(loss='mse',optimizer='adam')

callbacks = [lr_reducer, checkpoint]

autoencoder.fit(xtrain,ytrain,validation_data=(xtest,ytest),epochs=30,batch_size=batch_size)

autoencoder.save('finalmodel.h5')

print("Model saved....")


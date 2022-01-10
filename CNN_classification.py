#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:56:57 2022

@author: sarramargi
"""

import numpy as np # linear algebra
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Flatten, InputLayer, concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization

import cv2 as cv
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import timeit

from tensorflow.keras.callbacks import TensorBoard
import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import train_test_split

# Test prepreocess data
PATH = './PokemonData'
classes = os.listdir(PATH)

c1_path=os.path.join(PATH, classes[1]) # different folders of Pokemon, considered as outputs
c1_data_path=[os.path.join(c1_path, img) for img in os.listdir(c1_path)]
len(c1_data_path)

# Preprocessing 
image_size = (256,256)
seed=5
validation_split=0.2
dataset_train,dataset_test = train_test_split(PATH,image_size,seed,validation_split)

# Normailisation
IDG = ImageDataGenerator(rescale = 1./255 )

train_data = IDG.flow_from_directory(PATH,target_size=(256,256),batch_size=32) 

img_shape=(256,256,3) # 3 car RGB

# Model | CNN Classique
model_classic_CNN = keras.Sequential(name='Classic_CNN')
model_classic_CNN.add(keras.layers.Conv2D(128,3,input_shape=(img_shape),activation='relu'))
model_classic_CNN.add(keras.layers.MaxPool2D())
model_classic_CNN.add(keras.layers.Conv2D(128,3,activation='relu'))
model_classic_CNN.add(keras.layers.MaxPool2D())
model_classic_CNN.add(keras.layers.Conv2D(128,3,strides=(2,2),activation='relu'))
model_classic_CNN.add(keras.layers.MaxPool2D())
model_classic_CNN.add(keras.layers.BatchNormalization())
model_classic_CNN.add(keras.layers.Conv2D(64,3,strides=(2,2),activation='relu'))
model_classic_CNN.add(keras.layers.MaxPool2D())
model_classic_CNN.add(keras.layers.Flatten())
model_classic_CNN.add(keras.layers.Dropout(0.2))
model_classic_CNN.add(keras.layers.Dense(1024,activation='relu'))
model_classic_CNN.add(keras.layers.Dense(512,activation='relu'))
model_classic_CNN.add(keras.layers.Dense(len(classes),activation='softmax'))

model_classic_CNN.summary()

model_classic_CNN.compile(optimizer='adam',
             loss='mse', #Changer loss
             metrics=['accuracy']
             )

hist = model_classic_CNN.fit_generator(train_data,epochs=2) #Changer nb epochs

plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['accuracy'],label='accuracy',color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, step=0.04))
plt.show()

plt.figure(figsize=(20,20))
#for _ in range(3):
sam_x,sam_y = next(train_data) 
pred_ = model_classic_CNN.predict(sam_x)
for i in range(len(sam_x)):
    pred,y = pred_[i].argmax(), sam_y[i].argmax()
    plt.subplot(4,4,i+1)
    plt.imshow(sam_x[i])
    title_ = 'Predict:' + str(classes[pred])+ ';   Label:' + str(classes[y])
    plt.title(title_,size=11)
plt.show()


#GridSearchCV 
# kn = KNeighborsClassifier()
print('params', model_classic_CNN.get_params())

params = {
    'n_neighbors' : [5, 25],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
}

grid_kn = GridSearchCV(estimator = kn,
                        param_grid = params,
                        scoring = 'accuracy', 
                        cv = 5, 
                        verbose = 1,
                        n_jobs = -1)
grid_kn.fit(X_train, y_train)


# Brouillon
plt.figure(figsize=(20,20))
#for _ in range(3):
sam_x,sam_y = next(train_data) 
pred_ = model_classic_CNN.predict(sam_x)
for i in range(len(sam_x)):
    pred,y = pred_[i].argmax(), sam_y[i].argmax()
    plt.subplot(4,4,i+1)
    plt.imshow(sam_x[i])
    title_ = 'Predict:' + str(classes[pred])+ ';   Label:' + str(classes[y])
    plt.title(title_,size=11)
plt.show()
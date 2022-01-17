#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:56:57 2022

@author: sarramargi
"""

import numpy as np # linear algebra
import tensorflow
from tensorflow import keras
from sklearn.utils import validation # linear algebra
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Flatten, InputLayer, concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

import cv2 as cv
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


from tensorflow.keras.utils import to_categorical
from tensorflow import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import os

from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import get_dataset 
from utils import get_dataset_v2
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt
# Test prepreocess data
PATH = './PokemonData'
classes = [f for f in os.listdir(PATH) if not f.startswith(".DS_Store")] # Deletes hidden file .DS_Store on Finder of MacOS

# c1_path=os.path.join(PATH, classes[1]) # different folders of Pokemon, considered as outputs
# c1_data_path=[os.path.join(c1_path, img) for img in os.listdir(c1_path)]
# len(c1_data_path)

# Preprocessing 
image_size = (150,150)
seed=5
validation_split=0.2
# shuffle='False'
# dataset_train,dataset_test = train_test_split(PATH,image_size,seed,validation_split)

if "dataset-{0}-{1}.npz".format(image_size[0],image_size[1]) not in os.listdir("./"):
    X,y = get_dataset_v2(PATH,image_size)
else: 
    dataset = np.load("dataset-{0}-{1}.npz".format(image_size[0],image_size[1]))
    X,y = dataset["x"],dataset["y"]
    #y = y.reshape((150,1))
print("Dataset Loaded!")

# Analysis of dataset
eval = pd.DataFrame(y,columns=["y"])["y"].value_counts()
eval = dict(eval)
names = eval.keys()
values = eval.values()
plt.figure()
plt.bar(names, values)

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split,random_state=seed)
y_train = to_categorical(y_train,len(classes))
y_test = to_categorical(y_test,len(classes))

#Show few images
plt.figure()
for i in range(8):
    plt.subplot(180+i+1)
    plt.imshow(X_train[i])
#plt.show()
# Normailisation

# X_train = X_train/255
# X_test = X_test/255
print(X_train.shape)
# IDG = ImageDataGenerator(rescale = 1./255 )

# train_data = IDG.flow_from_directory(PATH,target_size=(256,256),batch_size=32) 

#img_shape=(32,32,3) # 3 car RGB

# Model | CNN Classique
# model_classic_CNN = Sequential(name='Classic_CNN')
# model_classic_CNN.add(tensorflow.keras.layers.Conv2D(128,3,input_shape=(img_shape),activation='relu'))
# model_classic_CNN.add(tensorflow.keras.layers.MaxPool2D())
# model_classic_CNN.add(tensorflow.keras.layers.Conv2D(128,3,activation='relu'))
# model_classic_CNN.add(tensorflow.keras.layers.MaxPool2D())
# model_classic_CNN.add(tensorflow.keras.layers.Conv2D(128,3,strides=(2,2),activation='relu'))
# model_classic_CNN.add(tensorflow.keras.layers.MaxPool2D())
# model_classic_CNN.add(tensorflow.keras.layers.BatchNormalization())
# model_classic_CNN.add(tensorflow.keras.layers.Conv2D(64,3,strides=(2,2),activation='relu'))
# model_classic_CNN.add(tensorflow.keras.layers.MaxPool2D())
# model_classic_CNN.add(tensorflow.keras.layers.Flatten())
# model_classic_CNN.add(tensorflow.keras.layers.Dropout(0.2))
# model_classic_CNN.add(tensorflow.keras.layers.Dense(1024,activation='relu'))
# model_classic_CNN.add(tensorflow.keras.layers.Dense(512,activation='relu'))
# model_classic_CNN.add(tensorflow.keras.layers.Dense(len(classes),activation='softmax'))

# model_classic_CNN.summary()

# model_classic_CNN.compile(optimizer='adam',
#              loss='mse', #Changer loss
#              metrics=['accuracy']
#              )

# hist = model_classic_CNN.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=32) #Changer nb epochs

# Model CNN simple
model_CNN_simple = Sequential(name="Simple_CNN")

# conv layers

model_CNN_simple.add(Conv2D(128, 3, activation='relu', input_shape=X_train.shape[1:],name="Conv1"))
model_CNN_simple.add(MaxPooling2D(name="Pool1"))


model_CNN_simple.add(Conv2D(128, 3, activation='relu',name="Conv2"))
model_CNN_simple.add(MaxPooling2D(name="Pool2"))

model_CNN_simple.add(Conv2D(128, 3, strides=(2,2), padding='same', activation='relu',name="Conv3"))
model_CNN_simple.add(MaxPooling2D(pool_size=(2,2),name="Pool3"))

model_CNN_simple.add(BatchNormalization())
model_CNN_simple.add(Conv2D(64, 3,strides=(2,2), padding='same', activation='relu',name="Conv4"))
model_CNN_simple.add(MaxPooling2D(pool_size=(2,2),name="Pool4"))

model_CNN_simple.add(Flatten()) # flatten output of conv
model_CNN_simple.add(Dropout(0.25))

# hidden layers
model_CNN_simple.add(Dense(1024, activation='relu',name="Dense1"))

model_CNN_simple.add(Dense(512, activation='relu',name="Dense2"))

# output layer
model_CNN_simple.add(Dense(len(classes), activation='softmax',name="Dense3"))


model_CNN_simple.summary()

model_CNN_simple.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = model_CNN_simple.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=32) #Changer nb epochs

plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['accuracy'],label='accuracy',color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, step=0.04))
plt.show()

# plt.figure(figsize=(20,20))
# #for _ in range(3):
# sam_x,sam_y = next(train_data) 
# pred_ = model_classic_CNN.predict(sam_x)
# for i in range(len(sam_x)):
#     pred,y = pred_[i].argmax(), sam_y[i].argmax()
#     plt.subplot(4,4,i+1)
#     plt.imshow(sam_x[i])
#     title_ = 'Predict:' + str(classes[pred])+ ';   Label:' + str(classes[y])
#     plt.title(title_,size=11)
# plt.show()


# #GridSearchCV 
# # kn = KNeighborsClassifier()
# print('params', model_classic_CNN.get_params())

# params = {
#     'n_neighbors' : [5, 25],
#     'weights': ['uniform', 'distance'],
#     'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
# }

# grid_kn = GridSearchCV(estimator = kn,
#                         param_grid = params,
#                         scoring = 'accuracy', 
#                         cv = 5, 
#                         verbose = 1,
#                         n_jobs = -1)
# grid_kn.fit(X_train, y_train)


# # Brouillon
# plt.figure(figsize=(20,20))
# #for _ in range(3):
# sam_x,sam_y = next(train_data) 
# pred_ = model_classic_CNN.predict(sam_x)
# for i in range(len(sam_x)):
#     pred,y = pred_[i].argmax(), sam_y[i].argmax()
#     plt.subplot(4,4,i+1)
#     plt.imshow(sam_x[i])
#     title_ = 'Predict:' + str(classes[pred])+ ';   Label:' + str(classes[y])
#     plt.title(title_,size=11)
# plt.show()

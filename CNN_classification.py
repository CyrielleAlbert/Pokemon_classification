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
from tensorflow.keras.models import Sequential, Model, load_model
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
from utils import get_test_dataset
from sklearn.model_selection import train_test_split 
import pandas as pd
import matplotlib.pyplot as plt

# GLOBAL VARIABLE
PATH = './PokemonData'
classes = [f for f in os.listdir(PATH) if not f.startswith(".DS_Store")] # Deletes hidden file .DS_Store on Finder of MacOS

image_size = (150,150)
seed=5
validation_split=0.2

def preprocessing(PATH, image_size=(150,150),seed=5, validation_split=0.2, reduce_dataset=False,reduction_ratio=None):
    """ A function that load the dataset and split it into training and validation data

        Parameters:
            PATH                (string)    : the path of the dataset
            image_size          (tuple)     : the size to reshape all the imported images
            seed                (int)       : set the randomisation of train_test_split
            validation_split    (float 0-1) : define the proportion of validation data and training data
            reduce_dataset      (bool)      : if True, reduce the size of the dataset to ease model training, needs a reduction_ratio (default False)
            reduction_ratio     (float 0-1) : define the proportion of data kept for the dataset, needs reduce_dataset set to True (default None)
    
        Returns:
            X_train (numpy.ndarray): Training partition of images data
            y_train (numpy.ndarray): Training partition of images categories
            X_test  (numpy.ndarray): Validation partition of images data
            y_test  (numpy.ndarray): Validation partition of images categories

    """
    
    if "dataset-{0}-{1}.npz".format(image_size[0],image_size[1]) not in os.listdir("./Datasets"):
        X,y = get_dataset_v2(PATH,image_size)
    else: 
        dataset = np.load("./Datasets/dataset-{0}-{1}.npz".format(image_size[0],image_size[1]))
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
    
    # Train-test-split and dataset reduction
    if reduce_dataset:
        X_reduced,y_reduced = get_test_dataset(X,y,reduction_ratio)
        X_train, X_test, y_train, y_test = train_test_split(X_reduced, y_reduced, test_size=validation_split,random_state=seed,stratify=y_reduced)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_split,random_state=seed,stratify=y)

    ## Analysis of dataset
    eval = pd.DataFrame(y_train,columns=["y_train"])["y_train"].value_counts()
    eval = dict(eval)
    names = eval.keys()
    values = eval.values()
    plt.figure()
    plt.bar(names, values)
    plt.show()

    # Prepare category    
    y_train = to_categorical(y_train,len(classes))
    y_test = to_categorical(y_test,len(classes))

    #Show few images
    plt.figure()
    for i in range(8):
        plt.subplot(180+i+1)
        plt.imshow(X_train[i])
    plt.show()

    return X_train,y_train,X_test,y_test


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



# conv layers
def basic_CNN_model(input_shape,activation='relu',nb_conv_layer=1):
    """ Create a simple CNN model architecture

            Parameters:
                input_shape         (tuple)     : Shape of one image
                activation          (string)    : Activation function of the Conv Layers. Refer to Con2D function to know more (default 'relu')
                nb_conv_layer       (int)       : Number of convolution layers in the hidden layer (default 1)
            Returns:
                model_CNN_simple    (tf.Sequential) : A simple CNN model (need to be trained)
    """
    # Model CNN simple
    model_CNN_simple = Sequential(name="Simple_CNN")
    model_CNN_simple.add(Conv2D(128, 3, activation=activation, input_shape=input_shape,name="Conv1"))
    model_CNN_simple.add(MaxPooling2D(name="Pool1"))

    for i in range(nb_conv_layer):
        model_CNN_simple.add(Conv2D(128, 3, activation=activation,name="Conv{}".format(i+2)))
        model_CNN_simple.add(MaxPooling2D(name="Pool{}".format(i+2)))

    model_CNN_simple.add(BatchNormalization())
    model_CNN_simple.add(Conv2D(64, 3,strides=(2,2), activation=activation))
    model_CNN_simple.add(MaxPooling2D(pool_size=(2,2)))

    model_CNN_simple.add(Flatten()) # flatten output of conv
    model_CNN_simple.add(Dropout(0.25))

    # hidden layers
    model_CNN_simple.add(Dense(1024, activation=activation,name="Dense1"))

    model_CNN_simple.add(Dense(512, activation=activation,name="Dense2"))

    # output layer
    model_CNN_simple.add(Dense(len(classes), activation='softmax',name="Dense3"))
    return model_CNN_simple

def run():
    X_train,y_train,X_test,y_test = preprocessing(PATH="./PokemonData",image_size=(150,150))
    
    if "model_and_weight-{0}-{1}-{2}.h5".format('adam',image_size[0],image_size[1]) not in os.listdir("./models"):
        model_CNN_simple = basic_CNN_model(X_train[1:])
        model_CNN_simple.summary()
        model_CNN_simple.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        hist = model_CNN_simple.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=30,batch_size=32) #Changer nb epochs
        model_CNN_simple.save("./models/model_and_weight-{0}-{1}-{2}.h5".format('adam',image_size[0],image_size[1]))
    else : 
        filepath = "./models/"+"model_and_weight-{0}-{1}-{2}.h5".format('adam',image_size[0],image_size[1])
        model_CNN_simple = load_model(filepath)
        loss,acc = model_CNN_simple.evaluate(X_test,y_test,verbose=2)
        print(loss,acc)

    # plt.style.use('fivethirtyeight')
    # plt.figure(figsize=(14,14))
    # plt.plot(hist.history['accuracy'],label='accuracy',color='green')
    # plt.plot(hist.history['val_accuracy'],label='accuracy',color='red')
    # plt.legend(["training data","validation data"])
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.yticks(np.arange(0, 1, step=0.04))
    # plt.show()

run()

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

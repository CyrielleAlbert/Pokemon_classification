#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 13:56:57 2022

@author: sarramargi
"""

import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Dense, Flatten, InputLayer, concatenate
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization

from tensorflow import keras
#from keras.layers.merge import Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import timeit

from tensorflow.keras.callbacks import TensorBoard
import datetime

# Preprocessing Code Cyrielle 


img_shape= (i_shape, i_shape, 3) # 3 car RGB

# Model | CNN Classique
model = keras.Sequential(name='Classic_CNN')
model.add(keras.layers.Conv2D(128,3,input_shape=(img_shape),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(128,3,activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Conv2D(128,3,strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Conv2D(64,3,strides=(2,2),activation='relu'))
model.add(keras.layers.MaxPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(1024,activation='relu'))
model.add(keras.layers.Dense(512,activation='relu'))
model.add(keras.layers.Dense(len(classes),activation='softmax'))

#GridSearchCV 



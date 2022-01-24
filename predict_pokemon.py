#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:52:14 2022
@author: sarramargi
"""

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
# from preprocess_img import preprocess_img


from utils import load_prediction_img
    
# %% Load the image
img_size=(150,150,3)
prediction_img = load_prediction_img('./Voltorb_test.jpeg',img_size)
prediction_img = np.expand_dims(prediction_img, axis=0)

#%% Load the model
filepath = 'model_denseNet201_pokemon.h5' # MAJ du modèle
model_denseNet201 = load_model(filepath)

# loss,acc = model_DenseNet.evaluate(X_test,y_test,verbose=2) ###

#%% Generate predictions for samples
predictions = model_denseNet201.predict(prediction_img)
print(predictions)

# # # # Check if predictions contains 1 which corresponds to the mask
# one_value = np.argwhere(predictions>0)
# print(one_value)

# prediction_img = np.expand_dims(prediction_img, axis=0)
# category = model_denseNet201.predict(prediction_img, verbose=1)
label = np.argmax(predictions,axis=1)
print(type(label),label)
# print(classes[label[0]]) # Pas compris 
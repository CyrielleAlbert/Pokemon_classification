#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:05:09 2022

@author: sarramargi
"""

# Load librairies 
from sklearn.model_selection import KFold, cross_val_score
from denseNet121_classification import create_denseNet121
from denseNet201_classification import create_denseNet201
from denseNet169_classification import create_denseNet169
from resNet151_classification import create_resNet152
from CNN_simple_classification import create_CNN_simple
from unet_classification import create_unet
import os
from utils import get_dataset_v2, preprocessing
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from matplotlib import pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau

# Load data 
image_size = (150,150)
PATH = './PokemonData'
classes = [f for f in os.listdir(PATH) if not f.startswith(".DS_Store")] # Deletes hidden file .DS_Store on Finder of MacOS


if "dataset-{0}-{1}.npz".format(image_size[0],image_size[1]) not in os.listdir("./"):
    X,y = get_dataset_v2(PATH,image_size)
else: 
    dataset = np.load("dataset-{0}-{1}.npz".format(image_size[0],image_size[1]))
    X,y = dataset["x"],dataset["y"]
print("Dataset Loaded!")


# Pre-process data
X_train,y_train,X_val,y_val,X_test,y_test=preprocessing(X,y)


# # Initialize all models
# create_denseNet121()
# create_denseNet201()
# create_denseNet169()
# create_resNet()

# Buil models as classifiers
denseNet121 = KerasClassifier(build_fn=create_denseNet121(), epochs=10, batch_size=32)
denseNet201 = KerasClassifier(build_fn=create_denseNet201(), epochs=10, batch_size=32)
denseNet169 = KerasClassifier(build_fn=create_denseNet169(), epochs=10, batch_size=32)
resNet152 = KerasClassifier(build_fn=create_resNet152(), epochs=10, batch_size=32)
simple_CNN = KerasClassifier(build_fn=create_CNN_simple(), epochs=10, batch_size=32)
u_net = KerasClassifier(build_fn=create_unet(), epochs=10, batch_size=32)

# Cross-validation with K-fold
models=[]
results=[]
names=[]
num_classes=len(classes)
scoring='accuracy'

models.append(('DenseNet121',denseNet121))
models.append(('DenseNet201',denseNet201))
models.append(('DenseNet169',denseNet169))
models.append(('ResNet',resNet152))
models.append(('Simple CNN',simple_CNN))
models.append(('U-Net',u_net))


for name,model in models :
    kfold= KFold(n_splits=num_classes)
    cv_results=cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    name.append(name)
    print('%s: %f %f', name, cv_results.mean(), cv_results.std())


# Keep best model
plt.figure()
plt.boxplot(results,labels=name) #MAJ

# Tuning #MAJ

# Run meilleur modele avec hyperparamètres fixés - denseNet169
filepath= "model_denseNet169_pokemon.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

early_stopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)

callbacks_list = [
        checkpoint,
        early_stopping,
        learning_rate_reduction
    ]

hist = denseNet169.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=32, callbacks=callbacks_list)

# Plot accuracy metric
plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['accuracy'],label='accuracy',color='green')
plt.plot(hist.history['val_accuracy'],label='accuracy',color='red')
plt.legend(["training data","validation data"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, step=0.04))
plt.show()
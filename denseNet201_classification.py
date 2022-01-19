#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 18:34:07 2022

@author: sarramargi
"""

import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
from utils import train_test_split2

# Test prepreocess data
PATH = './PokemonData'
classes = [f for f in os.listdir(PATH) if not f.startswith(".DS_Store")] # Deletes hidden file .DS_Store on Finder of MacOS

# c1_path=os.path.join(PATH, classes[1]) # different folders of Pokemon, considered as outputs
# c1_data_path=[os.path.join(c1_path, img) for img in os.listdir(c1_path)]
# len(c1_data_path)

# Preprocessing 
image_size = (32,32)
seed=5
validation_split=0.2
# shuffle='False'
# dataset_train,dataset_test = train_test_split(PATH,image_size,seed,validation_split)

if "dataset-{0}-{1}.npz".format(image_size[0],image_size[1]) not in os.listdir("./"):
    X,y = train_test_split2(PATH,image_size,seed,validation_split)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=10)
y_train = to_categorical(y_train,len(classes))
y_test = to_categorical(y_test,len(classes))

#Show few images
plt.figure()
for i in range(8):
    plt.subplot(180+i+1)
    plt.imshow(X_train[i])
#plt.show()


# Normailisation
X_train = X_train/255
X_test = X_test/255
print(X_train.shape)

img_size = 32
base_model = DenseNet201(include_top = False,
                         weights = 'imagenet',
                         input_shape = (img_size,img_size,3))

for layer in base_model.layers[:675]:
    layer.trainable = False

for layer in base_model.layers[675:]:
    layer.trainable = True

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(len(classes), activation=tf.nn.softmax))
model.compile(optimizer = tf.keras.optimizers.Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics=['accuracy'])

filepath= "model_pokemon.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', save_weights_only=False)

early_stopping = EarlyStopping(monitor='val_loss',min_delta = 0, patience = 5, verbose = 1, restore_best_weights=True)

learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.2, 
                                            min_lr=0.00001)

callbacks_list = [
        checkpoint,
        early_stopping,
        learning_rate_reduction
    ]

hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=32, callbacks=callbacks_list)


# hist = model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),
#                                         validation_data=testgen.flow(X_test,y_test,batch_size=32),
#                                         epochs=50,
#                                         callbacks=callbacks_list)
# DenseNet-121
# DenseNet-169
# DenseNet-201
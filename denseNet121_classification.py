#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 17:32:43 2022
@author: sarramargi
"""

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
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os
from utils import get_dataset_v2

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

if "dataset-{0}-{1}.npz".format(image_size[0],image_size[1]) not in os.listdir("./"):
    X,y = get_dataset_v2(PATH,image_size)
else: 
    dataset = np.load("dataset-{0}-{1}.npz".format(image_size[0],image_size[1]))
    X,y = dataset["x"],dataset["y"]
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

# Normailisation
X_train = X_train/255
X_test = X_test/255
print(X_train.shape)

img_size = 150
base_model = DenseNet121(include_top = False,
                         weights = 'imagenet',
                         input_shape = (img_size,img_size,3))

for layer in base_model.layers[:-8]:
    layer.trainable = False

for layer in base_model.layers[-8:]:
    layer.trainable = True

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(len(classes), activation=tf.nn.softmax))
model.compile(optimizer ='Ftrl', loss ='categorical_crossentropy', metrics=['accuracy'])

filepath= "model_denseNet121_pokemon.h5"
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

plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['accuracy'],label='accuracy',color='green')
plt.plot(hist.history['val_accuracy'],label='accuracy',color='red')
plt.legend(["training data","validation data"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, step=0.04))
plt.show()

# hist = model.fit_generator(datagen.flow(X_train,y_train,batch_size=32),
#                                         validation_data=testgen.flow(X_test,y_test,batch_size=32),
#                                         epochs=50,
#                                         callbacks=callbacks_list)
# DenseNet-121 # Test sur PC école 300
# DenseNet-169 # Envoi Cyrielle
# DenseNet-201 # Test sur MAc 300
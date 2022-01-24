#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:05:09 2022

@author: sarramargi
"""

# Load librairies 
from sklearn.model_selection import StratifiedKFold,KFold, cross_val_score
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
from sklearn.model_selection import RandomizedSearchCV


# Load data 
image_size = (150,150)
PATH = './PokemonData'
classes = [f for f in os.listdir(PATH) if not f.startswith(".DS_Store")] # Deletes hidden file .DS_Store on Finder of MacOS


if "dataset-{0}-{1}.npz".format(image_size[0],image_size[1]) not in os.listdir("./Datasets"):
    X,y = get_dataset_v2(PATH,image_size)
else: 
    dataset = np.load("./Datasets/dataset-{0}-{1}.npz".format(image_size[0],image_size[1]))
    X,y = dataset["x"],dataset["y"]
print("Dataset Loaded! ✅")


# Pre-process data
X_train,y_train,X_val,y_val,X_test,y_test=preprocessing(X,y)


# # Initialize all models
# create_denseNet121()
# create_denseNet201()
# create_denseNet169()
# create_resNet()

# Buil models as classifiers
denseNet121 = KerasClassifier(build_fn=create_denseNet121, epochs=4, batch_size=32)
denseNet201 = KerasClassifier(build_fn=create_denseNet201, epochs=4, batch_size=32)
denseNet169 = KerasClassifier(build_fn=create_denseNet169, epochs=4, batch_size=32)
resNet152 = KerasClassifier(build_fn=create_resNet152, epochs=4, batch_size=32)
simple_CNN = KerasClassifier(build_fn=create_CNN_simple, epochs=30, batch_size=32)
u_net = KerasClassifier(build_fn=create_unet, epochs=10, batch_size=32)

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

print("------------------------------------------------\n----------------CROSS VALIDATION----------------\n------------------------------------------------")

for name,model in models :
    kfold= KFold(n_splits=num_classes)
    cv_results=cross_val_score(model, X_train, np.argmax(y_train, axis=-1), cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('%s: %f %f', name, cv_results.mean(), cv_results.std())


# Keep best model
plt.figure()
plt.boxplot(results,labels=names) #MAJ

# Initialize hyperparameters for Tuning
poolings = ['avg','max']
optimizers = ['adam', 'sgd', 'adagrad']
index_training =[600,675]
epochs = [10]
batches = [32,64]
n_combinaisons = 5
cv = 3

# Apply Tuning 

param_grid = dict(pooling=poolings, index_trainable=index_training, epochs=epochs, batch_size=batches,opti=optimizers)
random_search  = RandomizedSearchCV(estimator=denseNet169, param_distributions=param_grid, n_iter=n_combinaisons, cv=cv, verbose=2,error_score='raise',n_jobs=-1)
random_search_result = random_search.fit(X_train, y_train, validation_data=(X_val, y_val))

# summarize results
print("------------------------------------------------\n-----------------TUNING RESULTS-----------------\n------------------------------------------------")

print("Best: %f using %s" % (random_search_result.best_score_, random_search_result.best_params_))
means = random_search_result.cv_results_['mean_test_score']
stds = random_search_result.cv_results_['std_test_score']
params = random_search_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
test_result = random_search.score(X_test, y_test)
print("CNN test Error is %.2f%% with best param" % (100-test_result*100))


# Run meilleur modele avec hyperparamètres fixés - denseNet169

best_model = create_denseNet169(pooling='max', opti= 'adam', index_training= 600)

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

hist = best_model.fit(X_train,y_train,validation_data=(X_val,y_val),epochs=50,batch_size=32, callbacks=callbacks_list)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("------------------------------------------------\n------------------FINAL RESULTS-----------------\n------------------------------------------------")

print("Error: %.2f%%" % (100-scores[1]*100))

print("------------------------------------------------\n----------------------END-----------------------\n------------------------------------------------")

# Plot accuracy metric
plt.style.use('fivethirtyeight')
plt.figure(figsize=(14,14))
plt.plot(hist.history['accuracy'],label='accuracy',color='green')
plt.plot(hist.history['val_accuracy'],label='accuracy',color='red')
plt.legend(["training data","validation data"])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.yticks(np.arange(0, 1, step=0.04))

plt.figure(figsize=(14,14))
plt.plot(hist.history['loss'],label='loss',color="green")
plt.plot(hist.history['val_loss'],label='loss',color='red')
plt.grid()
plt.title('Training history')
plt.legend(['Train', 'Test'])
plt.xlabel('Epochs')
plt.ylabel('Cost function')
plt.show()

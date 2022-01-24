#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 22:05:09 2022

@author: sarramargi
"""

# Load librairies 
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from denseNet121_classification import create_denseNet121
from denseNet201_classification import create_denseNet201

# Load data 

# Pre-process

# Call model 

# Kfold -> recréer un entraînement !!! - 
models=[]
results=[]
names=[]
num_classes=150
scoring='accuracy'

denseNet121_model = create_denseNet121()
denseNet210_model = create_denseNet201()

models.append(('DenseNet121', denseNet121_model))
models.append(('DenseNet201', denseNet210_model)) # MAJ
models.append(('DenseNet169', make_pipeline()))
models.append(('U-net', make_pipeline()))
models.append(('ResNet', make_pipeline()))

for name,model in models :
    kfold= KFold(n_splits=num_classes)
    cv_results=cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print('%s: %f %f', name, cv_results.mean(), cv_results.std())

# Choix du meilleur modele 

# Tuning 


# Run meilleur modele avec hyperparamètres fixés

# 





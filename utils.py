﻿from os import listdir
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
import numpy as np

PATH_DATASET = "./PokemonData/"

def train_test_split(path,img_size,seed,validation_split):
    labels = listdir(path)
    dataset_train = image_dataset_from_directory(path,label_mode = 'int',labels="inferred",class_names=labels, validation_split=validation_split,image_size=img_size,
    subset="training",seed=seed)
    dataset_valid = image_dataset_from_directory(path,label_mode = 'int',labels="inferred",class_names=labels, validation_split=validation_split,image_size=img_size,
    subset="validation",seed=seed)
    
    return dataset_train,dataset_valid

def train_test_split2(path,img_size,seed,validation_split):
    labels = listdir(path)
    X = []
    y = []
    i = 0
    for label in labels:
        for image in listdir(path+'/'+label):
            image_path = path+'/'+label+'/'+image 
            try: 
                print(image_path)
                img = load_img(image_path,target_size=img_size,interpolation='nearest')
                img_arr = img_to_array(img)
                img_arr = img_arr / 255
                X.append(img_arr)
                y.append(i)
            except Exception as e:
                print(e)
        i +=1
    print(np.array(X).shape,len(y))

    return np.array(X),np.array(y)

PATH = './PokemonData'
image_size = (150,150)
seed=5
validation_split=0.2
X,y = train_test_split2(PATH,image_size,seed,validation_split)

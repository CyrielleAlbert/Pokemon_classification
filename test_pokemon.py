import os
from tensorflow.keras.models import load_model
from utils import load_prediction_img
import numpy as np

# Load the model
filepath = './models/model_DenseNet121_and_weight-adam-150-150.h5' # MAJ du modèle
model_denseNet169 = load_model(filepath)

# Define all possible categories
classes =  [f for f in os.listdir("./PokemonData") if not f.startswith(".DS_Store")] # Deletes hidden file .DS_Store on Finder of MacOS

def test_prediction(image_arr):

    predictions = model_denseNet169.predict(image_arr)
    category_predicted = np.argmax(predictions,axis=1)[0]
    category_name = classes[category_predicted]
    return category_name

def test_equal():
    nb_pred = 0
    nb_error = 0
    for pokemon_name in classes:
        img_size=(150,150,3)
        image_loaded = True
        try :
            image = load_prediction_img('./Test_data/images/images/'+pokemon_name+'.jpg',img_size)
        except :
            try:
                image = load_prediction_img('./Test_data/images/images/'+pokemon_name+'.png',img_size)
            except:
                print("no such file",pokemon_name)
                image_loaded = False
        if image_loaded:
            nb_pred +=1
            image_arr = np.expand_dims(image, axis=0)
            pred = test_prediction(image_arr)
            if  pred != pokemon_name:
                "Error: {0} != {1}".format(pred,pokemon_name)
                nb_error += 1
            else: 
                "Success: {0} == {1}".format(pred,pokemon_name)
        print("{0} failure out of {1}".format(nb_error,nb_pred))

test_equal()


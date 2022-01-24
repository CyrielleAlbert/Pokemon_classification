import os
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report
from utils import preprocessing


def test_with_image_from_dataset():
    # Load the model
    image_size=(150,150)
    filepath = './models/model_DenseNet169_and_weight-adam-150-150.h5' # MAJ du modèle
    model_denseNet169 = load_model(filepath)
    dataset = np.load("./Datasets/dataset-{0}-{1}.npz".format(image_size[0],image_size[1]))
    X,y = dataset["x"],dataset["y"]
    # Define all possible categories
    classes =  [f for f in os.listdir("./PokemonData") if not f.startswith(".DS_Store")] # Deletes hidden file .DS_Store on Finder of MacOS

    _,_,_,_,X_test,y_test = preprocessing(X,y)
    y_pred = model_denseNet169.predict(X_test)
    pred = np.argmax(y_pred,axis=1)
    ground = np.argmax(y_test,axis=1)
    print(classification_report(ground,pred,target_names = classes))


test_with_image_from_dataset()

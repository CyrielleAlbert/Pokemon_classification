from os import listdir
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
PATH_DATASET = "./PokemonData/"

def get_dataset(path, img_size,seed,validation_split):
    labels = [ f for f in listdir(path) if not f.startswith(".DS_Store")]
    dataset_train = image_dataset_from_directory(path, labels="inferred", label_mode = 'int',class_names=labels, validation_split=validation_split,image_size=img_size,
    subset="training",seed=seed)
    dataset_valid = image_dataset_from_directory(path, labels="inferred", label_mode = 'int', class_names=labels, validation_split=validation_split,image_size=img_size,
    subset="validation",seed=seed)
    
    return dataset_train,dataset_valid

def get_dataset_v2(path,img_size):
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
    np.savez("./Datasets/dataset-{0}-{1}.npz".format(img_size[0],img_size[1]) ,x=X,y=y)
    return np.array(X),np.array(y)

# PATH = './PokemonData'
# image_size = (150,150)
# seed=5
# validation_split=0.2
# X,y = train_test_split2(PATH,image_size,seed,validation_split)

def get_test_dataset(X,y,reduction_rate):
    X_reduced,_,y_reduced,_ = train_test_split(X,y, train_size=reduction_rate,stratify=y)
    print("Reduced")
    return X_reduced,y_reduced


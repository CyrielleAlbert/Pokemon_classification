from os import listdir
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

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

def load_prediction_img(image_path,img_size):
    img = load_img(image_path,target_size=img_size,interpolation='nearest')
    img_arr = img_to_array(img)
    img_arr = img_arr / 255
    return img_arr

def preprocessing(PATH, image_size=(150,150),seed=5, validation_split=0.2, reduce_dataset=False,reduction_ratio=None):
    """ A function that load the dataset and split it into training and validation data

        Parameters:
            PATH                (string)    : the path of the dataset
            image_size          (tuple)     : the size to reshape all the imported images
            seed                (int)       : set the randomisation of train_test_split
            validation_split    (float 0-1) : define the proportion of validation data and training data
            reduce_dataset      (bool)      : if True, reduce the size of the dataset to ease model training, needs a reduction_ratio (default False)
            reduction_ratio     (float 0-1) : define the proportion of data kept for the dataset, needs reduce_dataset set to True (default None)
    
        Returns:
            X_train (numpy.ndarray): Training partition of images data
            y_train (numpy.ndarray): Training partition of images categories
            X_test  (numpy.ndarray): Validation partition of images data
            y_test  (numpy.ndarray): Validation partition of images categories

    """
    
    if "dataset-{0}-{1}.npz".format(image_size[0],image_size[1]) not in os.listdir("./Datasets"):
        X,y = get_dataset_v2(PATH,image_size)
    else: 
        dataset = np.load("./Datasets/dataset-{0}-{1}.npz".format(image_size[0],image_size[1]))
        X,y = dataset["x"],dataset["y"]
    print("Dataset Loaded!")

    # Train-test-split and dataset reduction
    if reduce_dataset:
        X_reduced,y_reduced = get_test_dataset(X,y,reduction_ratio)
        X_train, X_test_val, y_train, y_test_val = train_test_split(X_reduced, y_reduced, test_size=0.3,random_state=seed,stratify=y_reduced)
        X_val, X_test,y_val,y_test = train_test_split(X_test_val,y_test_val,test_size=0.5,random_state=seed,stratify=y_test_val)
    else:
        X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=0.3,random_state=seed,stratify=y)
        X_val, X_test,y_val,y_test = train_test_split(X_test_val,y_test_val,test_size=0.5,random_state=seed,stratify=y_test_val)

    # Prepare category    
    y_train = to_categorical(y_train,len(y))
    y_test = to_categorical(y_test,len(y))
    y_val = to_categorical(y_val,len(y))

    return X_train,y_train,X_val,y_val,X_test,y_test
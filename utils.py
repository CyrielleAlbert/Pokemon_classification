from os import listdir
from tensorflow.keras.preprocessing import image_dataset_from_directory

PATH_DATASET = "./PokemonData/"

def train_test_split(path, img_size,seed,validation_split):
    labels = [ f for f in listdir(path) if not f.startswith(".DS_Store")]
    dataset_train = image_dataset_from_directory(path, labels="inferred", label_mode = 'int',class_names=labels, validation_split=validation_split,image_size=img_size,
    subset="training",seed=seed)
    dataset_valid = image_dataset_from_directory(path, labels="inferred", label_mode = 'int', class_names=labels, validation_split=validation_split,image_size=img_size,
    subset="validation",seed=seed)
    
    return dataset_train,dataset_valid
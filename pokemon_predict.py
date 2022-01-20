#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 10:42:53 2022

@author: sarramargi
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:52:14 2022

@author: sarramargi
"""

from tensorflow.keras.models import load_model
import matplotlib.image as mpimg
import numpy as np

#%% Pre-processing step
def preprocess_img(img_init):
    img_reshaped = img_init.reshape
    return img_prepro

#%% Load the model
# File path
filepath = './unetFit.h5'

model = load_model(filepath)

# Generate predictions for samples
image_test = mpimg.imread('./image_predict_1D.bmp')
image_test_reshaped = image_test.reshape(1, 512,512, 1).astype('float32')
image_test_reshaped =image_test_reshaped / 255
predictions = model.predict(image_test_reshaped)
print(predictions)

# Check if predictions contains 1 which corresponds to the mask 
one_value = np.argwhere(predictions>0)
print(one_value)

#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 10:52:14 2022

@author: sarramargi
"""

from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np
from numpy import asarray
from preprocess_img import preprocess_img
from PIL import Image
from matplotlib import image
import skimage.io
import cv2

#%% Image pre-process
# Load the image
img_test=cv2.imread('./Pikatchu_test.png', cv2.IMREAD_GRAYSCALE)
# Process the data 
img_process = preprocess_img(img_test)

#%% Load the model
filepath = 'model_denseNet201_pokemon.h5' # MAJ du modèle
model = load_model(filepath)

#%% Generate predictions for samples
predictions = model.predict(img_process)
print(predictions)

# # Check if predictions contains 1 which corresponds to the mask
one_value = np.argwhere(predictions>0)
print(one_value)

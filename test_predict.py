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

# Pre-processing step
def preprocess_img():
    return img_prepro

# File path
filepath = './unetFit.h5'

# Load the model
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
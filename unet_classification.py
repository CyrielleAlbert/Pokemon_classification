"""
Created on Tue Jan 10 15:17:28 2022
@author: sarramargi
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate
from tensorflow.keras.layers import Conv2DTranspose

from tensorflow import keras

# %% U-Net CNN model

init = 'GlorotUniform' # 'HeNormal' à la base
acti = 'relu' # 'relu' à la base
optimizer= 'Adam'
img_size = (512,512)

# definition d'un bloc encodeur 
def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.1, max_pooling=True,activation=acti,initializer=init):
    conv = Conv2D(n_filters, 3, activation=acti, kernel_initializer=init, padding='same')(inputs) # kernel_initializer='HeNormal'
    conv = Conv2D(n_filters, 3, activation=acti, kernel_initializer=init, padding='same')(conv)
    #conv = BatchNormalization()(conv, training=False)
    
    if dropout_prob > 0:     
        conv = Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = MaxPooling2D(pool_size = (2,2))(conv)    
    else:
        next_layer = conv
    
    skip_connection = conv
    return next_layer, skip_connection

# definition d'un bloc décodeur 
def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32,activation=acti,initializer=init):
    up = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(prev_layer_input)
    merge = concatenate([up, skip_layer_input])
    conv = Conv2D(n_filters, 3, activation=acti, padding='same', kernel_initializer=init)(merge)
    conv = Conv2D(n_filters, 3, activation=acti, padding='same', kernel_initializer=init)(conv)
    return conv

def create_unet():
    inputs = keras.Input(shape=img_size + (1,))
    #inputs = InputLayer(input_shape=(512,512,1))
    [next_layer1, skip_connection1] = EncoderMiniBlock(inputs)
    [next_layer2, skip_connection2] = EncoderMiniBlock(next_layer1,n_filters=32)
    [next_layer3, skip_connection3] = EncoderMiniBlock(next_layer2,n_filters=64)
    [next_layer4, skip_connection4] = EncoderMiniBlock(next_layer3,max_pooling=False,n_filters=128)

    conv1 = DecoderMiniBlock(next_layer4,skip_connection3, n_filters=128)
    conv2 = DecoderMiniBlock(conv1,skip_connection2, n_filters=64)
    conv3 = DecoderMiniBlock(conv2,skip_connection1, n_filters=32)

    #[next_layer4, skip_connection4] = EncoderMiniBlock(next_layer3)
    #conv4 = DecoderMiniBlock(conv3,skip_connection4)

    outputs = Conv2D(1, 1, padding="same", activation="softmax")(conv3)

    model = Model(inputs, outputs, name="U-Net")
    model.compile(loss="categorical_crossentropy",optimizer=optimizer, metrics=['accuracy']) # Loss = 'mse'
    return model

# IDG = ImageDataGenerator(rescale = 1./255 )
# train_data = IDG.flow_from_directory(PATH,target_size=(256,256),batch_size=32) 
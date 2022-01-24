from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import DenseNet169

def create_denseNet169(num_classes=150,input_shape=(150,150,3),opti='adam',index_trainable=675,pooling='avg'):
    model = Sequential(name="DenseNet169")
    model.add(DenseNet169(include_top=False,input_shape=input_shape,pooling=pooling))
    for layer in model.layers[:index_trainable]:
        layer.trainable = False

    for layer in model.layers[index_trainable:]:
        layer.trainable = True
    model.add(Dense(num_classes, activation='softmax',name="Dense3"))
    model.summary()
    model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
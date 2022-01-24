from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import DenseNet169

def create_denseNet169(num_classes=150,input_shape=(150,150,3)):
    model = Sequential(name="DenseNet169")
    model.add(DenseNet169(include_top=False,input_shape=input_shape,pooling="avg"))
    for layer in model.layers[:675]:
        layer.trainable = False

    for layer in model.layers[675:]:
        layer.trainable = True
    model.add(Dense(num_classes, activation='softmax',name="Dense3"))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
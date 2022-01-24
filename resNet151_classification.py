from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import ResNet152

def create_resNet152(num_classes,input_shape):
    model= Sequential(name="ResNet152")
    model.add(ResNet152(include_top=False,input_shape=input_shape,pooling="avg"))
    for layer in model.layers[:143]:
        layer.trainable = False

    for layer in model.layers[143:]:
        layer.trainable = True
    #model.add(GlobalAveragePooling2D())
    model.add(Dense(num_classes, activation='softmax',name="Dense3"))
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
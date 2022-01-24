from tensorflow.keras.layers import Conv2D,MaxPool2D,BatchNormalization,Dropout,Flatten,Dense
from tensorflow.keras.models import Sequential

def create_CNN_simple(num_classes=150,input_shape=(150,150,3)):
    model_classic_CNN = Sequential(name='Classic_CNN')
    model_classic_CNN.add(Conv2D(128,3,input_shape=(input_shape),activation='relu'))
    model_classic_CNN.add(MaxPool2D())
    model_classic_CNN.add(Conv2D(128,3,activation='relu'))
    model_classic_CNN.add(MaxPool2D())
    model_classic_CNN.add(Conv2D(128,3,strides=(2,2),activation='relu'))
    model_classic_CNN.add(MaxPool2D())
    model_classic_CNN.add(BatchNormalization())
    model_classic_CNN.add(Conv2D(64,3,strides=(2,2),activation='relu'))
    model_classic_CNN.add(MaxPool2D())
    model_classic_CNN.add(Flatten())
    model_classic_CNN.add(Dropout(0.2))
    model_classic_CNN.add(Dense(1024,activation='relu'))
    model_classic_CNN.add(Dense(512,activation='relu'))
    model_classic_CNN.add(Dense(num_classes,activation='softmax'))
    model_classic_CNN.summary()
    model_classic_CNN.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model_classic_CNN
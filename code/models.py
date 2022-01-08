import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from keras.layers import BatchNormalization


def m1():
    model = Sequential([
    # data_augmentation,
    Conv2D(32, kernel_size=(3, 3),activation='linear',input_shape=(96,96,1),padding='same'),
    # BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D((2, 2),padding='same'),

    Conv2D(64, (3, 3), activation='linear',padding='same',kernel_regularizer=l2(0.01)),
    # BatchNormalization(),
    LeakyReLU(alpha=0.1),
    MaxPooling2D(pool_size=(2, 2),padding='same'),

    Conv2D(128, (3, 3), activation='linear',padding='same'),
    # BatchNormalization(),
    LeakyReLU(alpha=0.1),          
    MaxPooling2D(pool_size=(2, 2),padding='same'),

    Conv2D(256, (3, 3), activation='linear',padding='same',kernel_regularizer=l2(0.01)),
    # BatchNormalization(),
    LeakyReLU(alpha=0.1),                 
    MaxPooling2D(pool_size=(2, 2),padding='same'),
    Dropout(0.25),

    Flatten(),

    Dense(128, activation='linear'),
    # BatchNormalization(),
    LeakyReLU(alpha=0.1),  
    Dropout(0.25),

    Dense(num_classes, activation='softmax')])
    model.compile(optimizer =  Adam() , loss = "categorical_crossentropy", metrics=["accuracy"])
    return model

def m2(num_classes=11):
    model_master = Sequential()

    model_master.add(Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=(96,96,1), padding="same"))
    model_master.add(AveragePooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid'))
    model_master.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model_master.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model_master.add(Conv2D(120, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='valid'))
    model_master.add(Flatten())
    model_master.add(Dense(84, activation='relu'))
    model_master.add(Dense(num_classes, activation='softmax'))
    return model_master

def m3():
    model_master = Sequential()
    model_master.add(BatchNormalization())

    model_master.add(Conv2D(96, kernel_size=(3, 3), activation='relu', input_shape=(96,96,1), kernel_initializer='he_normal', padding="same"))
    model_master.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model_master.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))
    model_master.add(BatchNormalization())
    model_master.add(MaxPooling2D(pool_size=(3, 3), strides = 2))
	
    model_master.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model_master.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model_master.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model_master.add(BatchNormalization())
    model_master.add(MaxPooling2D(pool_size=(3, 3), strides = 2))

    model_master.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model_master.add(Conv2D(192, (1, 1), activation='relu'))
    model_master.add(Conv2D(11, (1, 1)))
    model_master.add(BatchNormalization())
    model_master.add(GlobalAveragePooling2D())

    model_master.add(Activation(activation='softmax'))
    return model_master

def m4():
    model = Sequential()
    
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding = 'same',input_shape=(96,96,1)))
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(96, (3, 3), activation='relu', padding = 'same', strides = 2))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same', strides = 2))
    model.add(Conv2D(192, (3, 3), activation='relu', padding = 'same'))
    model.add(Conv2D(192, (1, 1), activation='relu'))
    model.add(Conv2D(11, (1, 1)))
    model.add(GlobalAveragePooling2D())
    model.add(Activation(activation='softmax'))
    return model

def m5() :
    model = Sequential()

    #mlpconv block 1
    model.add(Conv2D(32, (5, 5), activation='relu',padding='valid', input_shape = (96,96,1)))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    #mlpconv block2
    model.add(Conv2D(64, (3, 3), activation='relu',padding='valid'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(Conv2D(64, (1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.5))
    
    #mlpconv block3
    model.add(Conv2D(128, (3, 3), activation='relu',padding='valid'))
    model.add(Conv2D(32, (1, 1), activation='relu'))
    model.add(Conv2D(11, (1, 1)))
    
    model.add(GlobalAveragePooling2D())
    model.add(Activation(activation='softmax'))    
    return model



# CNN
def create_cnn(num_classes):
    model = Sequential([
        
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96,96,1), padding='same'),
        LeakyReLU(alpha=0.1),
        MaxPooling2D((2, 2), padding='same'),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        LeakyReLU(alpha=0.1),
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        LeakyReLU(alpha=0.1),          
        MaxPooling2D(pool_size=(2, 2), padding='same'),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        LeakyReLU(alpha=0.1),                 
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Dropout(0.2),

        Flatten(),

        Dense(128, activation='relu'),
        LeakyReLU(alpha=0.1),  
        Dropout(0.2),

        Dense(num_classes, activation='softmax')
    ])
    return model

def simple_cnn(input_shape=(96, 96, 1), num_classes=11):
	
	# building a linear stack of layers with the sequential model
	model = Sequential()
	# convolutional layer
	model.add(Conv2D(25, kernel_size=(3,3), strides=(1,1), padding='valid', activation='relu', input_shape=input_shape))
	model.add(MaxPool2D(pool_size=(1,1)))
	# flatten output of conv
	model.add(Flatten())
	# hidden layer
	model.add(Dense(100, activation='relu'))
	# output layer
	model.add(Dense(num_classes, activation='softmax'))
	return model

def simple_cnn2(input_shape=(96, 96, 1), num_classes=11):
	
	# building a linear stack of layers with the sequential model
	model = Sequential()

	# convolutional layer
	model.add(Conv2D(50, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=input_shape))

	# convolutional layer
	model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(250, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
	model.add(MaxPool2D(pool_size=(2,2)))
	model.add(Dropout(0.25))

	# flatten output of conv
	model.add(Flatten())

	# hidden layer
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(250, activation='relu'))
	model.add(Dropout(0.3))
	# output layer
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

def resnet50(input_shape, num_classes=11):
    return tf.keras.applications.ResNet50V2(weights=None,
        input_shape=input_shape, pooling=None, classes=num_classes)
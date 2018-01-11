# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 20:17:55 2017

@author: JATIN
"""

# Import Statements
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten

#Initialising CNN
classifier = Sequential()

#Adding Convolution Layer 1

classifier.add(Convolution2D(32,3,3,input_shape=(64,64,3),activation='relu'))
#Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

#Adding Convolution Layer 2

classifier.add(Convolution2D(32,3,3,activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))

#Flatten
classifier.add(Flatten())

#Full Connection
classifier.add(Dense(output_dim=128,activation='relu'))
#classifier.add(Dense(output_dim=64,activation='relu'))
classifier.add(Dense(output_dim=1,activation='sigmoid'))

#compile
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the model to images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen= ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(64,64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                        target_size=(64,64),
                                                        batch_size=32,
                                                        class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000,
                        workers=4
                        )



# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 23:52:21 2022

@author: yehuh
"""

import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import pathlib

TRAIN_DATA_PATH = "./Images/waist"
TEST_DATA_PATH = "./Images/bra"


train_datagen = ImageDataGenerator( rescale =1./255, zoom_range = 0.2, horizontal_flip = True)
training_set = train_datagen.flow_from_directory(TRAIN_DATA_PATH,target_size = (64, 64),
batch_size = 32,class_mode = 'binary')

test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(TEST_DATA_PATH,
target_size = (64, 64),batch_size = 32, class_mode = 'binary')


cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=5, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='relu'))

cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x = training_set, validation_data = test_set, epochs = 5)


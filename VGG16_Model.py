# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 21:37:40 2022

@author: yehuh
"""
import keras

import tensorflow as tf
from keras.applications.vgg16 import VGG16
import os

model = VGG16(weights='imagenet', include_top=True)
model.summary()
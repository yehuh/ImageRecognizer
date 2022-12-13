# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 21:59:07 2022

@author: yehuh
"""
import cv2
import os
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
from keras.models import load_model
import keras


IMG_SIZE = 224

DATA_PATH = "./Images/"

def predict(filepath, model_k):
    img_arr = cv2.imread(filepath)[...,::-1] #convert BGR to RGB format
    resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)) # Reshaping images to preferred size
    img_arr = np.array(resized_arr)
    #predictions = model.predict_classes(img_arr)
    #predictions = predictions.reshape(1,-1)[0]
    img_arr_reshaped = img_arr.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return model_k.predict(img_arr_reshaped)

def expectLab(filepath, model):
    pred_rslts = predict(filepath, model)
    
    same_img_exist = False
    for preds in pred_rslts:
        for pred in preds:
            if(pred > 0.5):
                same_img_exist = True
                break;
                
        if(same_img_exist == True):
            break
    
    if(same_img_exist == False):
        return"Label Not Exist!!"
    
    
    labs = os.listdir(DATA_PATH)
    print(pred_rslts)
    expect_id = np.argmax(pred_rslts)
    return labs[expect_id]
    
    
    
model = keras.models.load_model('./Image.h5')

print(expectLab('breast_test0.jpg', model))
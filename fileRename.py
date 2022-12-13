# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:37:23 2022

@author: yehuh
"""

import os

# Absolute path of a file
data_path = "./Test/"


dirs = os.listdir(data_path)


for label in dirs:
    path = os.path.join(data_path, label)
    #print(path)
    imgs = os.listdir(path)
    for img in imgs:
        #print(img)
        if "火狐截图" in img:
            new_lab = img.replace("火狐截图","")
            #print(new_lab)
            new_path = os.path.join(path,new_lab)
            #print(new_path)
            old_path = os.path.join(path,img)
            #print(old_path)
            os.rename(old_path, new_path)
        #path = os.path.join(path, img)
        #imgs = os.listdir(path)
        '''
        for img in imgs:
            

'''
#old_name = r"E:\demos\files\reports\details.txt"
#new_name = r"E:\demos\files\reports\new_details.txt"

# Renaming the file
#os.rename(old_name, new_name)
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 22:37:23 2022

@author: yehuh
"""

import os

# Absolute path of a file
data_path = "./Images/waist/"


labels = os.listdir(data_path)

'''
for lab in labels:
    if "火狐截图" in lab:
        new_lab = lab.replace("火狐截图","")
        #print(new_lab)
        new_path = os.path.join(data_path,new_lab)
        old_path = os.path.join(data_path,lab)
        os.rename(old_path, new_path)
'''

#old_name = r"E:\demos\files\reports\details.txt"
#new_name = r"E:\demos\files\reports\new_details.txt"

# Renaming the file
#os.rename(old_name, new_name)
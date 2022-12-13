# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 20:34:41 2022

@author: yehuh
"""

import tensorflow as tf

#print(tf.__version__)

##查询tensorflow安装路径为:

#print(tf.__path__)


print(tf.__version__)


print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_gpu_available())# 或是版本比較低的tensorflow :
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
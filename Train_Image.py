"""Train MLP Model
Using MLP Model to Train Picture Recognition Model.
"""
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout

import numpy as np


# 設定 np 亂數種子
np.random.seed(10)

# 載入訓練資料集
n = 10000
img_feature = np.fromfile("./your/image/training/array.features", dtype=np.uint8)
img_feature = img_feature.reshape(n, 30, 30, 3)
img_label = np.fromfile("./your/image/training/array.labels", dtype=np.uint8)
img_label = img_label.reshape(n, 1)

# 打散資料集
indexs = np.random.permutation(img_label.shape[0])
rand_img_feature = img_feature[indexs]
rand_img_label = img_label[indexs]

# 資料正規化
# 將 feature 數字轉換為 0~1 的浮點數，能加快收斂，並提升預測準確度
# 把維度 (n,30,30,3) => (n, 30*30*3)後，再除255
img_feature_normalized = rand_img_feature.reshape(n, 30*30*3).astype('float32') / 255

# 將 label 轉換為 onehot 表示
img_label_onehot = np_utils.to_categorical(rand_img_label)

# 建立一個線性堆疊模型
model = Sequential()

# 建立輸入層與隱藏層
model.add(Dense(input_dim = 30*30*3, # 輸入層神經元數
                units = 1000, # 隱藏層神經元數
                kernel_initializer = 'normal', # 權重和誤差初始化方式:normal，使用常態分佈產生出始值
                activation = 'relu')) # 激活函數:relu函數，忽略掉負數的值

# 建立輸出層
model.add(Dense(units = 2, # 輸出層神經元數 (即[True, False])
                kernel_initializer = 'normal',
                activation = 'softmax')) # 激活函數:softmax函數，使輸出介於 0~1 之間

# 定義訓練方式
model.compile(loss='categorical_crossentropy', # 損失函數
             optimizer='adam', # 最佳化方法
             metrics=['accuracy']) # 評估方式:準確度

# 顯示模型摘要
print(model.summary())

# 開始訓練模型
train_history = model.fit(x=img_feature_normalized, # 指定 feature
                          y=img_label_onehot, # 指定 label 
                          validation_split=0.2, # 分80%訓練，20%驗證
                          epochs=5, # 執行 5 次訓練
                          batch_size=200, # 批次訓練，每批次 200 筆資料
                          verbose=2) # 顯示訓練過程

# 儲存模型
model.save("./your/image/training/models.dat")
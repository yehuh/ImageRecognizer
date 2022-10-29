import tensorflow as tf
import numpy as np
import os
from keras.utils import to_categorical
from tqdm import tqdm
import math
import keras 

DATA_PATH = "./Images/"

def load_image_features(label_name, img_paths=[]):
    # -----------------------------------------------------
    # Define tensorflow model
    # https://gist.github.com/eerwitt/518b0c9564e500b4b50f
    # -----------------------------------------------------
    # 定義 graph (tensor 和 flow)
    #.
    filename_queue = tf.train.string_input_producer(img_paths, shuffle=False)
    image_reader = tf.WholeFileReader()
    file_name, file_content = image_reader.read(filename_queue)
    decoded_image = tf.image.decode_jpg(file_content, channels=3)
    resized_image = tf.image.resize_images(decoded_image, [30, 30])

    final_image_ary = []
    # 執行 graph
    with tf.Session() as sess:
        # 執行tensorflow時要先初始化(初學者照抄即可!)
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord) # 建立多執行緒

        # image to RGB
        for i in range(len(img_paths)):
            img = sess.run(resized_image)
            final_image_ary.append(img)

        # 將最後的陣列轉換為 numpy 格式的陣列，以便存檔
        final_image_ary = np.array(final_image_ary, dtype=np.uint8)
        # 存檔
        final_image_ary.tofile("./"+label_name+".npy")

        # 停止多執行緒(初學者照抄前可!)
        coord.request_stop()
        coord.join(threads)

    return final_image_ary


def get_labels(path=DATA_PATH):
    labels = os.listdir(path)
    label_indices = np.arange(0, len(labels))
    return labels, label_indices, to_categorical(label_indices)

def save_data_to_array(path=DATA_PATH, max_len=11):
    labels, _, _ = get_labels(path)
    
    dirs = []
    for label in labels:
        dirs.append(path+label)
    
    
    for dir_ in dirs:
        label = dir_.replace("./Image","")
        img_files = os.listdir(dir_)
        img_files_path = []
        for img in img_files:
            img_files_path.append(dir_+"/"+img)
        
        load_image_features(label, img_files_path)






image_size = (180, 180)
batch_size = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_PATH,
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

train_dataset = train_dataset.prefetch(buffer_size=32)



class_names  = ['bra', 'breast', 'Iizuki', 'waist']

'''
#正規化將pixel從[0~255]變成[0~1]之間的範圍
def normalize(images, labels):
    images = tf.cast(images, tf.float32)    
    images /= 255    
    return images, labels

train_dataset =  train_dataset.map(normalize)

train_dataset =  train_dataset.cache()
'''


model = tf.keras.Sequential([    
    tf.keras.layers.Flatten(input_shape=(180, 180, 3)),  #輸入層    
    tf.keras.layers.Dense(128, activation=tf.nn.relu), #隱藏層    
    tf.keras.layers.Dense(10)                          #輸出層
    ])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])


num_train_examples = 6767

model.fit(train_dataset, epochs=5, steps_per_epoch=math.ceil(num_train_examples/batch_size))

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 16:33:46 2022

@author: yehuh
"""

import numpy as np 
import tensorflow as tf 
from tensorflow import keras
from matplotlib import pyplot as plt

ImageSize = 224

from tf_keras_vis.utils.model_modifiers import ReplaceToLinear

replace2linear = ReplaceToLinear()

# Instead of using ReplaceToLinear instance,
# you can also define the function from scratch as follows:
def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear

from tf_keras_vis.utils.scores import CategoricalScore



# Instead of using CategoricalScore object above,
# you can also define the function from scratch as follows:
def score_function(output):
    # The `output` variable refer to the output of the model,
    # so, in this case, `output` shape is `(1, 1000)` i.e., (samples, classes).
    return output[:, 20]

from tf_keras_vis.activation_maximization import ActivationMaximization


model = keras.models.load_model('./Image.h5')
activation_maximization = ActivationMaximization(model,
                                                 model_modifier=replace2linear,
                                                 clone=True)




from tf_keras_vis.activation_maximization.callbacks import Progress
from tf_keras_vis.activation_maximization.input_modifiers import Jitter, Rotate2D, Scale
from tf_keras_vis.activation_maximization.regularizers import Norm, TotalVariation2D


# 20 is the imagenet index corresponding to Ouzel.
score = CategoricalScore(20)

# Generate maximized activation
activations = activation_maximization(score,
                                      callbacks=[Progress()])

## Since v0.6.0, calling `astype()` is NOT necessary.
# activations = activations[0].astype(np.uint8)

# Render
f, ax = plt.subplots(figsize=(4, 4))
ax.imshow(activations[0])
ax.set_title('Ouzel', fontsize=16)
ax.axis('off')
plt.tight_layout()
plt.show()



'''
def compute_loss(input_image, filter_index):
     # We avoid border artifacts by only involving non-border pixels in the loss.
     activation = feature_extractor(input_image)     
     filter_activation = activation[:, 2:-2, 2:-2, filter_index]
     return tf.reduce_mean(filter_activation)


def gradient_ascent_step(img, filter_index, learning_rate):
     with tf.GradientTape() as tape:
         tape.watch(img)
         loss = compute_loss(img, filter_index)
         # Compute gradients.
         grads = tape.gradient(loss, img)
         # Normalize gradients.
         grads = tf.math.l2_normalize(grads)
         img += learning_rate * grads
         return loss, img


def initialize_image(image_size=224):     # We start from a gray image with some random noise
     img = tf.random.uniform((1, image_size, image_size, 3))     # ResNet50V2 expects inputs in the range [-1, +1].
     # Here we scale our random inputs to [-0.125, +0.125]
     return (img - 0.5) * 0.25


def deprocess_image(img):     # Normalize array: center on 0., ensure variance is 0.15
     img -= img.mean()     
     img /= img.std() + 1e-5     
     img *= 0.15     # Center crop     
     img = img[25:-25, 25:-25, :]     # Clip to [0, 1]
     img += 0.5
     img = np.clip(img, 0, 1)     # Convert to RGB array
     img *= 255
     img = np.clip(img, 0, 255).astype("uint8")
     return img

def visualize_filter(filter_index):     # We run gradient ascent for 20 steps
     iterations = 30
     learning_rate = 10.0
     img = initialize_image()
     for iteration in range(iterations):
         loss, img = gradient_ascent_step(img, filter_index, learning_rate)     # Decode the resulting input image
         
     img = deprocess_image(img[0].numpy())     
     return loss, img

from IPython.display import Image, display
model = keras.models.load_model('./Image.h5')

model.summary()

layer = model.get_layer("dense_1")#"conv2d",max_pooling2d
feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

loss, img = visualize_filter(0)
keras.preprocessing.image.save_img("0.png", img)
display(Image("0.png"))



all_imgs = [] 
for filter_index in range(32):
     print("Processing filter %d" % (filter_index,))
     loss, img = visualize_filter(filter_index)
     all_imgs.append(img)

margin = 5 
colum_cnt = 4;
row_cnt = 8;
n = 5 
cropped_width = ImageSize - 25 * 2 
cropped_height = ImageSize - 25 * 2 
width = colum_cnt * cropped_width + (colum_cnt - 1) * margin 
height = row_cnt * cropped_height + (row_cnt - 1) * margin 
stitched_filters = np.zeros((width, height, 3))

for i in range(colum_cnt):
     for j in range(row_cnt):
         img = all_imgs[i * colum_cnt + j]
         stitched_filters[
             (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
             (cropped_height + margin) * j : (cropped_height + margin) * j
             + cropped_height,
             :,         
         ] = img

keras.preprocessing.image.save_img("stiched_filters.png", stitched_filters)
display(Image("stiched_filters.png"))
'''
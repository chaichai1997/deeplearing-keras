# -*- coding: utf-8 -*-
# author = "chaichai"
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from keras import models

from keras.models import  load_model
model = load_model('cat_dog_small_2.h5')
img_path = 'F:/deep learning/code/kreastest/cat-data/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
# img_tensor.shape (1, 150, 150, 3)
layer_output = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_output)
activations = activation_model.predict(img_tensor)
frist_layer_activation = activations[0]
plt.matshow(frist_layer_activation[0, :, :, 4], cmap='viridis')
plt.show()



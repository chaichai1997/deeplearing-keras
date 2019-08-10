# -*- coding: utf-8 -*-
# author = "chaichai"
from keras import layers
import numpy as np
x = np.random.randint(128, 8, 8, 3)

y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
#  y = layers.add([y, x])
residual = layers.Conv2D(128, 1, activation='relu', padding='same')(x)
y = layers.add([y, residual])

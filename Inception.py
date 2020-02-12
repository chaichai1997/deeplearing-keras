# -*- coding: utf-8 -*-
# author = "chaichai"
from keras import layers
import numpy as np
from keras.applications import inception_v3

x = np.random.randint(128, 3, 3, 3)

branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3,activation='relu', strides=2)(branch_b)
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)

branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)

out_put = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)
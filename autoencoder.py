# -*- coding: utf-8 -*-
# auther = "chaichai"

import keras
from keras import layers
from keras.datasets import mnist
from keras.models import Sequential
import numpy as np


# Data preprocess
(X_train, _), (X_test, _) = mnist.load_data(path="E:\\.keras\\datasets\\mnist.npz")
x_train = X_train.astype('float32')/255.0
x_test = X_test.astype('float32')/255.0

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

# Data Augment (Add noise) :make auto-encoder be more robust  Gaussian Distribution(0,1)
x_train_noise = x_train + 0.3 * np.random.normal(loc=0., scale=1., size=x_train.shape)
x_test_noise = x_test + 0.3 * np.random.normal(loc=0., scale=1., size=x_test.shape)
x_train_noise = np.clip(x_train_noise, 0., 1)
x_test_noise = np.clip(x_test_noise, 0., 1)

# building model
model = Sequential()
model.add(layers.Dense(500, activation='relu', input_shape=(28*28, )))
model.add(layers.Dense(784, activation='sigmoid'))

# compile
model.compile(
    optimizer='adam',
    loss='binary_crossentropy'
)
model.fit(
    x_train_noise, x_train,
    epochs=20,
    batch_size=123,
    verbose=1,
    validation_data=(x_test, x_test)
)

# result show
import matplotlib.pyplot as plt

decoder_img = model.predict(x_test_noise)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(3, n, i+1)
    plt.imshow(x_test_noise[i].reshape(28, 28))
    plt.gray()
    ax = plt.subplot(3, n, i+1+n)
    plt.imshow(decoder_img[i].reshape(28, 28))
    plt.gray()
    ax = plt.subplot(3, n, i+1+2*n)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()






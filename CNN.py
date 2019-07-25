# -*- coding: utf-8 -*-
"""
实例化一个小型CNN网络(Le-Net 经典卷积网络五层模型)
卷积神经网络接收的张量为(image_height, image.width, image_channels)
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320      (3*3+1)*32
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496    3*3*32*64+64
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 64)          36928    3*3*64*64+64
flatten_1 (Flatten)          (None, 576)               0
_________________________________________________________________
dense_1 (Dense)              (None, 64)                36928    576*64+64
_________________________________________________________________
dense_2 (Dense)              (None, 10)                650      64*10+10
=================================================================
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
"""
from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical


"""
    模型搭建
"""
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# 输出展平
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

"""
    在mnist数据集上训练模型
"""
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255  # 数字归一化处理
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_images, train_labels, epochs=6, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("test_acc=", test_acc)
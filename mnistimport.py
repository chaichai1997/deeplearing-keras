# -*- coding: utf-8 -*-

from keras.datasets import mnist
import  numpy as np
(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path="E:/deep learning/code/kreastest/mnist.npz")
# print(train_images.shape)
# print(len(train_labels))
# print(test_labels)
from keras import models
from keras import layers

"""
定义网络架构   
layer为神经网络的核心组件，是一种数据处理模块
"""
network = models.Sequential()
# dense，密集连接（全连接）
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

"""
编译, 损失函数、优化器、训练测试过程中监控的指标
"""

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

"""
准备图像数据
"""
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255  # 数字归一化处理
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

"""
准备标签
    对标签进行分类编码
"""
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

"""
训练网络
"""
network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("test_acc=", test_acc)
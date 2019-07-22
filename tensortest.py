# -*- coding: utf-8 -*-
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt
"""
 0D张量：标量
"""
x = np.array(12)
print(x, x.ndim)

"""
    1D张量：向量
"""
x = np.array([11, 12, 13, 11])
print(x, x.ndim)

"""
    2D张量：矩阵
"""
x = np.array([
    [1, 2, 3],
    [1, 2, 4]]
)
print(x, x.ndim)

(train_images, train_labels), (test_images, test_labels) = \
    mnist.load_data(path="E:/deep learning/code/kreastest/mnist.npz")
print(train_images.ndim, train_images.shape, train_images.dtype)

"""
    显示第4个数字
"""
# digit = train_images[4]
# plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()
my_slice = train_images[10:100]
print(my_slice.shape)
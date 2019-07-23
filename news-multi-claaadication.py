# -*- coding: utf-8 -*-

from keras.datasets import reuters
import numpy as np
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
"""
多分类问题：
    单标签、多分类,输出为46，则中间层的隐藏单元个数不应该比46小太多
"""
"""
    加载数据集,并进行编码
        将数据限定为前1000个单词
        训练样本8982
        测试样本2246
        输出类别46
"""

(train_data, train_labels), (test_data, test_labels) = \
    reuters.load_data('E:/deep learning/code/kreastest/reuters.npz', num_words=10000)


# 训练数据标签化
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(x_train.shape)


# 标签向量化
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1
    return results


one_hot_train_label = to_one_hot(train_labels)  # one_hot_train_label = to_categorical(train_labels)
one_hot_test_label = to_one_hot(test_labels)   # one_hot_test_label = to_categorical(test_labels)


# 验证集分割
x_val = x_train[:1000]
part_x_train = x_train[1000:]
y_val = one_hot_train_label[:1000]
part_y_train = one_hot_train_label[1000:]


"""
    构建网络
"""
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))


"""
    模型编译与运行
"""
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    part_x_train,
    part_y_train,
    epochs=20,
    batch_size=512,
    validation_data=(x_val, y_val)
)


"""
    绘制训练损失与验证损失
"""
loss_value = history.history['loss']
val_loss_value = history.history['val_loss']
epoch = range(1, len(loss_value) + 1)
plt.plot(epoch, loss_value, 'bo', label='Training loss')  # bo,蓝色远点
plt.plot(epoch, val_loss_value, 'b', label='Validation loss')  # 蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

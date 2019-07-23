# -*- coding: utf-8 -*-
from keras.datasets import imdb

"""
    电影影评分类，根据电影影评的文字将其分为正面与负面
    数据集:imdb,数据集为单词索引组成的列表，将其转换为矩阵并进行归一化
    网络模型：16全连接+16全连接+1 sigmoid
    数据集规模：训练15000*1000；交叉验证10000*1000；测试集250000
"""
"""
    加载imdb数据集，仅保留前1000个最常出现的单词
    path: c:/Users/chaichai/.keras/datasets
"""
(train_data, train_labels), (test_data, test_labels) = \
    imdb.load_data(num_words=10000)

# print(train_data)
# print(train_labels[0])
# print(max([max(sequence) for sequence in train_data]))

"""
    准备数据,将整数序列编码为二进制矩阵
"""

import numpy as np

# one-hot编码


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # results[i]的指定索引为1 [3,5]->10000维，只有3，5的元素是1
    return results


x_train = vectorize_sequences(train_data)  # shape 2500*1000
x_test = vectorize_sequences(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


"""
    构建网络
"""
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


# """
#     编译模型
# """
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#

# """
#     优化配置器
# """
# from keras import optimizers
# model.compile(optimizer=optimizers.RMSprop(lr=0.001),
#               loss='binary_crossentropy',
#               metrics=['accuracy'])


"""
    构造验证集
"""
x_val = x_train[:10000]
part_x_train = x_train[10000:]
y_val = y_train[:10000]
part_y_train = y_train[10000:]


"""
    训练模型
"""
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(part_x_train,
                    part_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=[x_val, y_val]
                    )
history_dict = history.history
print(history_dict.keys())


"""
    绘制训练损失和验证损失    
"""
import matplotlib.pyplot as plt

loss_value = history_dict['loss']
val_loss_value = history_dict['val_loss']
epoch = range(1, len(loss_value) + 1)
plt.plot(epoch, loss_value, 'bo', label='Training loss')  # bo,蓝色远点
plt.plot(epoch, val_loss_value, 'b', label='Validation loss')  # 蓝色实线
plt.title('Training and validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


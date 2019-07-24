# -*- coding: utf-8 -*-
from keras import models
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
"""
    回归问题：
    数据集：波士顿房价数据集404+102，且输入数据的特征取值范围不同
"""
from keras.datasets import boston_housing
(train_data, train_labels), (test_data, test_labels) = \
    boston_housing.load_data()
print(train_data.shape)


"""
准备数据，对每个数据特征进行标准化，  （x-特征平均值）/特征标准差
"""
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std


"""
    最后一层为线性层，没有激活函数，这是标量回归的典型配置
    mse：均方误差，回归问题常用损失函数
    mae: 平均绝对误差，表示预测值与目标值之间误差的绝对值
"""


def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(
        optimizer='rmsprop',
        loss='mse',
        metrics=['mae']
    )
    return model


"""
    k折交叉验证,verbose=0 训练为静默模式
"""
k = 4
num_val_sample = len(train_data) // k
num_epochs = 100
all_score = []


for i in range(k):
    print("process" + str(i) + 'fold')
    # 验证集数据，第k个分区
    val_data = train_data[i*num_val_sample:(i+1)*num_val_sample]
    val_target = train_labels[i*num_val_sample:(i+1)*num_val_sample]
    part_train_data = np.concatenate([
        train_data[:i*num_val_sample],
        train_data[(i+1)*num_val_sample:]],
        axis=0
    )
    part_train_label = np.concatenate([
        train_labels[:i*num_val_sample],
        train_labels[(i+1)*num_val_sample:]],
        axis=0
    )
    model = build_model()
    history = model.fit(
        part_train_data,
        part_train_label,
        validation_data=[val_data, val_target],
        epochs=num_epochs,
        batch_size=1,
        verbose=0
    )

    mae_history = history.history['val_mean_absolute_error']
    all_score.append(mae_history)


"""
    计算平均值
"""
average_mae_history = [np.mean([x[i] for x in all_score]) for i in range(num_epochs)]
print(average_mae_history)


"""
    绘制验证分数
"""
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('epochs')
plt.ylabel('mae')
plt.show()


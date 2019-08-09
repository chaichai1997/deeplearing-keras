# -*- coding: utf-8 -*-
# author = "chaichai"
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
"""
温度预测问题:
    一个时间步为10分钟，每个steps(steps = 6)个时间步采一次数据，给定过去lookback(720, 5天)内的
    观测数据，预测delay(144,24小时)后的数据
"""
# 数据集
data_dir = 'F:/deep learning/code/data'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]  # len(lines) 420551行数据
# 解析数据,将420551行数据转换为一个数组
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
# 绘制温度时间的序列
temp = float_data[:, 1]  # 温度
# 数据标准化,每个时间序列减去其平均值，然后除以标准差
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

"""
成序列样本及目标的生成器
shuffle:打乱样本与否
min_index, max_index:data数组的索引，用于界定需要抽取哪些时间步。
"""


def generator(data, lookback, delay, min_inndex, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data)-delay-1
    i = min_inndex + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_inndex + lookback, max_index, size=batch_size
            )
        else:
            if i + batch_size >= max_index:
                i = min_inndex + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((
            len(rows),
            lookback // step,
            data.shape[-1]
        ))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

# 准备训练生成器、验证生成器、和测试生成器
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_inndex=0,
    max_index=200000,
    shuffle=True,
    step=step,
    batch_size=batch_size
)
val_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_inndex=200001,
    max_index=300000,
    step=step,
    batch_size=batch_size
)
test_gen = generator(
    float_data,
    lookback=lookback,
    delay=delay,
    min_inndex=300001,
    max_index=None,
    step=step,
    batch_size=batch_size
)
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size  # 抽取次数


# 评估的循环代码
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append((mae))
    return batch_maes


# # 训练并评估一个密集连接模型
# model = Sequential()
# model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(
#     train_gen,
#     steps_per_epoch=500,
#     epochs=20,
#     validation_data=val_gen,
#     validation_steps=val_steps
#     )


# # 循环网络
# model = Sequential()
# model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(
#     train_gen,
#     steps_per_epoch=500,
#     epochs=20,
#     validation_data=val_gen,
#     validation_steps=val_steps
# )


# 使用dropout正则化基于GRU的模型
# model = Sequential()
# model.add(layers.GRU(
#     32,
#     dropout=0.2,
#     recurrent_dropout=0.2,
#     input_shape=(None, float_data.shape[-1])))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(
#     train_gen,
#     steps_per_epoch=500,
#     epochs=40,
#     validation_data=val_gen,
#     validation_steps=val_steps
# )

# 循环层堆叠
# model = Sequential()
# model.add(layers.GRU(
#     32,
#     dropout=0.1,
#     recurrent_dropout=0.5,
#     return_sequences=True,
#     input_shape=(None, float_data.shape[-1])
# ))
# model.add(layers.GRU(
#     64,
#     activation='relu',
#     dropout=0.1,
#     recurrent_dropout=0.5
# ))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(
#     train_gen,
#     steps_per_epoch=500,
#     epochs=40,
#     validation_data=val_gen,
#     validation_steps=val_steps
# )


# 双向GRU
# model = Sequential()
# model.add(layers.Bidirectional(
#     layers.GRU(32), input_shape=(None, float_data.shape[-1])
# ))
# model.add(layers.Dense(1))
# model.compile(optimizer=RMSprop(), loss='mae')
# history = model.fit_generator(
#     train_gen,
#     steps_per_epoch=500,
#     epochs=40,
#     validation_data=val_gen,
#     validation_steps=val_steps
# )


# CNN
model = Sequential()
model.add(layers.Conv1D(32, 5, activation='relu',
                        input_shape=(None, float_data.shape[-1])))
model.add(layers.MaxPooling1D(3))
model.add(layers.Conv1D(32, 5, activation='relu'))
model.add(layers.MaxPooling1D(3))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))
model.compile(
    optimizer=RMSprop(),
    loss='mae'
)
history = model.fit_generator(
    train_gen,
    steps_per_epoch=500,
    epochs=20,
    validation_data=val_gen,
    validation_steps=val_steps
)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()



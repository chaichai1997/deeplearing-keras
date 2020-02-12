# -*- coding: utf-8 -*-
# author = "chaichai"
"""
实现共享层权重，每次调用使用相同的权重，
可构建具有共享分支的模型
"""
from keras import layers
from keras import Input
from keras.models import Model

lstm = layers.LSTM(32)  # LSTM实例化一次

left_input = Input(shape=(None, 128))  # 输入是长度为128的向量组成的变长序列
left_output = lstm(left_input)

right_input = Input(shape=(None, 128))
right_output = lstm(right_input)

merged = layers.concatenate([left_output, right_output], axis=-1)
prediction = layers.Dense(1, activation='sigmoid')(merged)

model = Model([left_input, right_input], prediction)  # 将模型实例化
# model.fit([left_data, right_data], target)

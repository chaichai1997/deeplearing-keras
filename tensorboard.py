# -*- coding: utf-8 -*-
# author = "chaichai"
import keras
from keras.datasets import imdb
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
max_features = 2000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)

# 模型搭建
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len, name='embed'))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1))

# 模型编译与运行
model.compile(
    optimizer=RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)
callbacks = [
    keras.callbacks.TensorBoard(
        log_dir='tensorlog',  # 日志文件保存路径
        histogram_freq=1,   # 每一轮之后记录激活直方图
        embeddings_freq=1   # 每一轮之后记录嵌入数据
    )
]
history = model.fit(
    x_train,
    y_train,
    epochs=20,
    batch_size=128,
    validation_split=0.2,
    callbacks=callbacks
)
# # 创建一张PNG图像
# from keras.utils import plot_model
# plot_model(model, to_file='model.PNG')

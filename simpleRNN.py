# -*- coding: utf-8 -*-
# author = "chaichai"
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense, LSTM
from keras.datasets import imdb
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
"""
keras中的循环层
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, None, 32)          320000
_________________________________________________________________
simple_rnn_1 (SimpleRNN)     (None, 32)                2080  32*（32+32+1）
=================================================================
Total params: 322,080
Trainable params: 322,080
Non-trainable params: 0
_________________________________________________________________
None
"""

# model = Sequential()
# model.add(Embedding(10000, 32))
# model.add(SimpleRNN(32))
# print(model.summary())
# 准备IMDB数据
max_feature = 10000
maxlen = 500
batch_size = 32
(input_train, y_train), (input_test, y_test) = imdb.load_data(
    num_words=max_feature)
# print(len(input_train), 'train sequence')
# print(len(input_test), 'test sequence')
# print('Pad sequence(samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
# print('input_train shape:', input_train.shape)
# print('input_test shape:', input_test.shape)

# # 用Embedding和simpleRNN来训练模型
# model = Sequential()
# model.add(Embedding(max_feature, 32))
# model.add(SimpleRNN(32))
# model.add(Dense(1, activation='sigmoid'))
#
# model.compile(
#     optimizer='rmsprop',
#     loss='binary_crossentropy',
#     metrics=['acc']
# )
# history = model.fit(
#     input_train,
#     y_train,
#     epochs=10,
#     batch_size=128,
#     validation_split=0.2
# )

# keras中的LSTM
model = Sequential()
model.add(Embedding(max_feature, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(
    optimizer='rmsprop',
    loss = 'binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    input_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)
# 绘制结果
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

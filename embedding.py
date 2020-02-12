# -*- coding: utf-8 -*-
# author = "chaichai"
from keras.layers import Embedding, Flatten, Dense
from keras.models import Sequential
from keras.datasets import imdb
from keras import preprocessing
# Embedding层将整数索引(表示特定单词)映射为密集向量
embedding_layer = Embedding(1000, 64)  # 标记的最大个数1000， 嵌入的维度64
max_feature = 10000
max_len = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feature)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)   # 将列表转换为(samples, maxlen)的二位整数张量
model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
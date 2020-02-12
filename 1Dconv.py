# -*- coding: utf-8 -*-
# author = "chaichai"
from keras.datasets import imdb
from keras import layers
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
import matplotlib.pyplot as plt
max_features = 10000
max_len = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train, maxlen=max_len)  # (2500,500)
x_test = sequence.pad_sequences(x_test, maxlen=max_len)  # (2500,500)

# 模型搭建
model = Sequential()
model.add(layers.Embedding(max_features, 128, input_length=max_len))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalAveragePooling1D())
model.add(layers.Dense(1))

# 模型编译与运行
model.compile(
    optimizer=RMSprop(lr=1e-4),
    loss='binary_crossentropy',
    metrics=['acc']
)
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
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

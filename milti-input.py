# -*- coding: utf-8 -*-
# author = "chaichai"
from keras import layers
from keras.models import Model
from keras import Input, utils
import numpy as np
import matplotlib.pyplot as plt

text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500

# 函数式API
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)
question_input = Input(shape=(None,), dtype='int32', name='question')
embeded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(32)(embeded_question)
# 连接编码后的问题和文本
contact = layers.concatenate([encoded_text, encoded_text], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(contact)
# 模型实例化
model = Model([text_input, question_input], answer)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['acc']
)

num_samples = 1000
max_length = 100

text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
answer = np.random.randint(answer_vocabulary_size, size=(num_samples))
answer = utils.to_categorical(answer, answer_vocabulary_size)

history = model.fit([text, question], answer, epochs=10, batch_size=128, validation_split=0.2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss)+1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

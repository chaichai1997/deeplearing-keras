# -*- coding: utf-8 -*-
# author = "chaichai"
from keras import layers
from keras import Input
from keras.models import Model
import matplotlib.pyplot as plt
vocabulary_size = 50000
num_income_groups = 10


post_input = Input(shape=(None,), dtype='int32', name='posts')
embeded_input = layers.Embedding(256, vocabulary_size)(post_input)
x = layers.Conv1D(128, 5, activation='relu')(embeded_input)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

# 多输出，输出层皆具有名称
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups, activation='softmax', name='income')(x)
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
model = Model(post_input, [age_prediction, income_prediction, gender_prediction])


# 多输入的编译
model.compile(
    optimizer='rmsprop',
    loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
    # loss = {
    #     'age':'mse',
    #     'income': 'categorical_crossentropy',
    #     'gender': 'binary_crossentropy'
    # }
    loss_weights=[0.25, 1., 10.]  # 损失加权
    # loss_weights = {
    #     'age':0.25,
    #     'income': 1.,
    #     'gender': 10.
    # }
)

# model.fit(post_input, [age_targets, income_targets, gender_targets], epochs=10, batch_size=64)
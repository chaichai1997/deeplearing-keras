# -*- coding: utf-8 -*-
# author = "chaichai"
import numpy as np
from keras.preprocessing.text import Tokenizer
"""
单词级别的one-hot编码
result
[[[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]

 [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]]

"""
samples = ['The car sat on the mat.', 'The dog ate my homework.']  # 初始数据
token_index = {}  # 构建数据集中所有标记的索引
for i in samples:
    for w in i.split():
        if w not in token_index:
            token_index[w] = len(token_index) + 1  # 为每个单词指定唯一的索引
max_length = 10  # 只考虑每个样本前10个max_length的值
results = np.zeros(shape=(    # 将结果保存至result中
    len(samples),
    max_length,
    max(token_index.values())+1
))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1
"""
使用keras实现ont-hot
"""
tokenizer = Tokenizer(num_words=1000)  # 创建一个分词器，只考虑前1000个最常见的单词
tokenizer.fit_on_texts(samples)    # 构建单词索引
sequences = tokenizer.texts_to_sequences(samples)  # 将字符串转换为整数索引组成的列表
ont_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
word_index = tokenizer.word_index   # 找回单词索引
print('Found %s unique tokens.' % len(word_index))


# -*- coding: utf-8 -*-
# author = "chaichai"

import keras
import numpy as np
from keras import layers
import random
import sys


# 下载并解析初始文本文件，将预料转换为小写
path = keras.utils.get_file('nietzsche.txt',
                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path).read().lower()
print('length:', len(text))


# 将字符序列向量化
maxlen = 60         # 提取60个字符组成的序列
step = 3            # 每三个字符采样一个新序列
sentences = []      # 保留所提取的序列
next_char = []      # 保存目标，及下一个字符
for i in range(0, len(text) - maxlen, step):   # 每隔三个字符采样一个长度为60的序列
    sentences.append(text[i: i + maxlen])
    next_char.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

chars = sorted(list(set(text)))  # 语料中唯一字符组成的列表
print('Number of sequence:', len(chars))
char_indices = dict((char, chars.index(char)) for char in chars)   # 将字符映射为它在列表chars中的索引

print("Vectorization...")   # 将字符one-hot编码为二进制数组
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_char[i]]] = 1


# 构建网络
model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

# 模型编译
optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# 给定模型预测，采样下一个字符,对模型得到的原始概率分布进行重新加权，并抽取一个字符索引
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)   # 概率分布重新加权
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 文本生成循环
for epoch in range(1, 60):
    print('opoch:', epoch)
    model.fit(x, y, batch_size=128, epochs=1)   # 将模型在数据上拟合一次
    start_index = random.randint(0, len(text)-maxlen-1)
    generate_text = text[start_index:start_index+maxlen]
    print('--Generate seed: ',  generate_text)    # 随机选择一个文本种子
    for t in [0.2, 0.5, 1.0, 1.2]:
        print('temperature:', t)
        sys.stdout.write(generate_text)

        for i in range(400):     # 从种子文件开始，生成400字符
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generate_text):   # 对生成的字符进行one-hot编码
                sampled[0, t, char_indices[char]] = 1
            preds = model.predict(sampled, verbose=0)[0]  # 对下一个字符进行采样
            next_index = sample(preds, t)
            next_char = chars[next_index]

            generate_text += next_char
            generate_text = generate_text[1:]

            sys.stdout.write(next_char)




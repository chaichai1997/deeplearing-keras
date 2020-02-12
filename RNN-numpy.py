# -*- coding: utf-8 -*-
# author = "chaichai"
import numpy as np
timesteps = 100   # 序列时间步
input_feature = 32   # 输出特征的维度
output_feature = 64   # 输入特征的维度
inputs = np.random.random((timesteps, input_feature))  # 输入数据
state_t = np.zeros((output_feature, ))
w = np.random.random((output_feature, input_feature))
u = np.random.random((output_feature, output_feature))
b = np.random.random((output_feature, ))
success_outputs = []
for i in inputs:
    output_t = np.tanh(np.dot(w, i) + np.dot(u, state_t) + b)
    success_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(success_outputs, axis=0)  # 最终的输出(timesteps, output_features)
# -*- coding: utf-8 -*-
# author = "chaichai"
import keras

callbacks_list = [
    keras.callbacks.EarlyStopping(  # EarlyStopping:如果不在改善，则中断训练
        monitor='acc',   # monitor：监控模型
        patience=1    # patience：精度在多于一轮的1时间内不再改善，则中断训练
    ),

    keras.callbacks.ModelCheckpoint(  # ModelCheckpoint，在每轮过后保存当前模型权重
        filepath='XX.h5',   # 模型保存路径
        monitor='val_loss',  # 监控模型
        save_best_only=True   # 如果val_loss没有改善，则不覆盖模型文件，即保存训练中的最佳模型
    ),
    keras.callbacks.ReduceLROnPlateau(  # 损失平台，模型不改善时降低学习率，跳出局部最小值
        monitor='val_loss',
        factor=0.1,  # 触发时，将学习率/10
        patience=10  # val_loss十轮内没有改善，则触发此回调函数
    )
]
# model.fit(    # fit时传入callbacks_list
#     x,
#     y,
#     epoch=10,
#     batch_size=32,
#     callbacks=callbacks_list,
#     validation_data=(x_val, y_val)
# )
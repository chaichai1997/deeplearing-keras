# -*- coding: utf-8 -*-
import os, shutil
"""
将图像复制到训练、验证、测试目录下
"""
original_dataset_dir = 'E:\\deep learning\\code\\kreastest\\dogs-vs-cats\\train'
base_dir = 'E:\\deep learning\\code\\kreastest\\cat-data'
os.mkdir(base_dir)

train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)

# 猫的训练路径/验证路径/测试路径
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)
test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)

# 狗的训练路径/验证路径/测试路径
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)

# 数据生成
# 1000->train_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for i in fnames:
    src = os.path.join(original_dataset_dir, i)
    dst = os.path.join(train_cats_dir, i)
    shutil.copyfile(src, dst)

# 500->validation_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for i in fnames:
    src = os.path.join(original_dataset_dir, i)
    dst = os.path.join(validation_cats_dir, i)
    shutil.copyfile(src, dst)

# 500->test_cats_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for i in fnames:
    src = os.path.join(original_dataset_dir, i)
    dst = os.path.join(test_cats_dir, i)
    shutil.copyfile(src, dst)

# 1000->train_dog_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for i in fnames:
    src = os.path.join(original_dataset_dir, i)
    dst = os.path.join(train_dogs_dir, i)
    shutil.copyfile(src, dst)

# 500->validation_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for i in fnames:
    src = os.path.join(original_dataset_dir, i)
    dst = os.path.join(validation_dogs_dir, i)
    shutil.copyfile(src, dst)

# 500->test_cats_dir
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for i in fnames:
    src = os.path.join(original_dataset_dir, i)
    dst = os.path.join(test_dogs_dir, i)
    shutil.copyfile(src, dst)

print('total training cat images', len(os.listdir(train_cats_dir)))
print('total validation cat images', len(os.listdir(validation_cats_dir)))
print('total test cat images', len(os.listdir(test_cats_dir)))
print('total training cat images', len(os.listdir(train_dogs_dir)))
print('total validation cat images', len(os.listdir(validation_dogs_dir)))
print('total test cat images', len(os.listdir(test_dogs_dir)))

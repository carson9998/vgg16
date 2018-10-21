# -*- coding:utf-8 -*-
#读取olivettifaces.pkl文件，分为训练集（40*8个样本），验证集（40*1个样本），测试集（40*1个样本）
import cPickle
import numpy
read_file=open('/home/carson/one/olivettifaces.pkl','rb')  
faces=cPickle.load(read_file)
label=cPickle.load(read_file)  
read_file.close() 

train_data=numpy.empty((320,2679))
train_label=numpy.empty(320)
valid_data=numpy.empty((40,2679))
valid_label=numpy.empty(40)
test_data=numpy.empty((40,2679))
test_label=numpy.empty(40)

for i in range(40):
    train_data[i*8:i*8+8]=faces[i*10:i*10+8]
    train_label[i*8:i*8+8]=label[i*10:i*10+8]
    valid_data[i]=faces[i*10+8]
    valid_label[i]=label[i*10+8]
    test_data[i]=faces[i*10+9]
    test_label[i]=label[i*10+9]
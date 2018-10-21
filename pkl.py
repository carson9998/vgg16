# -*- coding:utf-8 -*-
#读取人脸库olivettifaces，并存储为pkl文件
import numpy
from PIL import Image
import cPickle

#读取原始图片并转化为numpy.ndarray，将灰度值由0～256转换到0～1
img = Image.open('/home/carson/one/olivettifaces.gif')
img_ndarray = numpy.asarray(img, dtype='float64')/256


#图片大小时1190*942，一共20*20个人脸图，故每张人脸图大小为（1190/20）*（942/20）即57*47=2679
#将全部400个样本存储为一个400*2679的数组，每一行即代表一个人脸图，并且第0～9、10～19、20～29...行分别属于同个人脸
#另外，用olivettifaces_label表示每一个样本的类别，它是400维的向量，有0～39共40类，代表40个不同的人。
olivettifaces=numpy.empty((400,2679))
for row in range(20):
    for column in range(20):
        olivettifaces[row*20+column]=numpy.ndarray.flatten(img_ndarray [row*57:(row+1)*57,column*47:(column+1)*47])

#建一个<span style="font-family: SimSun;">olivettifaces_label</span>
olivettifaces_label=numpy.empty(400)
for label in range(40):
    olivettifaces_label[label*10:label*10+10]=label
olivettifaces_label=olivettifaces_label.astype(numpy.int)

#保存olivettifaces以及olivettifaces_label到olivettifaces.pkl文件
write_file=open('/home/carson/one/olivettifaces.pkl','wb')  
cPickle.dump(olivettifaces,write_file,-1) 
cPickle.dump(olivettifaces_label,write_file,-1) 
write_file.close() 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 16:28:49 2018

@author: huafenguo
"""
 ########尝试着操作图像分类#####
  ##导入的包
import os
import numpy as np
import matplotlib.pyplot as plt
 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
########尝试导入自己的数据集，but got some questions about it#####
def get_files(file_dir):
      notsanta=[]
      label_notsanta=[]
      santa=[]
      label_santa=[]
      
      for file in os.listdir(file_dir+'/not_santa'):
          notsanta.append(file_dir+'/not_santa'+'/'+file)
          label_notsanta.append(0)     # 添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
      for file in os.listdir(file_dir+'/santa'):
          santa.append(file_dir+'/santa'+'/'+file)
          label_santa.append(1)
      
      #把cat和dog合起来组成一个list （img &lab）
      image_list=np.hstack((notsanta,santa))
      label_list=np.hstack((label_notsanta,label_santa))
      
      #利用shuffle打乱顺序
      temp=np.array((image_list,label_list))
      temp=temp.transpose()
      np.random.shuffle(temp)
      
      #从打乱的temp 中再取出list(img &lab)
      image_list=list(temp[:,0])
      label_list=list(temp[:,1])
      label_list=[int(i)for i in label_list]
      
      return image_list,label_list   #返回两个list分别为图片文件名及其标签，顺序已被打乱
  
  
train_dir='///Users/huafenguo/Desktop/'
image_list,label_list = get_files(train_dir)  
    
print(len(image_list))  
print(len(label_list)) 
  
print(image_list[230])
print(type(image_list[230]))  # right now, it's <class 'numpy.str_'>
 
# #450 为数据长度的20%
Train_image=np.random.rand(len(image_list)-450,2).astype('float64')
Train_label=np.random.rand(len(image_list)-450,1).astype('float64')
  
Test_image=np.random.rand(450,2).astype('float64')
Test_label=np.random.rand(450,1).astype('float64')
  
for i in range(len(image_list)-450,2):
      Train_image[i]=np.array(plt.imread(image_list[i]))
      Train_label[i]=np.array(label_list[i])
      
for i in range(len(image_list)-450,2):
      Test_image[i+450-len(image_list)]=np.array(plt.imread(image_list[i]))
      Test_label[i+450-len(image_list)]=np.array(label_list[i])
 
 
train_x=Train_image
train_y=Train_label
test_x=Test_image
test_y=Test_label  
 
print(type(train_x[200]))  # 已经从一开始的<class'numpy.str_> 变成了<class'numpy.ndarray'>
 
###支持向量机模型 
svm=SVC()
svm.fit(train_x,train_y)
preds=svm.predict(test_x)
print("Accuracy:",accuracy_score(test_y,preds))   ### 测试准确率
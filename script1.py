#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:38:49 2018

@author: huafenguo
"""
import os  
import numpy as np  
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft



def get_files(file_dir):  
    label_cats = []  
    label_dogs = []
    for file in os.listdir(file_dir+'/cat1/'):          # 获取路径
            label_cats.append(0)     #添加标签，该类标签为0，此为2分类例子，多类别识别问题自行添加
         
    for file in os.listdir(file_dir+'/dog1/'):          #获取路径
            label_dogs.append(1)                      #加上标签\
    
            target_y = np.hstack((label_cats, label_dogs))    # 堆起来
    return  target_y

train_dir = '///Users/huafenguo/Desktop/'  
target_y= get_files(train_dir) 



def loadPicture():
    cat_list=[]
    dog_list=[]
    for i in np.arange(37):   # 一共有37 张照片
        image = mpimg.imread('/Users/huafenguo/Desktop/cat1/'+str(i)+'.jpeg');
        cat_list.append(image)          # make a list to 存他们？
        data=np.array(cat_list)
        data_x=data.astype('float64')
        
        

    for i in np.arange(28):  #一共有28张照片ow 
        image=mpimg.imread('/Users/huafenguo/Desktop/dog1/'+str(i)+'.jpeg');
        dog_list.append(image)
        data=np.array(dog_list)
        data_y=data.astype('float')
        

    image_x=np.concatenate([data_x,data_y],axis=0)
    return image_x

    
image_x=loadPicture()


train_x,test_x,train_y,test_y=train_test_split(image_x,target_y,test_size=0.35)

#####

radius = 1;
n_point = radius * 8;
 
def texture_detect():
     train_hist = np.zeros( (42,256) );
     test_hist = np.zeros( (23,256) );
    
     for i in np.arange(42):      # for train hist
         #使用LBP方法提取图像的纹理特征.
         lbp=skft.local_binary_pattern(train_x[i],n_point,radius,'default');
         #统计图像的直方图
         max_bins = int(lbp.max() + 1);
         #hist size:256
         train_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));
 
     for i in np.arange(23):    #for test hist
         lbp = skft.local_binary_pattern(test_x[i],n_point,radius,'default');
         #统计图像的直方图
         max_bins = int(lbp.max() + 1);
         #hist size:256
         test_hist[i], _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));
 
 
     return train_hist,test_hist;   

train_hist,test_hist = texture_detect();

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1);
OneVsRestClassifier(svr_rbf,-1).fit(train_hist, train_y).score(test_hist,test_y)

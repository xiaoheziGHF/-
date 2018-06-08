#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 16:05:21 2018

@author: huafenguo
"""
import cv2
import numpy as np  
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import metrics
# ======##=====##========####=============##===========####======##=============####=
label_benign = []        #make a list
label_malignant = []     #make a list
for i in np.arange(367):  #一共114张照片
    mpimg.imread('/Users/huafenguo/Desktop/良性/40X/'+str(i)+'.png'); #路径
    label_benign.append(0) #标签   
    
for i in np.arange(358):   # 一共有80张照片
    mpimg.imread('/Users/huafenguo/Desktop/恶性/40X/'+str(i)+'.png'); #路径
    label_malignant.append(1) #标签
         
target_y=np.concatenate([label_benign,label_malignant],axis=0) # 放在一起
# ======##=====##========####=============##===========####======##=============####=
#####Gabor 特征
gabor_list1=[]
gabor_list2=[]
for i in np.arange(367):
    image = mpimg.imread('/Users/huafenguo/Desktop/良性/40X/'+str(i)+'.png');
    gabor_list1.append(image)
    image=np.array(gabor_list1)
         
    kern = cv2.getGaborKernel((31,31),3.85,np.pi/4,8.0, 1.0, 0, ktype=cv2.CV_32F)
    image = cv2.filter2D(image, cv2.CV_8UC3, kern)
    
    img1=image.astype('float64')
for i in np.arange(358):
    image = mpimg.imread('/Users/huafenguo/Desktop/恶性/40X/'+str(i)+'.png');
    gabor_list2.append(image)
    image=np.array(gabor_list2)
    
    kern = cv2.getGaborKernel((31,31),3.85,np.pi/4,8.0, 1.0, 0, ktype=cv2.CV_32F)
    image= cv2.filter2D(image, cv2.CV_8UC3, kern)
    
    img2=image.astype('float64')
 
a=img1.reshape(367,10000)
b=img2.reshape(358,10000)
# ======##=====##========####=============##===========####======##=============####=
#####Gauss 特征
#gauss1=[]
#gauss2=[]
#for i in np.arange(367):
#          image=mpimg.imread('/Users/huafenguo/Desktop/良性/40X/'+str(i)+'.png');
#          gauss1.append(image)
#          image1=np.array(gauss1)
#          
#          Gauss1= cv2.GaussianBlur(image1,(5,5),0)
#          img1=Gauss1.astype('float64')
#
#for i in np.arange(358):
#          image=mpimg.imread('/Users/huafenguo/Desktop/恶性/40X/'+str(i)+'.png');
#          gauss2.append(image)
#          image2=np.array(gauss2)
#          
#          Gauss2= cv2.GaussianBlur(image2,(5,5),0)
#          img2=Gauss2.astype('float64')
#
#a=img1.reshape(367,10000)
#b=img2.reshape(358,10000)
# ======##=====##========####=============##===========####======##=============####=
#####尝试
# ======##=====##========####=============##===========####======##=============####=
image_x=np.concatenate([a,b])
train_x,test_x,train_y,test_y=train_test_split(image_x,target_y,test_size=0.2,random_state=75)

# ======##=====##========####=============##===========####======##=============####=
###########################分类器   
###K近邻
knn=KNeighborsClassifier()

#####QDA
qda=QuadraticDiscriminantAnalysis()

### RF随机森林
rf=RandomForestClassifier()

######SVM
svc=SVC(C=1e6, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
####kernel= sigmoid;  kernel=rbf, and rbf stand for Gaussian
#### somehow, change C = 1 or 1e6, might help at some point

#######SVR
#svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1);     ###this value of gamma is specific, 
#svr=OneVsRestClassifier(svr_rbf,-1)                ### for adenosis40 & ductal40 folder


svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.0001);  ### this value fo gamma is sepcific for,
svr=OneVsRestClassifier(svr_rbf,-1)                ### 恶性40X & 良性40X folder

#####somehow, modify, C= 1e3, could see some difference 
#########if amount of picture as input changed, modify value of gamma might help at some point.




###神经网络
nn1=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(30, 6),random_state=1)

nn2=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(10,7), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
##### parameter---> hidden_layer_sizes= (5,3)---> for adenosis40 & ductal40 folder
##### parameter---> hidden_layer_sizes= (10,7)---> for  恶性40X & 良性40X folder

######could try Leaky Relu/Maxout/ ELU, never try tanh and sigmoid 
######solver=lbfgs 是最好的；而activation=relu最好；
######changing hidden_layer_size, might help at some point

# ======##=====##========####=============##===========####======##=============####=
##################################Result, 结果
trained_model=knn.fit(train_x,train_y)   ##.前面可变换不同的分类器
trained_model.fit(train_x,train_y)                                    ##fit 
predictions =trained_model.predict(test_x)                            ####预测 
train_accuracy=accuracy_score(train_y,trained_model.predict(train_x)) ###训练的准确率

################################Result,结果
test_accuracy=accuracy_score(test_y,predictions)                         ###测试的准确率
confusion_matrix=metrics.confusion_matrix(test_y,predictions)            ### 


print(test_y)
print(predictions)
print(train_accuracy)
print(test_accuracy)
print(confusion_matrix)





 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:38:49 2018

@author: huafenguo
"""
import numpy as np  
import matplotlib.image as mpimg
from sklearn.cross_validation import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVR
from skimage import feature as skft
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
for i in np.arange(367):  #一共X张照片
    mpimg.imread('/Users/huafenguo/Desktop/良性/40X/'+str(i)+'.png'); #路径
    label_benign.append(0) #标签   
for i in np.arange(358):   # 一共有X张照片
    mpimg.imread('/Users/huafenguo/Desktop/恶性/40X/'+str(i)+'.png'); #路径
    label_malignant.append(1) #标签
         
target_y=np.concatenate([label_benign,label_malignant],axis=0) # 放在一起

# ======##=====##========####=============##===========####======##=============####=
benign_list=[]             
malignant_list=[]
for i in np.arange(367):   # 一共有X张照片
    image = mpimg.imread('/Users/huafenguo/Desktop/良性/40X/'+str(i)+'.png');
    benign_list.append(image)          # make a list to 存他们？
    data=np.array(benign_list)        # 把list放进数组的形式里
    data_x=data.astype('float64')   # 改变类型
         
for i in np.arange(358):  #一共有X张照片
    image=mpimg.imread('/Users/huafenguo/Desktop/恶性/40X/'+str(i)+'.png');
    malignant_list.append(image)
    data=np.array(malignant_list)
    data_y=data.astype('float64')
         
image_x=np.concatenate([data_x,data_y])
# ======##=====##========####=============##===========####======##=============####=
train_x,test_x,train_y,test_y=train_test_split(image_x,target_y,test_size=0.2,random_state=75)
                                                 ####add 了random_state 之后，准确率固定下来了。

## ======##=====##========####=============##===========####======##=============####=
train_hist = np.zeros( (580,256) );  ###shape and train; check train_x for specific parameter
test_hist = np.zeros( (145,256) );   ###shape and test: check test_x for specific parameter

####lbp的参数
radius = 1;
n_point = radius * 8;

   
for i in np.arange(580):      # for train hist
    ###使用LBP方法提取图像的纹理特征.
    lbp=skft.local_binary_pattern(train_x[i],n_point,radius,'default');
 
    max_bins = int(lbp.max() + 1);      ####lbp need to be 2D arrary, float64
    ###hist size:256
    train_hist[i],_ =np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));
         
for i in np.arange(145):    #for test hist
    ###使用LBP方法提取图像的纹理特征.
    lbp = skft.local_binary_pattern(test_x[i],n_point,radius,'default');

    max_bins = int(lbp.max() + 1);
    ###hist size:255
    test_hist[i],_=np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins));
## ======##=====##========####=============##===========####======##=============####=
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

######SVR

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1);
svr=OneVsRestClassifier(svr_rbf,-1)
#####somehow, modify, C= 1e3, could see some difference 

###神经网络
nn1=MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(15, 5),random_state=1)

nn2=MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5,3), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

######could try Leaky Relu/Maxout/ ELU, never try tanh and sigmoid 
######solver=lbfgs 是最好的；而activation=relu最好；
######changing hidden_layer_size, might help at some point


# ======##=====##========####=============##===========####======##=============####=
#################################Result, 结果
trained_model=rf.fit(train_hist,train_y)   ##.前面可变换不同的分类器
trained_model.fit(train_hist,train_y)                                    ##fit 
predictions =trained_model.predict(test_hist)                            ####预测 
train_accuracy=accuracy_score(train_y,trained_model.predict(train_hist)) ###训练的准确率

################################Result,结果
test_accuracy=accuracy_score(test_y,predictions)                         ###测试的准确率
confusion_matrix=metrics.confusion_matrix(test_y,predictions)            ### 


print(test_y)
print(predictions)
print(train_accuracy)
print(test_accuracy)
print(confusion_matrix)


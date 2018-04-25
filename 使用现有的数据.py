#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:05:25 2018

@author: huafenguo
"""
 ###############动用sklearn里面的数据######
#导入模块#
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer    # 数据导入

 
#不同模型的包
from sklearn import svm
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
###
from sklearn.metrics import accuracy_score
 ## #数据#
breastcancer=load_breast_cancer()    
breastcancer_X=breastcancer.data
breastcancer_y=breastcancer.target
  
  
X_train,X_test,y_train,y_test=train_test_split(breastcancer_X,breastcancer_y,test_size=0.3) 
 ##  # 0.3 即测试集占总数据的30%

  
 ###################### 不同的模型############################
###支持向量机模型 
clf=svm.SVC()
clf.fit(X_train,y_train)
preds=clf.predict(X_test)
print("Accuracy:",accuracy_score(y_test,preds))  #### 用来预测准确率  0.63左右
  

 
#### 5 cross validation 
from sklearn.model_selection import cross_val_score
clf=svm.SVC(kernel='linear', C=1)
scores = cross_val_score(clf,  breastcancer_X, breastcancer_y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)) 


#####################可尝试其他模式#################################
####决策树 

clf=tree.DecisionTreeClassifier()
clf.fit(X_train,y_train)
preds=clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,preds)) #####  0.93左右
 
 # ##神经网络模型
clf=MLPClassifier()
clf.fit(X_train,y_train)
preds=clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,preds))  #### 数字不稳定？！！！
 
 ## 最邻近模型#
clf=KNeighborsClassifier()
clf.fit(X_train,y_train)   # 训练
preds=clf.predict(X_test)
print('Accuracy:',accuracy_score(y_test,preds))   ### 0.92 左右
 
####随机森林
clf = RandomForestClassifier()


RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_split=1e-07, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=10, n_jobs=1, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
clf.fit(X_train,y_train)
preds = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test,preds))   # 0.93


































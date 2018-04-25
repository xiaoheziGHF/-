# -
still working on it, try to debug or modify it 
########尝试着操作图像分类
##导入的包
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

######导入自己的数据集
def get_files(file_dir):
        notsanta=[]
        label_notsanta[]
        santa=[]
        label_santa=[]
        
        for file in os.listdir(file_dir+'/not_santa'):
            notsanta.append(file_dir+'/not_santa'+'file')
            label_notsanta.append(0)         #添加标签，该标签为0，此为2分类例子
        for file in os.listdir(file_dir+'/santa'):
            santa.append(file_dir+'/santa'+file)
            label_santa.append(1)
        
        ##把两者合起来组成一个list （image &label）
        image_list=np.hstack(notsanta,santa)
        label_list=np.hstack((label_notsanta,label_santa))
        
        ##用shuffle 打乱顺序
        temp=np.array((image_list,label_list))
        temp=temp.transpose()
        np.random.shuffle(temp)
        
        ##从打乱的temp中再取出list(image&label)
        image_list=list(temp[:,0])
        label_list=list(temp[:,1])
        label_list=[int(i)for i in label_list]
        
        return image_list, label_list #返回两个list分别为图片文字名及其标签，顺序已打乱
        

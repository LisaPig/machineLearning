# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:svm_hrrp.py
@func:Use SVM to achieve Hrrp classification
@time:2018/6/3 0003下午 1:32
"""

from sklearn import svm
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split
from LDA_reduce_dimension import  LDA_reduce_dimension


#1.读取数据集
path='F:/Python_Project/SVM/data/hrrp_train_data.csv'
data=np.loadtxt(path, dtype=float, delimiter=',')



#2.划分数据与标签
y,x=np.split(data,indices_or_sections=(1,),axis=1) #x为数据，y为标签
#print(len(y))
y = y.reshape((len(y),))  # 要转化为数组形式
print(type(y))

#3.LDA特征提取
W = LDA_reduce_dimension(x, y, 2)  # 得到投影矩阵
newX = np.dot(x, W)  # (m*n) *(n*k)=m*k

train_data,test_data,train_label,test_label =train_test_split(newX,y, random_state=1, train_size=0.6,test_size=0.4) #sklearn.model_selection.
print(train_data.shape)


#4.训练svm分类器
classifier=svm.SVC(C=20,kernel='rbf',gamma=50,decision_function_shape='ovo') # ovr:一对多策略
classifier.fit(train_data,train_label.ravel()) #ravel函数在降维时默认是行序优先

#4.计算svc分类器的准确率
print("训练集：",classifier.score(train_data,train_label))
print("测试集：",classifier.score(test_data,test_label))

#也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score
tra_label=classifier.predict(train_data) #训练集的预测标签
tes_label=classifier.predict(test_data) #测试集的预测标签
print("训练集：", accuracy_score(train_label,tra_label) )
print("测试集：", accuracy_score(test_label,tes_label) )

#查看决策函数
# print('train_decision_function:\n',classifier.decision_function(train_data)) # (90,3)
# print('predict_result:\n',classifier.predict(train_data))

#5.绘制图形
#确定坐标轴范围
x1_min, x1_max=newX[:,0].min(), newX[:,0].max() #第0维特征的范围
x2_min, x2_max=newX[:,1].min(), newX[:,1].max() #第1维特征的范围
x1,x2=np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j ] #生成网络采样点
grid_test=np.stack((x1.flat,x2.flat) ,axis=1) #测试点
#指定默认字体
matplotlib.rcParams['font.sans-serif']=['SimHei']#显示汉字
matplotlib.rcParams['axes.unicode_minus'] = False #能够显示符号（负号）
#设置颜色
cm_light=matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
cm_dark=matplotlib.colors.ListedColormap(['g','r','b'] )

#训练样本 三类分开显示
x1_1=[]
x1_2=[]
x2_1=[]
x2_2=[]
x3_1=[]
x3_2=[]
for i in range(len(train_data)):
    if train_label[i] == 0:
        x1_1.append(train_data[i,0])
        x1_2.append(train_data[i,1])
    if train_label[i] == 1:
        x2_1.append(train_data[i,0])
        x2_2.append(train_data[i,1])
    if train_label[i] == 2:
        x3_1.append(train_data[i,0])
        x3_2.append(train_data[i,1])



#图一：特征提取
plt.figure(1)
#plt.scatter(newX[:,0],newX[:,1],c=y,marker='o',cmap=cm_dark) #c=y,
plt.scatter(x1_1,x1_2,c='g',label='战斗机')
plt.scatter(x2_1,x2_2,c='r',label='螺旋桨')
plt.scatter(x3_1,x3_2,c='b',label='民航')

plt.title('LDA特征提取')
plt.legend()  #显示图例

#图二：SVM分类
plt.figure(2)

grid_hat = classifier.predict(grid_test)       # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)     # 预测值的显示

plt.scatter(x1_1,x1_2,c='g',label='战斗机')
plt.scatter(x2_1,x2_2,c='r',label='螺旋桨')
plt.scatter(x3_1,x3_2,c='b',label='民航')

#plt.scatter(newX[:, 0], newX[:, 1], c=y, s=30,cmap=cm_dark,label='train_data')  # 样本
plt.scatter(test_data[:,0],test_data[:,1], c=test_label,s=30,edgecolors='k', zorder=2,cmap=cm_dark) #圈中测试集样本点
# plt.xlabel('花萼长度', fontsize=13)
# plt.ylabel('花萼宽度', fontsize=13)
plt.xlim(x1_min,x1_max)
plt.ylim(x2_min,x2_max)
plt.title('基于hrrp的SVM分类')

plt.legend()  #显示图例
#plt.show()
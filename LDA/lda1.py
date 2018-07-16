# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:svm_Iris.py
@func:Use LDA to achieve Iris flower feature extraction
@time:2018/6/1 0030上午 9:58
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris
import matplotlib

#define converts(字典)
def Iris_label(s):
    it={b'Iris-setosa':0, b'Iris-versicolor':1, b'Iris-virginica':2 }
    return it[s]


def LDA_reduce_dimension(X,y,nComponents):
    '''
    输入：X为数据集(m*n)，y为label(m*1)，nComponents为目标维数
    输出：W 矩阵（n * nComponents）
    '''
    y1= set(y) #set():剔除矩阵y里的重复元素,化为集合的形式
    labels=list(y1) #list():将其转化为列表
    #list 是可变类型，无法进行 hash，或者说凡可变类型都无法进行 hash；
    """
    eg:
        >>> a=[3,2,1,2]
        >>> set(a)
        {1, 2, 3} 
        >>> list(set(a))
        [1, 2, 3]
        
        >>> e=set(a)
        >>> type(e)
        <class 'set'> #集合
        >>> f=list(e)
        >>> type(f)
        <class 'list'>#列表
    """


    xClasses={} #字典
    for label in labels:
       xClasses[label]=np.array([ X[i] for i in range(len(X)) if y[i]==label ])  #list解析
    """
    x=[1,2,3,4]
    y=[5,6,7,8]
    我想让着两个list中的偶数分别相加，应该结果是2+6,4+6,2+8,4+8
    下面用一句话来写:
    >>>[a + b for a in x for b in y if a%2 == 0 and b%2 ==0]  
    """

    #整体均值
    meanAll=np.mean(X,axis=0) # 按列求均值，结果为1*n(行向量)
    meanClasses={}

    #求各类均值
    for label in labels:
        meanClasses[label]=np.mean(xClasses[label],axis=0) #1*n

    #全局散度矩阵
    St=np.zeros((len(meanAll), len(meanAll) ))
    St=np.dot((X - meanAll).T, X - meanAll)

    #求类内散度矩阵
    # Sw=sum(np.dot((Xi-ui).T, Xi-ui))   i=1...m
    Sw=np.zeros((len(meanAll), len(meanAll) )) # n*n
    for i in labels:
        Sw+=np.dot( (xClasses[i]-meanClasses[i]).T, (xClasses[i]-meanClasses[i]) )

    # 求类间散度矩阵
    Sb = np.zeros((len(meanAll), len(meanAll)))  # n*n
    Sb=St-Sw

    #求类间散度矩阵
    # Sb=sum(len(Xj) * np.dot((uj-u).T,uj-u))  j=1...k
    # Sb=np.zeros((len(meanAll), len(meanAll) )) # n*n
    # for i in labels:
    #     Sb+= len(xClasses[i]) * np.dot( (meanClasses[i]-meanAll).T.reshape(len(meanAll),1),
    #                                     (meanClasses[i]-meanAll).reshape(1,len(meanAll))
    #                                )

    # 计算Sw-1*Sb的特征值和特征矩阵
    eigenValues,eigenVectors=np.linalg.eig(
        np.dot( np.linalg.inv(Sw), Sb)
    )
    #提取前nComponents个特征向量
    sortedIndices=np.argsort(eigenValues) #特征值排序
    W=eigenVectors[:,sortedIndices[:-nComponents-1:-1] ] # 提取前nComponents个特征向量
    return W

    """
    np.argsort()
    eg:
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

    Two-dimensional array:

    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])

    >>> np.argsort(x, axis=0)
    array([[0, 1],
           [1, 0]])

    >>> np.argsort(x, axis=1)
    array([[0, 1],
           [0, 1]])
    """




if '__main__'== __name__:

    #1.读取数据集
    path = 'F:/Python_Project/SVM/data/Iris.data'
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: Iris_label})
    # converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
    #y1 = np.empty((150,), dtype=np.int)
    X, y = np.split(data, indices_or_sections=(4,), axis=1)  # x为数据，y为标签
    y=y.reshape((150,) ) #要转化为数组形式

    print(y)
    print("x type:",type(X),"\ny type:",type(y))
    #2.LDA特征提取
    W=LDA_reduce_dimension(X, y, 2) #得到投影矩阵
    newX=np.dot(X,W)# (m*n) *(n*k)=m*k
    #3.绘图
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    plt.figure(1)
    plt.scatter(newX[:,0],newX[:,1],c=y,marker='o') #c=y,

    plt.title('自行实现LDA')


    #4.与sklearn自带库函数对比
    lda_Sklearn=LinearDiscriminantAnalysis(n_components=2)
    lda_Sklearn.fit(X,y)
    newX1=lda_Sklearn.transform(X)
    plt.figure(2)
    plt.scatter(newX1[:, 0], newX1[:, 1], marker='o', c=y)
    plt.title('sklearn自带LDA库函数')


    plt.show()



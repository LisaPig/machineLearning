# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:svm_Iris.py
@func:Use SVM to achieve Iris flower classification
@time:2018/6/1 0030上午 9:58
"""

from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris



def LDA_reduce_dimension(X, y, nComponents):
    '''
    输入：X为数据集(m*n)，y为label(m*1)，nComponents为目标维数
    输出：W 矩阵（n * nComponents）
    '''
    # y1= set(y) #set():剔除矩阵y里的重复元素,化为集合的形式
    labels = list(set(y))  # list():将其转化为列表


    xClasses = {}  # 字典
    for label in labels:
        xClasses[label] = np.array([X[i] for i in range(len(X)) if y[i] == label])  # list解析


    # 整体均值
    meanAll = np.mean(X, axis=0)  # 按列求均值，结果为1*n(行向量)
    meanClasses = {}

    # 求各类均值
    for label in labels:
        meanClasses[label] = np.mean(xClasses[label], axis=0)  # 1*n

    # 全局散度矩阵
    St = np.zeros((len(meanAll), len(meanAll)))
    St = np.dot((X - meanAll).T, X - meanAll)

    # 求类内散度矩阵
    # Sw=sum(np.dot((Xi-ui).T, Xi-ui))   i=1...m
    Sw = np.zeros((len(meanAll), len(meanAll)))  # n*n
    for i in labels:
        Sw += np.dot((xClasses[i] - meanClasses[i]).T, (xClasses[i] - meanClasses[i]))

    # Sb = np.zeros((len(meanAll), len(meanAll)))  # n*n
    # Sb=St-Sw

    # 求类间散度矩阵
    # Sb=sum(len(Xj) * np.dot((uj-u).T,uj-u))  j=1...k
    Sb = np.zeros((len(meanAll), len(meanAll)))  # n*n
    for i in labels:
        Sb += len(xClasses[i]) * np.dot((meanClasses[i] - meanAll).T.reshape(len(meanAll), 1),
                                        (meanClasses[i] - meanAll).reshape(1, len(meanAll))
                                        )

    # 计算Sw-1*Sb的特征值和特征矩阵
    eigenValues, eigenVectors = np.linalg.eig(
        np.dot(np.linalg.inv(Sw), Sb)
    )
    # 提取前nComponents个特征向量
    sortedIndices = np.argsort(eigenValues)  # 特征值排序
    W = eigenVectors[:, sortedIndices[:-nComponents - 1:-1]]  # 提取前nComponents个特征向量
    return W


if '__main__' == __name__:
    # 1.读取数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    X=X[0:100,:]
    y=y[0:100]

    # 2.LDA特征提取
    W = LDA_reduce_dimension(X, y, 1)  # 得到投影矩阵
    newX = np.dot(X, W)  # (m*n) *(n*k)=m*k
    # 3.绘图
    plt.scatter(newX[:, 0],y, marker='o')  # c=y,
    plt.show()



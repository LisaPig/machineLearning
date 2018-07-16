# -*- coding:utf-8 -*-
"""
 线性判别分析(LDA)和python实现（多分类问题）   https://blog.csdn.net/z962013489/article/details/79918758
@file:lda_sample2.py
@time:2018/6/1 0001下午 10:03
"""
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def LDA_dimensionality(X, y, k):
    '''
    X为数据集，y为label，k为目标维数
    '''
    label_ = list(set(y)) #set():去掉列表里的重复元素,化为集合的形式； list():将其转化为列表

    X_classify = {}

    for label in label_:
        X1 = np.array([X[i] for i in range(len(X)) if y[i] == label])
        X_classify[label] = X1

    mju = np.mean(X, axis=0)
    mju_classify = {}

    for label in label_:
        mju1 = np.mean(X_classify[label], axis=0)
        mju_classify[label] = mju1

    #St = np.dot((X - mju).T, X - mju)

    Sw = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
    for i in label_:
        Sw += np.dot((X_classify[i] - mju_classify[i]).T,
                     X_classify[i] - mju_classify[i])

    # Sb=St-Sw

    Sb = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
    for i in label_:
        Sb += len(X_classify[i]) * np.dot((mju_classify[i] - mju).reshape(
            (len(mju), 1)), (mju_classify[i] - mju).reshape((1, len(mju))))

    eig_vals, eig_vecs = np.linalg.eig(
        np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵

    sorted_indices = np.argsort(eig_vals)
    topk_eig_vecs = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征向量
    return topk_eig_vecs


if '__main__' == __name__:

    iris = load_iris()
    X = iris.data
    y = iris.target

    W = LDA_dimensionality(X, y, 2)
    X_new = np.dot((X), W)
    plt.figure(1)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)

    # 与sklearn中的LDA函数对比
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(X, y)
    X_new = lda.transform(X)
    print(X_new)
    plt.figure(2)
    plt.scatter(X_new[:, 0], X_new[:, 1], marker='o', c=y)

    plt.show()
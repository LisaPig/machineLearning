# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:logisticRegression.py
@note:logistic回归
@time:2018/7/11 0011下午 10:45
"""

import numpy as np
import matplotlib.pyplot as plt

"""
函数：加载数据集
"""
def loadDataSet():
    dataMat=[]  #列表list
    labelMat=[]
    txt=open('testSet.txt')
    for line in txt.readlines():
        lineArr=line.strip().split()        #strip():返回一个带前导和尾随空格的字符串的副本
                                            #split():默认以空格为分隔符，空字符串从结果中删除
        dataMat.append( [1.0, float(lineArr[0]), float(lineArr[1]) ])  #将二维特征扩展到三维，第一维都设置为1.0
        labelMat.append(int(lineArr[2]) )

    return dataMat,labelMat


"""
函数：sigmoid函数
"""
def sigmoid(z):
    return 1.0/(1+np.exp(-z) )

"""
函数：梯度上升算法
"""
def gradAscent(dataMat,labelMat):
    dataSet=np.mat(dataMat)                          # m*n
    labelSet=np.mat(labelMat).transpose()            # 1*m->m*1
    m,n=np.shape(dataSet)                            # m*n: m个样本，n个特征
    alpha=0.001                                      # 学习步长
    maxCycles=500                                    # 最大迭代次数
    weights=np.ones( (n,1) )
    for i in range(maxCycles):
        y=sigmoid(dataSet * weights)                 # 预测值
        error=labelSet -y
        weights=weights+ alpha *dataSet.transpose()*error
    #print(type(weights))
    return weights.getA(),weights  ##getA():将Mat转化为ndarray,因为mat不能用index

"""
函数：随机梯度上升算法0.0
改进：每次用一个样本来更新回归系数
"""
def stocGradAscent0(dataMat,labelMat):
    m, n = np.shape(dataMat)  # m*n: m个样本，n个特征
    alpha = 0.001  # 学习步长
    maxCycles=500
    weights = np.ones(n)
    for cycle in range(maxCycles):
        for i in range(m):
            y = sigmoid(sum(dataMat[i] * weights) )  # 预测值
            error = labelMat[i] - y
            weights = weights + alpha  * error* dataMat[i]
        # print(type(weights))
    return weights


"""
函数：改进的随机梯度上升法1.0
改进：1.alpha随着迭代次数不断减小，但永远不会减小到0
     2.通过随机选取样本来更新回归系数
"""
def stocGradAscent1(dataMat,labelMat):

    m, n = np.shape(dataMat)  # m*n: m个样本，n个特征
    maxCycles = 150
    weights = np.ones(n)
    for cycle in range(maxCycles):
        dataIndex=list( range(m))
        for i in range(m):
            alpha = 4 / (1.0 + cycle + i) + 0.01                 # 学习步长
            randIndex=int(np.random.uniform(0,len(dataIndex) ))  #随机选取样本
            y = sigmoid(sum(dataMat[randIndex] * weights ))          # 预测值
            error = labelMat[randIndex] - y
            weights = weights + alpha  * error * dataMat[randIndex]
            del(dataIndex[randIndex])
            # print(type(weights))
    return weights


"""
函数：画出决策边界
"""
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr=np.array(dataMat)
    m,n=np.shape(dataArr)
    x1=[]           #x1,y1:类别为1的特征
    x2=[]           #x2,y2:类别为2的特征
    y1=[]
    y2=[]
    for i in range(m):
        if (labelMat[i])==1:
            x1.append(dataArr[i,1])
            y1.append(dataArr[i,2])
        else:
            x2.append(dataArr[i,1])
            y2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.scatter(x1,y1,s=30,c='red',marker='s')
    ax.scatter(x2,y2,s=30,c='green')

    #画出拟合直线
    x=np.arange(-3.0, 3.0, 0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]    #直线满足关系：0=w0*1.0+w1*x1+w2*x2
    ax.plot(x,y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()




def main():
    dataMat,labelMat=loadDataSet()
    #weights=gradAscent(dataMat,labelMat)[0]
    weights = stocGradAscent0(np.array(dataMat), labelMat)
    #weights = stocGradAscent1(np.array(dataMat), labelMat)
    plotBestFit(weights)



if __name__ == '__main__':
    main()
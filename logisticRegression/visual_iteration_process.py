# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:visual_iteration_process.py
@note:可视化迭代次数与回归系数的关系
@time:2018/7/12 0012下午 8:08
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
    weights=np.ones((n,1))
    weights_array=np.array([])
    for i in range(maxCycles):
        y=sigmoid(dataSet * weights)                 # 预测值
        error=labelSet -y
        weights=weights+ alpha *dataSet.transpose()*error
        weights_array=np.append(weights_array,weights)
    weights_array = weights_array.reshape(maxCycles, n)
    #print(weights_array)
    return weights.getA(),weights_array  ##getA():将Mat转化为ndarray,因为mat不能用index

"""
函数：随机梯度上升算法0.0
改进：每次用一个样本来更新回归系数
"""
def stocGradAscent0(dataMat,labelMat):
    m, n = np.shape(dataMat)  # m*n: m个样本，n个特征
    alpha = 0.001  # 学习步长
    maxCycles=500
    weights = np.ones(n)
    weights_array = np.array([])
    for cycle in range(maxCycles):
        for i in range(m):
            y = sigmoid(sum(dataMat[i] * weights) )  # 预测值
            error = labelMat[i] - y
            weights = weights + alpha  * error* dataMat[i]
            weights_array = np.append(weights_array, weights)
    # print(type(weights))
    weights_array = weights_array.reshape(maxCycles*m, n)
    return weights,weights_array


"""
函数：改进的随机梯度上升法1.0
改进：1.alpha随着迭代次数不断减小，但永远不会减小到0
     2.通过随机选取样本来更新回归系数
"""
def stocGradAscent1(dataMat,labelMat):

    m, n = np.shape(dataMat)  # m*n: m个样本，n个特征
    maxCycles = 500
    weights = np.ones(n)
    weights_array=np.array([])
    for cycle in range(maxCycles):
        dataIndex=list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + cycle + i) + 0.01                 # 学习步长
            randIndex=int(np.random.uniform(0,len(dataIndex) ))  #随机选取样本
            y = sigmoid(sum(dataMat[randIndex] * weights ))          # 预测值
            error = labelMat[randIndex] - y
            weights = weights + alpha  * error * dataMat[randIndex]
            del(dataIndex[randIndex])
            # print(type(weights))
            weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(maxCycles*m, n)
    return weights,weights_array

"""
函数：可视化迭代过程
"""
def visual(weights_array1,weights_array2,weights_array3):

    import matplotlib as mpl
# 指定默认字体
    mpl.rcParams['font.sans-serif'] = ['SimHei'] #显示汉字
    mpl.rcParams['axes.unicode_minus'] = False  # 能够显示符号（负号）
    fig,axs=plt.subplots(3,3,figsize=(15,10))

    x1=np.arange(0,len(weights_array1),1)
    #绘制w0和迭代次数的关系
    axs[0][0].plot(x1,weights_array1[:,0])
    axs[0][0].set_title('梯度上升算法：回归系数与迭代次数的关系')
    axs[0][0].set_ylabel('w0',)
    # 绘制w1和迭代次数的关系
    axs[1][0].plot(x1, weights_array1[:, 1])
    axs[1][0].set_ylabel('w1')
    # 绘制w2和迭代次数的关系
    axs[2][0].plot(x1, weights_array1[:, 2])
    axs[2][0].set_ylabel('w2')
    axs[2][0].set_xlabel('迭代次数')

    x2 = np.arange(0, len(weights_array2), 1)
    # 绘制w0和迭代次数的关系
    axs[0][1].plot(x2, weights_array2[:, 0])
    axs[0][1].set_title('随机梯度上升算法')
    axs[0][1].set_ylabel('w0')
    # 绘制w1和迭代次数的关系
    axs[1][1].plot(x2, weights_array2[:, 1])
    axs[1][1].set_ylabel('w1')
    # 绘制w2和迭代次数的关系
    axs[2][1].plot(x2, weights_array2[:, 2])
    axs[2][1].set_ylabel('w2')
    axs[2][1].set_xlabel('迭代次数')

    x3 = np.arange(0, len(weights_array3), 1)
    # 绘制w0和迭代次数的关系
    axs[0][2].plot(x3, weights_array3[:, 0])
    axs[0][2].set_title('改进的随机梯度上升算法')
    axs[0][2].set_ylabel('w0')
    # 绘制w1和迭代次数的关系
    axs[1][2].plot(x3, weights_array3[:, 1])
    axs[1][2].set_ylabel('w1')
    # 绘制w2和迭代次数的关系
    axs[2][2].plot(x3, weights_array3[:, 2])
    axs[2][2].set_ylabel('w2')
    axs[2][2].set_xlabel('迭代次数')

    plt.show()

if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    weights1, weights_array1 = gradAscent(dataMat, labelMat)
    weights2, weights_array2 = stocGradAscent0(np.array(dataMat), labelMat)
    weights3, weights_array3 = stocGradAscent1(np.array(dataMat), labelMat)

    visual(weights_array1, weights_array2,weights_array3)
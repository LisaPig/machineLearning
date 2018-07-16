# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:k_means.py
@time:2018/7/14 0014下午 8:27
"""
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


"""
函数：读取数据集文件
输出：读取到的数据集（列表形式）
"""
def loadDataSet(filename):
    dataMat=[]
    fr=open(filename)
    for line in fr.readlines():
        curLine=line.strip().split('\t')
        floatLine=list(map(float,curLine) )   #将每一行的数据映射成float型
        dataMat.append(floatLine)
    return dataMat


"""
函数：计算欧式两个向量之间的距离
输入：两个向量vecA、vecB
"""

def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB , 2)))

"""
函数：k个质心随机初始化
返回值：随机初始化的k个质心数据点
说明：此函数为k-means函数初始化k个质心
注意：质心的每一维度的取值范围应 确保在数据的边界之内
"""
def randCentroids(dataSet,k):
    m,n=np.shape(dataSet)
    centroids=np.mat(np.zeros( (k,n)))
    for i in range(n):
        minI=np.min(dataSet[:,i])
        rangeI=float( np.max(dataSet[:,i]) - minI)
        centroids[:,i]=minI+ rangeI*np.random.rand(k,1)
    return centroids

"""
函数：k-means函数
说明：此为kmeans最基本的函数，后续会在此基础上有改进版----二分K-Means
"""
def k_means(dataSet,k,distCompute=distEclud,creatCentroids=randCentroids):
    m,n=np.shape(dataSet)
    clusterAssign=np.mat(np.zeros( (m,2))) #存储分配的簇和误差
    centroids=creatCentroids(dataSet,k)
    clusterChanged=True
    clusChangedCount=0

    #为每个样本分配簇，直到簇分配不发生变化-》稳定态
    while clusterChanged:
        clusterChanged =False
        clusChangedCount =clusChangedCount+1
        for i in range(m):
            minDist = np.Inf
            minIndex = -1
            for j in range(k):
                dist=distCompute(centroids[j,:],dataSet[i,:])
                if dist<minDist:
                    minDist=dist
                    minIndex=j
            if clusterAssign[i,0]!=minIndex:
                clusterChanged = True
            clusterAssign[i,0]=minIndex
            clusterAssign[i,1]=minDist**2  #SSE：均方误差
        print("第%d次分配簇的结果为：" %(clusChangedCount))
        print(centroids)

        #重新计算质心：每一次有样本簇变化，都要重新计算质心
        # 1.找到所有属于第i簇的样本
        # 2.计算该簇的样本均值，作为质心
        for i in range(k):
            samplesI = dataSet[ np.nonzero(clusterAssign[:,0].A== i )[0] ]
            centroids[i,:]=np.mean(samplesI,axis=0)  #按列求均值
    print( '\n')
    # draw(dataSet,centroids,clusterAssign,clusChangedCount)
    return centroids,clusterAssign


"""
函数：二分K-Means聚类算法
说明：是在基本k-means算法的基础上改进的二分k-means算法->优点是可以避免陷入局部最小值，而非全局最小值
算法步骤: 
1.将所有点看成一个簇，求质心
2.当簇数目小于k时
   对于每一个簇
       计算总误差
       在给定的簇上面进行k-均值聚类（k=2），即一分为2
       计算将该簇一分为二后的总误差
   选择使得误差最小的那个簇进行划分操作
"""
def bi_K_means(dataSet,k,distCompute=distEclud):
    m,n=np.shape(dataSet)
    #初始化
    clusterAssign=np.mat(np.zeros( (m,2) ))
    #创建初始的一个簇
    centroid0=np.mean(dataSet,axis=0).tolist()[0]
    centroidList=[centroid0]
    #绘制初始划分簇
    draw(dataSet, np.mat(centroidList), clusterAssign, k=len(centroidList))

    for i in range(k):
        clusterAssign[i,1]=distCompute(np.mat(centroid0), dataSet[i,:]) **2
    while (len(centroidList)<k ):
        sseMin=np.inf
        for i in range(len(centroidList)):
            curCluster=dataSet[np.nonzero(clusterAssign[:,0].A== i)[0],:]
            #为每一簇进行二分类
            centroidMat,splitClustAssign=k_means(curCluster,k=2)
            sseSplit=sum(splitClustAssign[:,1])   #当前划分簇的sse误差
            sseNotSplit=sum(clusterAssign[np.nonzero(clusterAssign[:,0].A!= i)[0],1]) #非当前划分簇的sse误差
            print("sseSplit: %d, sseNotSplit: %d" %(sseSplit,sseNotSplit))
            if (sseSplit+sseNotSplit) < sseMin:
                bestClustToSplit=i
                bestCentroid=centroidMat
                bestClustAssign=splitClustAssign.copy()
                sseMin=sseSplit+sseNotSplit
        #根据误差最小的那个簇进行划分操作
        #print(bestClustToSplit,bestCentroid)
        bestClustAssign[np.nonzero(bestClustAssign[:,0].A ==1)[0],0]=len(centroidList)
        bestClustAssign[np.nonzero(bestClustAssign[:,0].A ==0)[0],0]= bestClustToSplit
        #print("the best clust to split is: %d" %(bestClustToSplit))
        #print('the len of bestClustAssign is: ' ,(bestClustAssign))
        centroidList[bestClustToSplit]=bestCentroid[0,:].tolist()[0]  #caution!
        #centroidList.append(bestCentroid[0,:].tolist()[0])  #error
        centroidList.append(bestCentroid[1,:].tolist()[0])
        print('len of centroids is : %d' %(len(centroidList)))
        clusterAssign[np.nonzero(clusterAssign[:,0].A==bestClustToSplit)[0],:]=bestClustAssign
        #绘制划分过程
        #draw(dataSet,np.mat(centroidList),clusterAssign,k=len(centroidList))
    return np.mat(centroidList),clusterAssign



"""
函数：绘制簇分类结果
"""
def draw(dataMat,centroids,clusterAssign,k):
    # 指定默认字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
    matplotlib.rcParams['axes.unicode_minus'] = False  # 能够显示符号（负号）

    colors=['r','g','b','yellow','black','pink']

    plt.figure(1)
    #绘制质心
    for i in range(k):
        plt.scatter(x=centroids[i,0],y=centroids[i,1],marker='*',color=colors[i])
    #绘制数据点
    m=np.shape(dataMat)[0]
    for j in range(m):
        colorJ=int( clusterAssign[j, 0] )
        plt.scatter(x=dataMat[j,0],y=dataMat[j,1],c=colors[colorJ])

    plt.show()




if __name__ =="__main__":
    dataSet=loadDataSet("data\\testSet2.txt")
    dataMat=np.mat(dataSet)
    centroids,clusterAssign=k_means(dataMat,k=3)#bi_K_means(dataMat,3)
    draw(dataMat,centroids,clusterAssign,3)






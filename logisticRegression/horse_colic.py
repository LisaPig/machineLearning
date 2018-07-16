# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:horse_colic.py
@note:疝气马能否痊愈预测
@time:2018/7/12 0012下午 10:11
"""
import numpy as np
import logisticRegression as LR

"""
函数：sigmoid回归分类
"""
def classifyVector(dataIn,weights):
    h=LR.sigmoid(sum(dataIn*weights))
    if h>0.5:
        return 1.0
    else:
        return 0.0

"""
函数：疝气预测
"""
def colicTest():
    trainData=open('data\horseColicTraining.txt')
    testData = open('data\horseColicTest.txt')
    trainSet=[]
    trainLabel=[]
    for line in trainData.readlines():
        curLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float (curLine[i]))
        trainSet.append(lineArr)
        trainLabel.append(float(curLine[21]))
    trainWeights=LR.stocGradAscent1(np.array(trainSet),trainLabel)
    errorCount=0
    numTestVec=0

    testSet = []
    testLabel = []
    for line in testData.readlines():
        numTestVec+=1
        curLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float (curLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(curLine[21]):
                errorCount+=1
    errorRate=float(errorCount/numTestVec)
    print("the error rate is: %f" % errorRate)
    return errorRate

def multiTest():
    numTests=10
    errorSum=0
    for k in range(numTests):
        errorSum+=colicTest()
    print("after %d iterations the average error rate is: %f"
          %(numTests,errorSum/float(numTests)))

if __name__ == '__main__':
    multiTest()
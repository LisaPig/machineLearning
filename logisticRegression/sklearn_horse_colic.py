# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:sklearn_horse_colic.py
@note:利用sklearn的LR方法实现 疝气马预测
@time:2018/7/13 0013上午 11:30
@reference:
"""

from sklearn.linear_model import LogisticRegression

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


    testSet = []
    testLabel = []
    for line in testData.readlines():
        curLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float (curLine[i]))
        testSet.append(lineArr)
        testLabel.append(float(curLine[21]))
    #分类器
    classifier = LogisticRegression(solver='sag', max_iter=5000).fit(trainSet, trainLabel)
    test_accurcy = classifier.score(testSet, testLabel) * 100
    print("the accurate rate is: %f" % test_accurcy)
"""
solver：优化算法选择参数，只有五个可选参数，即newton-cg,lbfgs,liblinear,sag,saga。
默认为liblinear。solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择，分别是： 
liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部
        分的样本来计算梯度，适合于样本数据多的时候。
saga：线性收敛的随机优化算法的的变重。

总结： 
liblinear适用于小数据集，而sag和saga适用于大数据集因为速度更快。
对于多分类问题，只有newton-cg,sag,saga和lbfgs能够处理多项损失，而liblinear受限于一对剩余(OvR)。
啥意思，就是用liblinear的时候，如果是多分类问题，得先把一种类别作为一个类别，剩余的所有类别作为
另外一个类别。一次类推，遍历所有类别，进行分类。
newton-cg,sag和lbfgs这三种优化算法时都需要损失函数的一阶或者二阶连续导数，因此不能用于没有
连续导数的L1正则化，只能用于L2正则化。而liblinear和saga通吃L1正则化和L2正则化。
同时，sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，
比如大于10万，sag是第一选择。但是sag不能用于L1正则化，所以当你有大量的样本，又需要L1正则化的话
就要自己做取舍了。要么通过对样本采样来降低样本量，要么回到L2正则化。
从上面的描述，大家可能觉得，既然newton-cg, lbfgs和sag这么多限制，如果不是大样本，
我们选择liblinear不就行了嘛！错，因为liblinear也有自己的弱点！我们知道，逻辑回归有二元逻辑
回归和多元逻辑回归。对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many(MvM)两种。
而MvM一般比OvR分类相对准确一些。郁闷的是liblinear只支持OvR，不支持MvM，这样如果我们需要相对
精确的多元逻辑回归时，就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归
不能使用L1正则化了。


对于我们这样的小数据集，sag算法需要迭代上千次才收敛，而liblinear只需要不到10次。
"""


if __name__ == '__main__':
    colicTest()
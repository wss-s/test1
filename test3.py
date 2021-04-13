# -*- coding: utf-8 -*-
# @Time    : 2021/3/27 19:17
# @Author  : Yanfeng Li
# @FileName: demo2.py
# @Software: PyCharm
import sys
from random import random
from pandas import read_csv  # 导入pandas库
from sklearn.decomposition import PCA  # 用python的sklearn库进行PCA（主成分分析）
from sklearn.svm import SVC  # 从sklearn中导入svm模型
from sklearn import datasets  # 加载数据集
from sklearn.model_selection import train_test_split  # 导入train_test_split函数进行训练集、测试集划分
from sklearn.neighbors import KNeighborsClassifier  # 从sklearn中导入knn模型
from sklearn.linear_model import LogisticRegression  # 从sklearn中导入逻辑回归模型
from sklearn.ensemble import RandomForestClassifier  # 从sklearn中导入随机森林模型
from sklearn import tree  # 从sklearn中导入决策树模型
from sklearn.ensemble import GradientBoostingClassifier  # 从sklearn中导入全称梯度下降模型
from sklearn.ensemble import AdaBoostClassifier  # 从sklearn中导入自适应增强模型
from sklearn.naive_bayes import GaussianNB  # 从sklearn中导入高斯贝叶斯分类器模型
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # 从sklearn中导入线性判别分析模型
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis  # 从sklearn中导入二次判别分析模型
from sklearn.decomposition import FastICA  # 用python的sklearn库进行ICA（独立成分分析）
from sklearn.model_selection import cross_val_score  # 用python的sklearn库进行交叉验证
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import matthews_corrcoef, roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from scipy.signal import savgol_filter

nameList = ['KNN','SVM','逻辑回归','随机森林','决策树','全称梯度下降','自适应增加算法','高斯贝叶斯分类器','二次判别分析']
mcList = [[],[],[],[],[],[],[],[],[]]
rasList = [[],[],[],[],[],[],[],[],[]]
f1List = [[],[],[],[],[],[],[],[],[]]
psList = [[],[],[],[],[],[],[],[],[]]
rsList = [[],[],[],[],[],[],[],[],[]]
accuracyList = [[],[],[],[],[],[],[],[],[]]

filename = 'CodNIR320withlabel.csv'  # 文件名
dataset = read_csv(filename, header=None)  # 使用read_csv函数读取文件参数
newY = dataset.iloc[:, -1].values  # 读取最后一列的数据，即读取标签
dataset = dataset.iloc[:, :-1]  # 删除最后一列，读取除标签外的数据
classifyNum = input('是否需要转成二分类(Y/N):')
# 将读取的分类数据进行转化，即1-4类均转化成0，5-8类转化成1
if classifyNum == 'Y' or classifyNum == 'y':
    lst = []
    for i in newY:
        if i <= 4:
            lst.append(float(0))
        else:
            lst.append(float(1))
    newY = np.array(lst)
    # print(newY)
# print(type(newY))     # 打印newY的类型
# print(dataset.shape)  # 打印dataset的行列数
# print(dataset)    # 打印dataset

# 光谱的预处理：平滑、归一化
plt.plot(dataset)
plt.show()
y = savgol_filter(dataset, 5, 3, mode= 'nearest')
# 可视化图线
min_max_scaler = preprocessing.MinMaxScaler()   # min-max标准化
newX = min_max_scaler.fit_transform(dataset)
plt.plot(newX)
plt.show()
# 特征变量提取或筛选：
choice = input('是否需要降维(Y/N):')
if choice == 'Y' or choice == 'y':
    dimensionality_reduction = input("请输入降维方式(PCA/ICA):")
    if dimensionality_reduction == 'PCA' or dimensionality_reduction == 'pca':
        n_com = int(input('需要降至几维:'))
        tem = '%d' % n_com
        pca = PCA(n_components=n_com)  # 指定维数，特征变量提取，降至n_com维
        newX = pca.fit_transform(dataset)  # 训练并返回
        print('贡献率：', pca.explained_variance_ratio_)  # 输出贡献率

    else:
        n_com = int(input('多少个独立成分:'))
        tem = '%d' % n_com
        fast_ica = FastICA(n_components=n_com)  # 独立成分为n_comp个
        newX = fast_ica.fit_transform(dataset)  # newX为解混后的n_comp个独立成分，shape=[m,n]
        # print(fast_ica.explained_variance_ratio_)  # 输出贡献率
else:  # 输入其他信息则不降维
    newX = dataset.values  # 整个训练集进行训练


# print(type(newX))
# print(newX)   # 打印newX
# print(len(newX))  # 打印newX的长度
# print(type(newX)) # 打印dataset的类型
# print('-------------------------')
# 创建一个txt文件，文件名为mytxtfile
def text_create(name):
    desktop_path = "D:\\"
    # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')


filename = 'log'
text_create(filename)
output = sys.stdout
outputfile = open("D:\\" + filename + '.txt', 'w')


def appendList(X_test, y_test, model, i):
    y_pre = model.predict(X_test)
    mcList[i].append(matthews_corrcoef(y_test, y_pre))
    rasList[i].append(roc_auc_score(y_test, y_pre))
    f1List[i].append(f1_score(y_test, y_pre))
    psList[i].append(precision_score(y_test, y_pre))
    rsList[i].append(recall_score(y_test, y_pre))
    accuracyList[i].append(accuracy_score(y_test, y_pre))


def outputAverNum():
    for i in range(9):
        print('--------' + nameList[i] + '----------')
        print('matthews_corrcoef（交叉验证）:', np.mean(mcList[i]))
        print('roc_auc_score（交叉验证）:', np.mean(rasList[i]))
        print('f1_score（交叉验证）:', np.mean(f1List[i]))
        print('precision_score（交叉验证）:', np.mean(psList[i]))
        print('recall_score（交叉验证）:', np.mean(rsList[i]))
        print('accuracy_score（交叉验证）:', np.mean(accuracyList[i]))
        print('========================================')


cross_validation = input("是否需要交叉验证(Y/N):")
if cross_validation == 'Y' or cross_validation == 'y':

    # # knn = KNeighborsClassifier(n_neighbors=6)
    # # knnScores = cross_val_score(knn, newX, newY, cv=10, scoring='accuracy')
    # # print(knnScores)
    # # print('knn准确率(交叉验证)：', knnScores.mean())
    # '''
    # all_data = pd.read_csv('CodNIR320withlabel.csv', header=None)
    # print(all_data)
    # trainingSet = list(range(all_data))  # 创建存储训练集的索引值的列表
    # testSet = []  # 储存测试集的索引值的列表
    # for i in range(test_num):  # 从all_data个数据中，随机挑选出(all_data - test_num)个作为训练集,test_num个做测试集
    #     randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索索引值
    #     testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
    #     del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    # '''
    # k_range = range(1, 31)
    # k_scores = []
    # # 循环遍历寻找最好的k值
    # for k in k_range:
    #     knn = KNeighborsClassifier(n_neighbors=k)
    #     knnScores = cross_val_score(knn, newX, newY, cv=5, scoring='accuracy')
    #     k_scores.append(knnScores.mean())
    # # 显示图例
    # plt.plot(k_range, k_scores)
    # plt.xlabel('Values of K for KNN')
    # plt.ylabel('Cross-Validated Accuracy')
    # plt.show()
    #
    print('参数:  是否二分类:' + classifyNum, '是否需要降维:' + choice, '降维方式:'
          + dimensionality_reduction, '降维数：' + tem, '交叉验证:' + cross_validation,
          file=outputfile)
    sys.stdout = outputfile

    sKFold = StratifiedKFold(n_splits=4, shuffle=True)
    for train, test in sKFold.split(newX, newY):
        X_train, y_train, X_test, y_test = newX[train], newY[train], newX[test], newY[test]
        knn = KNeighborsClassifier(n_neighbors=5)  # 调用knn模型，设置n_neighbors参数
        knn.fit(X_train, y_train)  # 用knn进行训练
        appendList(X_test, y_test, knn, 0)
        svm = SVC()  # 调用svm
        svm.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, svm, 1)
        lr = LogisticRegression(penalty='l2', solver='newton-cg',
                                multi_class='multinomial')  # 调用逻辑回归，参数分别为：正则化选择参数，优化算法选择参数，分类方式选择参数
        lr.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, lr, 2)
        rf = RandomForestClassifier(n_estimators=8)  # 调用随机森林
        rf.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, rf, 3)
        dt = tree.DecisionTreeClassifier()  # 调用决策树
        dt.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, dt, 4)
        gb = GradientBoostingClassifier(n_estimators=200)  # 调用全称梯度下降算法
        gb.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, gb, 5)
        adaBoost = AdaBoostClassifier()  # 调用自适应增强算法
        adaBoost.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, adaBoost, 6)
        gau = GaussianNB()  # 调用高斯贝叶斯分类器
        gau.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, gau, 7)
        qda = QuadraticDiscriminantAnalysis()  # 调用二次判别分析
        qda.fit(X_train, y_train)  # 训练
        appendList(X_test, y_test, qda, 8)
    outputAverNum()
    #
    # svm = SVC()  # 调用svm
    # svmScores = cross_val_score(svm, newX, newY, cv=5, scoring='accuracy')
    # print('svm准确率(交叉验证)：', svmScores.mean())
    #
    # # print(newX)
    # # print(newY)
    # # print(len(newX), len(newY))
    # # print(type(newX), type(newY))
    #
    # '''报错ValueError: The number of classes has to be greater than one; got 1 class'''
    # train_sizes, train_loss, test_loss = learning_curve(svm, newX, newY, cv=5, scoring='neg_mean_squared_error',
    #                                                     train_sizes=np.linspace(.1, 1.0,
    #                                                                             5))  # neg_mean_squared_error代表求均值平方差
    # train_loss_mean = -np.mean(train_loss, axis=1)  # 计算train_loss第一行的均值,负数要加负号边正
    # test_loss_mean = -np.mean(test_loss, axis=1)  # 计算test_loss第一行的均值,负数要加负号边正
    # # 设置样式与label
    # plt.plot(train_sizes, train_loss_mean, 'o-', color='r', label='Training')
    # plt.plot(train_sizes, test_loss_mean, 'o-', color='g', label='Cross_validation')
    # plt.xlabel('Training examples')
    # plt.ylabel('Loss')
    # # 显示图例
    # plt.legend(loc='best')
    # plt.show()
    #
    # lr = LogisticRegression(penalty='l2', solver='newton-cg',
    #                         multi_class='multinomial')  # 调用逻辑回归，
    # # 参数分别为：正则化选择参数，优化算法选择参数，分类方式选择参数
    # lrScores = cross_val_score(lr, newX, newY, cv=5, scoring='accuracy')
    # print('逻辑回归准确率(交叉验证)：', lrScores.mean())  # 输出准确率
    #
    # rf = RandomForestClassifier(n_estimators=8)  # 调用随机森林
    # rfScores = cross_val_score(rf, newX, newY, cv=5, scoring='accuracy')
    # print('随机森林准确率(交叉验证)：', rfScores.mean())  # 输出准确率
    #
    # dt = tree.DecisionTreeClassifier()  # 调用决策树
    # dtScores = cross_val_score(dt, newX, newY, cv=5, scoring='accuracy')
    # print('决策树准确率(交叉验证)：', dtScores.mean())  # 输出准确率
    #
    # gb = GradientBoostingClassifier(n_estimators=200)  # 调用全称梯度下降算法
    # gbScores = cross_val_score(gb, newX, newY, cv=5, scoring='accuracy')
    # print('全称梯度下降准确率(交叉验证)：', gbScores.mean())  # 输出准确率
    #
    # adaBoost = AdaBoostClassifier()  # 调用自适应增强算法
    # adaBoostScores = cross_val_score(adaBoost, newX, newY, cv=5, scoring='accuracy')
    # print('自适应增强算法准确率(交叉验证)：', adaBoostScores.mean())  # 输出准确率
    #
    # gau = GaussianNB()  # 调用高斯贝叶斯分类器
    # gauScores = cross_val_score(gau, newX, newY, cv=5, scoring='accuracy')
    # print('高斯贝叶斯分类器准确率(交叉验证)：', gauScores.mean())  # 输出准确率
    #
    # qda = QuadraticDiscriminantAnalysis()  # 调用二次判别分析
    # qdaScores = cross_val_score(qda, newX, newY, cv=5, scoring='accuracy')
    # print('二次判别分析准确率(交叉验证)：', qdaScores.mean())  # 输出准确率


def output(X_test, y_test, model):
    y_pre = model.predict(X_test)
    mc = matthews_corrcoef(y_test, y_pre)
    ras = roc_auc_score(y_test, y_pre)
    f1 = f1_score(y_test, y_pre)
    ps = precision_score(y_test, y_pre)
    rs = recall_score(y_test, y_pre)
    accuracy = accuracy_score(y_test, y_pre)
    string = str(model)
    print('--------' + string + '----------')
    print('matthews_corrcoef:', mc)
    print('roc_auc_score:', ras)
    print('f1_score:', f1)
    print('precision_score:', ps)
    print('recall_score:', rs)
    print('accuracy_score:', accuracy)
    print('========================================')


if cross_validation == 'N' or cross_validation == 'n':
    print('-------------------------')
    X_train, X_test, y_train, y_test = train_test_split(newX, newY, test_size=0.3)  # 30%的数据划分为测试集，70%的数据划分为训练集
    '''
    KNN参数说明
    https://blog.csdn.net/weixin_41990278/article/details/93169529
    '''
    knn = KNeighborsClassifier(n_neighbors=5)  # 调用knn模型，设置n_neighbors参数
    knn.fit(X_train, y_train)  # 用knn进行训练
    output(X_test, y_test, knn)
    # y_pre = knn.predict(X_test)
    # mc = matthews_corrcoef(y_test, y_pre)
    # ras = roc_auc_score(y_test, y_pre)
    # f1 = f1_score(y_test, y_pre)
    # ps = precision_score(y_test, y_pre)
    # rs = recall_score(y_test, y_pre)
    # accuracy = accuracy_score(y_test, y_pre)
    # print('--------KNN----------')
    # print('matthews_corrcoef:',mc)
    # print('roc_auc_score:',ras)
    # print('f1_score:',f1)
    # print('precision_score:',ps)
    # print('recall_score:',rs)
    # print('accuracy_score:',accuracy)
    # print('---------------------')
    # print('KNN准确率：', knn.score(X_test, y_test))  # 输出准确率

    '''
    SVC参数说明
    https://blog.csdn.net/sinat_23338865/article/details/80290162
    '''
    svm = SVC()  # 调用svm
    svm.fit(X_train, y_train)  # 训练
    output(X_test, y_test, svm)
    # print('SVM准确率：', svm.score(X_test, y_test))  # 输出准确率

    '''
    LR参数说明
    https://blog.csdn.net/qq_42370261/article/details/84852595
    '''
    lr = LogisticRegression(penalty='l2', solver='newton-cg',
                            multi_class='multinomial')  # 调用逻辑回归，参数分别为：正则化选择参数，优化算法选择参数，分类方式选择参数
    lr.fit(X_train, y_train)  # 训练
    output(X_test, y_test, lr)
    # print('逻辑回归准确率：', lr.score(X_test, y_test))  # 输出准确率

    '''
    RF参数说明
    https://blog.csdn.net/MG_ApinG/article/details/84872092
    '''
    rf = RandomForestClassifier(n_estimators=8)  # 调用随机森林
    rf.fit(X_train, y_train)  # 训练
    output(X_test, y_test, rf)
    # print('随机森林准确率：', rf.score(X_test, y_test))  # 输出准确率

    '''
    DT参数说明
    https://blog.csdn.net/bylfsj/article/details/104453310/
    '''
    dt = tree.DecisionTreeClassifier()  # 调用决策树
    dt.fit(X_train, y_train)  # 训练
    output(X_test, y_test, dt)
    # print('决策树准确率：', dt.score(X_test, y_test))  # 输出准确率

    '''
    GB参数说明
    https://blog.csdn.net/han_xiaoyang/article/details/52663170
    '''
    gb = GradientBoostingClassifier(n_estimators=200)  # 调用全称梯度下降算法
    gb.fit(X_train, y_train)  # 训练
    output(X_test, y_test, gb)
    # print('全称梯度下降准确率：', gb.score(X_test, y_test))  # 输出准确率

    '''
    AdaBoost参数说明
    https://www.cnblogs.com/pinard/p/6136914.html
    '''
    adaBoost = AdaBoostClassifier()  # 调用自适应增强算法
    adaBoost.fit(X_train, y_train)  # 训练
    output(X_test, y_test, adaBoost)
    # print('自适应增强算法准确率：', adaBoost.score(X_test, y_test))  # 输出准确率

    '''
    GAU参数说明
    ?
    '''
    gau = GaussianNB()  # 调用高斯贝叶斯分类器
    gau.fit(X_train, y_train)  # 训练
    output(X_test, y_test, gau)
    # print('高斯贝叶斯分类器准确率：', gau.score(X_test, y_test))  # 输出准确率

    # lda = LinearDiscriminantAnalysis()    # 调用线性判别分析分类器
    # lda.fit(X_train, y_train)   # 训练
    # print('线性判别分析准确率：', lda.score(X_test, y_test))    # 输出准确率

    '''
    QDA参数说明
    https://blog.csdn.net/qsczse943062710/article/details/75977118
    '''
    qda = QuadraticDiscriminantAnalysis()  # 调用二次判别分析
    qda.fit(X_train, y_train)  # 训练
    output(X_test, y_test, qda)
    # print('二次判别分析准确率：', qda.score(X_test, y_test))  # 输出准确率

# preResult = knn.predict(X_test)
# trueResult = y_test
# print('KNN预测结果集：',preResult)
# print('KNN真实结果集：',trueResult)
# print('-------------------------')
# preResult = svm.predict(X_test)
# trueResult = y_test
# print('SVM预测结果集：',preResult)
# print('SVM真实结果集：',trueResult)

outputfile.close()  # close后才能看到写入的数据

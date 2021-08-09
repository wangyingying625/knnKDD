import os
import csv
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
from sklearn import neighbors

def toNum(str):
    if(str=='normal.'):
        return int(1)
    elif(str=='neptune.'):
        return int(2)
    elif(str=='smurf.'):
        return int(3)
    elif(str=='buffer_overflow.'):
        return int(4)
    else:
        print(str)

# 1. 加载数据集
# -----------------------------------------第一步 加载数据集-----------------------------------------
def loadData():
    fr = open("./dataset/kddcup.data_10_percent")
    lines = fr.readlines()
    line_nums = len(lines)
    line_nums = 1000
    # 创建line_nums行 para_num列的矩阵
    # TODO: FIX ME
    x_mat = np.zeros((line_nums, 36), dtype=str)
    y_label = []
    # 划分数据集和特征和类标
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = (line.split(','))
        # TODO : FIXME 先不用带着字符串的特征
        x_mat[i, :] = item_mat[5:41]  # 前41个特征,从0开始取41个
        y_label.append(int(toNum(item_mat[-1]))) # 类标
    fr.close()
    return x_mat, y_label


# 2. 划分数据集
# -----------------------------------------第二步 划分数据集-----------------------------------------
# 划分数据集 测试集40%

def division():
    print("start loadData")
    x_mat, y_label = loadData()
    print("over loadData")
    y = []
    print(f"length for y_label {len(y_label)}")
    i = 0
    # start = time.time()
    # last_point = start
    # for n in y_label:
    #     point = time.time()
    #     y.append(n)
    #     y1 = np.array(y, dtype=str)  # list转换数组
    #     i = i + 1
    #     if not (i % 5000):
    #         print(f'{i}/{len(y_label)} == {i/len(y_label)} '
    #               f'|| all used time {point - start} '
    #               f'|| 5000 cycle used time {point - last_point}')
    #         last_point = point

    # print("over append")
    train_data, test_data, train_target, test_target = train_test_split(x_mat, y_label, test_size=0.4, random_state=42)
    print(np.array(train_data).shape, np.array(train_target).shape)
    print(np.array(test_data).shape, np.array(test_target).shape)

    # 3. KNN训练
    # -----------------------------------------第三步 KNN训练-----------------------------------------
    print("begin training")
    clf = neighbors.KNeighborsClassifier(30)
    print(clf)
    clf.fit(train_data, train_target)
    print(clf)
    print("end training")
    result = clf.predict(test_data)
    print(result)
    # print(test_target)
    # 4. 评价算法
    # -----------------------------------------第四步 评价算法-----------------------------------------
    print (sum(result==test_target)) #预测结果与真实结果比对
    print(metrics.classification_report(test_target, result))  #准确率 召回率 F值
    print(type(y_label[0]))
    # 5. 降维可视化
    #----------------------------------------第五步 降维可视化---------------------------------------
    pca = PCA(n_components=2)
    newData = pca.fit_transform(test_data)
    plt.figure()
    plt.scatter(newData[:,0], newData[:,1], c=test_target, s=50)
    plt.show()


if __name__ == '__main__':
    division()
    print("finish")

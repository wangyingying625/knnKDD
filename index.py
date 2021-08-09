import os
import csv
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import FeatureHasher
from sklearn.decomposition import PCA
from sklearn import neighbors



def preHandel_data():
    source_file='./dataset/kddcup.data_10_percent'
    handled_file='./dataset/kddcup.data_10_percent.csv'
    data_file=open(handled_file,'w',newline='')     #python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file,'r') as data_source:
        csv_reader=csv.reader(data_source)
        csv_writer=csv.writer(data_file)
        count=0   #记录数据的行数，初始化为0
        for row in csv_reader:
            temp_line=np.array(row)   #将每行数据存入temp_line数组里
            temp_line[1]=handleProtocol(row)   #将源文件行中3种协议类型转换成数字标识
            temp_line[2]=handleService(row)    #将源文件行中70种网络服务类型转换成数字标识
            temp_line[3]=handleFlag(row)       #将源文件行中11种网络连接状态转换成数字标识
            temp_line[41]=handleLabel(row)   #将源文件行中23种攻击类型转换成数字标识
            csv_writer.writerow(temp_line)
            count+=1
            #输出每行数据中所修改后的状态
            print(count,'status:',temp_line[1],temp_line[2],temp_line[3],temp_line[41])
        data_file.close()
#将相应的非数字类型转换为数字标识即符号型数据转化为数值型数据
def find_index(x,y):
    return [i for i in range(len(y)) if y[i]==x]


# 定义将源文件行中3种协议类型转换成数字标识的函数
def handleProtocol(input):
    protocol_list = ['tcp', 'udp', 'icmp']
    if input[1] in protocol_list:
        return find_index(input[1], protocol_list)[0]


# 定义将源文件行中70种网络服务类型转换成数字标识的函数
def handleService(input):
    service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                    'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                    'hostnames',
                    'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                    'ldap',
                    'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
                    'nntp',
                    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
                    'shell',
                    'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i',
                    'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    if input[2] in service_list:
        return find_index(input[2], service_list)[0]
#定义将源文件行中11种网络连接状态转换成数字标识的函数
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3] in flag_list:
        return find_index(input[3],flag_list)[0]


# 定义将源文件行中攻击类型转换成数字标识的函数(训练集中共出现了22个攻击类型，而剩下的17种只在测试集中出现)
def handleLabel(input):
    # label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
    # 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
    # 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
    # 'spy.', 'rootkit.']
    global label_list  # 在函数内部使用全局变量并修改它
    if input[41] in label_list:
        return find_index(input[41], label_list)[0]
    else:
        label_list.append(input[41])
        return find_index(input[41], label_list)[0]


# if __name__ == '__main__':
#     start_time = time.clock()
#     global label_list  # 声明一个全局变量的列表并初始化为空
#     label_list = []
#     preHandel_data()
#     end_time = time.clock()
#     print("Running time:", (end_time - start_time))  # 输出程序运行时间


# 1. 加载数据集
# -----------------------------------------第一步 加载数据集-----------------------------------------
def loadData():
    fr = open("./dataset/kddcup.data_10_percent.csv")
    lines = fr.readlines()
    line_nums = len(lines)
    line_nums = 1000
    # 创建line_nums行 para_num列的矩阵
    x_mat = np.zeros((line_nums, 36), dtype=str)
    y_label = []
    # 划分数据集和特征和类标
    for i in range(line_nums):
        line = lines[i].strip()
        item_mat = (line.split(','))
        x_mat[i, :] = item_mat[5:41]  # 前41个特征,从0开始取41个
        y_label.append(int(item_mat[-1])) # 类标
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

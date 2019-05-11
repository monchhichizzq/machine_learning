import numpy as np
import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from sklearn import model_selection
from keras.preprocessing.sequence import pad_sequences

from utilities import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf  # Version 1.0.0 (some previous versions are used in past commits)
from sklearn import metrics

from collections import Counter

import os
import warnings

warnings.filterwarnings("ignore")
print("TensorFlow Version: %s" % tf.__version__)


# DTW可以针对两个时间序列找到最优的非线定位（non-linear alignment)
def euclid_dist(t1,t2):
    return np.sqrt(sum((t1-t2)**2))

def dtw(s1,s2):
    r, c = len(s1), len(s2)
    D0 = np.zeros((r+1,c+1))
    D0[0,1:,] = np.inf
    D0[1:,0] = np.inf
    D1 = D0[1:,1:]

    for i in range(r):
        for j in range(c):
            # s1[i] = np.array(s1[i])
            # s2[j] = np.array(s2[j])
            # manhattan_distances
            # D1[i, j] = abs(s2[j]-s1[i])
            # Euclidean distances
            D1[i, j] = np.sqrt(np.sum(np.square(s2[j]-s1[i])))

    M = D1.copy()
    # print('normal_dist',M)
    D0[1, 1] = D1[0, 0]
    for i in range(r-1):
        for j in range(c-1):
            D0[i+2,j+2] = min(D1[i,j],D1[i,j+1],D1[i+1,j])+D1[i+1, j+1]

    # print('D0',D0)
    # print('D1',D1)
    #代码核心，动态计算最短距离

    i,j = np.array(D0.shape) - 2
    # print('i',i,'j',j)
    #最短路径
    # print i,j
    p,q = [i],[j]
    while(i>0 or j>0):
        tb = np.argmin((D0[i,j],D0[i,j+1],D0[i+1,j]))

        if tb==0 :
            i-=1

            j-=1

        elif tb==1 :
            i-=1

        else:
            j-=1
        p.insert(0,i)
        q.insert(0,j)


    # print('M', M)
    # # 原始距离矩阵
    # print(zip(p, q))
    # # 匹配路径过程
    # print('D1',D1)
    # Cost Matrix或者叫累积距离矩阵
    dtw_path_min = D0[-1, -1]
    print('dtw_dist', D1[-1, -1])
    dtw_path_min = D0[-1, -1]
    plt.scatter(p, q, alpha=0.6)  # 绘制散点图，透明度为0.6（这样颜色浅一点，比较好看）
    plt.show()
    # 序列距离
    return D0[-1, -1]

def DTWDistance(s1, s2):
    DTW={}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            # dist= (s1[i]-s2[j])**2
            dist = np.sqrt(np.sum(np.square(s2[j]-s1[i])))
            DTW[(i, j)] = dist + min(DTW[(i-1, j)], DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])


 # 第一种方法是强制执行局部性约束。这种方法假设当i和j相距太远，则q_i和c_j不需要匹配。这个阈值则由一个给定的窗口大小w决定。这种方法可以提高窗口内循环的速度
def FastDTWDistance(s1, s2, w):
    DTW={}

    w = max(w, abs(len(s1)-len(s2)))

    for i in range(-1,len(s1)):
        for j in range(-1,len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i-w), min(len(s2), i+w)):
            dist= (s1[i]-s2[j])**2
            DTW[(i, j)] = dist + min(DTW[(i-1, j)],DTW[(i, j-1)], DTW[(i-1, j-1)])

    return np.sqrt(DTW[len(s1)-1, len(s2)-1])

# 另一种方法是使用LB Keogh下界方法DTW的边界
def LB_Keogh(s1,s2,r):
    LB_sum=0
    for ind,i in enumerate(s1):

        lower_bound=min(s2[(ind-r if ind-r>=0 else 0):(ind+r)])
        upper_bound=max(s2[(ind-r if ind-r>=0 else 0):(ind+r)])

        if i>upper_bound:
            LB_sum=LB_sum+(i-upper_bound)**2
        elif i<lower_bound:
            LB_sum=LB_sum+(i-lower_bound)**2

    return np.sqrt(LB_sum)



def rnn_input():
    data_path = '/Users/babalia/Desktop/final_project/data_processing/Gesture_data/txt_rnn'
    data, labels, n_channels, n_classes, seq_len, LABELS = read_data(data_path)

    print('Data', data.shape, 'labels', labels.shape)
    # 这里我们得到了rnn_input （batch, time_length, joints）
    # (68, 1960, 18)

    #x为数据集的feature熟悉，y为label.
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.8)

    print ("Training data shape: N = {:d}, steps = {:d}, channels = {:d}".format(x_train.shape[0],
                                                                                 x_train.shape[1],
                                                                                 x_train.shape[2]))
    print ("Test data shape: N = {:d}, steps = {:d}, channels = {:d}".format(x_test.shape[0],
                                                                             x_test.shape[1],
                                                                             x_test.shape[2]))
    # y = one_hot(labels,n_classes)
    # print('y', y.shape)
    # one_hot encoding
    # y_train = one_hot(labels_train, n_classes)
    # y_test = one_hot(labels_test, n_classes)

    # X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, random_state = 123)
    # X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, test_size = 0.2)

    # train_ratio = 0.75
    # x_train_, x_val, y_train_, y_val = train_test_split(x_train,y_train,train_size=train_ratio,random_state=123)



    # one_hot encoding
    # y_train_ = one_hot(y_train,n_classes)
    # # y_val = one_hot(y_val, n_classes)
    # y_test = one_hot(y_test, n_classes)


    return data, labels, n_channels, n_classes, seq_len, x_train, x_test, y_train, y_test,LABELS

data, labels, n_channels, n_classes, seq_len, X_train, X_test, y_train, y_test, LABELS = rnn_input()
print('x_tr', X_train.shape, 'y_tr', y_train.shape)
# print('x_val', x_val.shape, 'x_val', y_val.shape)
print('x_test', X_test.shape, 'y_test', y_test.shape)

print(data.shape[0])
print(data[0].shape)
def dtw_train(x_train,y_train):
    train_same = []
    train_diff = []
    max_same = []
    max_diff = []
    for i in range(x_train.shape[0]):
        # for j in [i+1, x_train.shape[0]+1]:
        for j in range(x_train.shape[0]):
            distance_DTW = DTWDistance(x_train[i], x_train[j])
            if y_train[i] == y_train[j]:
                train_same.append(distance_DTW)
                # print('distance_same', distance_DTW)
            else:
                train_diff.append(distance_DTW)
                # print('distance_diff', distance_DTW)
        max_1 = max(train_same)
        max_2 = min(train_diff)
        max_same.append(max_1)
        max_diff.append(max_2)
    return max_diff

max = dtw_train(X_train,y_train)
print('max',max)

def dtw_test(x_train,y_train,x_test,y_test,max):
    predictions = []
    for i in range(X_test.shape[0]):
        pred = []
        for j in range(X_train.shape[0]):
            distance_DTW = DTWDistance(X_test[i], X_train[j])

            if distance_DTW <= max[j]:
                pred.append(y_train[j])
            # else:
            #     pred.append(None)

        if pred:
            pred = Counter(pred)

            top = pred.most_common(1)[0][0]

        else:

            top = X_train.shape[0] + 1
        print('pred', pred)
        print('top',i, top)
        predictions.append(top)

    return np.array(predictions)

predictions = dtw_test(X_train,y_train,X_test,y_test,max)
accuracy = np.mean(y_test == predictions)
# Results

# predictions = one_hot_predictions.argmax(1)
print(predictions.shape)

print("Testing Accuracy: {}%".format(100*accuracy))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))

print("")
print("Confusion Matrix:")
confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)
normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100

print("")
print("Confusion matrix (normalised to % of total test data):")
print(normalised_confusion_matrix)
print("Note: training and testing data is not equally distributed amongst classes, ")
print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")

# Plot Results:
width = 8
height = 8
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(n_classes)
plt.xticks(tick_marks, LABELS, rotation=90)
plt.yticks(tick_marks, LABELS)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()




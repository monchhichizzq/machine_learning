# from skimage import io,transform
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# from CNN_input import get_files

import tensorflow as tf
import numpy as np
import time
import cv2
from scipy.interpolate import spline
import keras
from sklearn.model_selection import train_test_split

import time
import math
import os
import sys
import os, os.path,shutil
import numpy as np
import sklearn
from sklearn import svm
import sklearn.metrics as sm
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import itertools
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB as nb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as lr
from sklearn.neural_network import MLPClassifier as mlp
from sklearn import metrics




# #读取图片


def load_data(data_path):
    print('data path:', data_path)
    class_names = []
    data = []
    labels = []
    label_index = 0
    data_new = []
    label_new = []
    for label_name in os.listdir(data_path):
        class_names.append(label_name)
        # for image_directory in os.listdir(data_path + '/' + label_name):
        #     print('image_directory',image_directory)
        for image_name in os.listdir(data_path + '/' + label_name):

            if 'D' in os.path.splitext(image_name)[0]:

                image = cv2.imread(data_path + '/' + label_name +'/' + image_name)
                # image = cv2.resize(image, (img_rows, img_cols))

                data.append(image)
                labels.append(label_index)

        label_index = label_index + 1

    print('labels', np.array(labels).shape)
    print('label_index',label_index)
    length_data = range(len(data))
    # labels = keras.utils.to_categorical(labels, len(class_names))
    print('class_number:', len(class_names), 'class names:', str(class_names))
    data = np.array(data)
    data = data.astype('float32')
    return data, labels, len(class_names), class_names




img_rows = 240
img_cols = 360
# data_path = '/Users/babalia/Desktop/final_project/data/action_data/Vmax_0.5d/CNN_3parts_maxV_0.5d'
data_path = "/Users/babalia/Desktop/final_project/data/action_data/Vmax_1d/CNN_3parts"
# data_path = '/Users/babalia/Desktop/final_project/data/action_data/Vmax_0.7d/CNN_3parts_Vmax_0.7d'
data, labels, n_classes, LABELS = load_data(data_path)
print('img', np.array(data).shape)
print('y', np.array(labels).shape)


#x为数据集的feature熟悉，y为label.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.4)


# 数据重塑
x_train_rows = X_train.reshape(X_train.shape[0], img_rows * img_cols * 3)
x_test_rows = X_test.reshape(X_test.shape[0], img_rows * img_cols * 3)
# print(x_train_rows.shape, x_test_rows.shape)
# 对像素进行缩放
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

x_train_rows = minmax.fit_transform(x_train_rows)
x_test_rows = minmax.fit_transform(x_test_rows)


# 进行logistic regression 计算


from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(max_depth = 3, min_samples_leaf = 10, min_samples_split =  80, n_estimators = 10, oob_score=True, random_state=10)
n_estimators = range(10, 71, 10)
max_features = 'auto'
max_depth = range(3, 20, 2)
min_samples_split = range(80, 150, 20)
# for i in min_samples_split:
#     rf = RandomForestClassifier(n_estimators = 30, max_depth = 5, min_samples_split = i, oob_score=True, random_state=10)
#     rf.fit(x_train_rows, y_train)
#     rf_pred_prob = rf.predict_proba(x_test_rows) # 查看第一个测试样本属于各个类别的概率
#     predictions = rf.predict(x_test_rows)
#     accuracy = np.mean(y_test == predictions)
#     acc = rf.score(x_test_rows,y_test)
#     # print('pred_prob',rf_pred_prob)
#     # print('pred', predictions)
#     # print('labels_test', y_test)
#     print(i, ' Accuracy = %f' % accuracy,acc)

# 可见袋外分数已经很高（理解为袋外数据作为验证集时的准确率，也就是模型的泛化能力），而且AUC分数也很高（AUC是指从一堆样本中随机抽一个，抽到正样本的概率比抽到负样本的概率 大的可能性）。相对于GBDT的默认参数输出，RF的默认参数拟合效果对本例要好一些。


def rf(X_train, X_test, y_train, y_test):
    # 开始调优使用GridSearchCV找到,最优参数
    rf = RandomForestClassifier()
    # 首先对n_estimators进行网格搜索
    n_estimators = range(10, 71, 10)
    # 对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
    max_depth = range(3, 14, 2)
    # 再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参
    min_samples_split = range(80, 150, 20)
    min_samples_leaf = range(10, 60, 10)
    param_grid = dict(n_estimators =n_estimators, max_depth=max_depth, min_samples_split = min_samples_split, min_samples_leaf=min_samples_leaf)
    gridrf = GridSearchCV(rf, param_grid, cv=10, scoring='accuracy', verbose=1)
    gridrf.fit(x_train_rows, y_train)
    print('best score is:', str(gridrf.best_score_))
    print('best params are:', str(gridrf.best_params_))

rf(x_train_rows,x_test_rows,y_train,y_test)


#
# # Results
#
# # predictions = one_hot_predictions.argmax(1)
# print(predictions.shape)
#
# print("Testing Accuracy: {}%".format(100*accuracy))
#
# print("")
# print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
# print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
# print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))
#
# print("")
# print("Confusion Matrix:")
# confusion_matrix = metrics.confusion_matrix(y_test, predictions)
# print(confusion_matrix)
# normalised_confusion_matrix = np.array(confusion_matrix, dtype=np.float32)/np.sum(confusion_matrix)*100
#
# print("")
# print("Confusion matrix (normalised to % of total test data):")
# print(normalised_confusion_matrix)
# print("Note: training and testing data is not equally distributed amongst classes, ")
# print("so it is normal that more than a 6th of the data is correctly classifier in the last category.")
#
# # Plot Results:
# width = 8
# height = 8
# plt.figure(figsize=(width, height))
# plt.imshow(
#     normalised_confusion_matrix,
#     interpolation='nearest',
#     cmap=plt.cm.rainbow
# )
# plt.title("Confusion matrix \n(normalised to % of total test data)")
# plt.colorbar()
# tick_marks = np.arange(n_classes)
# plt.xticks(tick_marks, LABELS, rotation=90)
# plt.yticks(tick_marks, LABELS)
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
#

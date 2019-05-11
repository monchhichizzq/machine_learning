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
# data_path = "/Users/babalia/Desktop/final_project/data/action_data/Vmax_1d/CNN_3parts"
# data_path = '/Users/babalia/Desktop/final_project/data/action_data/Vmax_0.7d/CNN_3parts_Vmax_0.7d'
# data_path = '/Users/babalia/Desktop/final_project/data/action_data/Vmax_0.5d/CNN_3parts_maxV_0.5d'
data_path = '/Users/babalia/Desktop/final_project/data/action_data/Vmax_1.5d/CNN_3parts_Vmax_1.5d'
data, labels, n_classes, LABELS = load_data(data_path)
print('img', np.array(data).shape)
print('y', np.array(labels).shape)


#x为数据集的feature熟悉，y为label.
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size = 0.4)

# print ("Training data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_train.shape[0],
#                                                                              X_train.shape[1],
#                                                                              X_train.shape[2]))
# print ("Test data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_test.shape[0],
#                                                                          X_test.shape[1],
#                                                                          X_test.shape[2]))

# 数据重塑
x_train_rows = X_train.reshape(X_train.shape[0], img_rows * img_cols * 3)
x_test_rows = X_test.reshape(X_test.shape[0], img_rows * img_cols * 3)
# print(x_train_rows.shape, x_test_rows.shape)
# 对像素进行缩放
from sklearn.preprocessing import MinMaxScaler
minmax = MinMaxScaler()

x_train_rows = minmax.fit_transform(x_train_rows)
x_test_rows = minmax.fit_transform(x_test_rows)
# print('minmax',x_train_rows, x_test_rows)

# 进行knn计算
from sklearn.neighbors import KNeighborsClassifier

# 我们通过迭代来探索k值多少才合适
k = k_range = list(range(1, 20, 2))
print('k',k)
# #
for i in k:
    model = KNeighborsClassifier(n_neighbors=i, weights='uniform', algorithm='auto', p=2, n_jobs=-1)
    model.fit(x_train_rows, y_train)
    predictions = model.predict(x_test_rows)
    accuracy = np.mean(y_test == predictions)
    print('pred', predictions)
    print('labels_test', y_test)
    print('k = %s, Accuracy = %f' % (i, np.mean(y_test == predictions)))


model = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='auto', p=2, n_jobs=-1)
model.fit(x_train_rows, y_train)
predictions = model.predict(x_test_rows)
accuracy = np.mean(y_test == predictions)
print('pred', predictions)
print('labels_test', y_test)
print('k = %s, Accuracy = %f' % (7, np.mean(y_test == predictions)))



def k(X_train, X_test, y_train, y_test):
    # 开始调优使用GridSearchCV找到,最优参数
    knn = KNeighborsClassifier()
    # 设置k的范围
    k_range = list(range(3, 15, 2))
    leaf_range = list(range(1, 2))
    weight_options = ['uniform', 'distance']
    algorithm_options = ['auto', 'ball_tree', 'kd_tree', 'brute']
    param_gridknn = dict(n_neighbors=k_range, weights=weight_options, algorithm=algorithm_options, leaf_size=leaf_range)
    gridKNN = GridSearchCV(knn, param_gridknn, cv=10, scoring='accuracy', verbose=1)
    gridKNN.fit(X_train, y_train)
    predictions = gridKNN.predict(x_test_rows)
    accuracy = np.mean(y_test == predictions)
    print('k = %s, Accuracy = %f' % (k_range, np.mean(y_test == predictions)))
    print('best score is:', str(gridKNN.best_score_))
    print('best params are:', str(gridKNN.best_params_))
#
# # 从KNN的分类准确率来看，是要比我们随机猜测类别提高了不少。我们随机猜测图片类别时，准确率大概是10%，KNN方式的图片分类可以将准确率提高到35%左右。当然有兴趣的小伙伴还可以去测试一下其他的K值，同时在上面的算法中，默认距离衡量方式是欧式距离，还可以尝试其他度量距离来进行建模。 虽然KNN在test数据集上表现有所提升，但是这个准确率还是太低了。除此之外，KNN有一个缺点，就是所有的计算时间都在predict阶段，当一个新的图来的时候，涉及到大量的距离计算，这就意味着一旦我们要拿它来进行图像识别，那可能要等非常久才能拿到结果，而且还不是那么的准。
# k(x_train_rows,x_test_rows,y_train,y_test)

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


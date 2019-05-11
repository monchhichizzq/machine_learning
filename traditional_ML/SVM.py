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
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels




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
# data_path = "/Users/babalia/Desktop/final_project/data/action_data/Vmax_1d/CNN_3parts"
# data_path = '/Users/babalia/Desktop/final_project/data/action_data/Vmax_0.7d/CNN_3parts_Vmax_0.7d'
data_path = '/Users/babalia/Desktop/final_project/data/action_data/Vmax_1.5d/CNN_3parts_Vmax_1.5d'
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
print('minmax',x_train_rows.shape, x_test_rows.shape)

# 进行SVM计算
from sklearn.svm import SVC
C = [0.1,1,10,100]
for i in C:
    # 用来平衡分类间隔margin和错分样本的
    svclassifier = SVC(C=i, kernel='linear')
    # {'C': 10, 'gamma': 0.0001, 'kernel': 'rbf'}
    # svclassifier = SVC(C = 10, kernel='rbf', gamma=0.0001)
    svclassifier.fit(x_train_rows, y_train)
    predictions = svclassifier.predict(x_test_rows)
    accuracy = np.mean(y_test == predictions)
    print('pred', predictions)
    print('labels_test', y_test)
    print(i,'Accuracy = %f' % (np.mean(y_test == predictions)))
#
#
# tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-1,1e-2,1e-3, 1e-4],
#                      'C': [0.1, 1, 10, 100, 1000]},
#                     {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
# #
# scores = ['precision', 'recall']
# for score in scores:
#     print("# Tuning hyper-parameters for %s" % score)
#     print()
#
#     # 调用 GridSearchCV，将 SVC(), tuned_parameters, cv=5, 还有 scoring 传递进去，
#     clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
#                        scoring='%s_macro' % score)
#     # 用训练集训练这个学习器 clf
#     clf.fit(x_train_rows, y_train)
#
#     print("Best parameters set found on development set:")
#     print()
#
#     # 再调用 clf.best_params_ 就能直接得到最好的参数搭配结果
#     print(clf.best_params_)
#
#     print()
#     print("Grid scores on development set:")
#     print()
#     means = clf.cv_results_['mean_test_score']
#     stds = clf.cv_results_['std_test_score']
#
#     # 看一下具体的参数间不同数值的组合后得到的分数是多少
#     for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean, std * 2, params))
#
#     print()
#
#     print("Detailed classification report:")
#     print()
#     print("The model is trained on the full development set.")
#     print("The scores are computed on the full evaluation set.")
#     print()
#     y_true, y_pred = y_test, clf.predict(x_test_rows)
#
#     # 打印在测试集上的预测结果与真实值的分数
#     print(classification_report(y_true, y_pred))
#
#     print()

# Results
#
# # predictions = one_hot_predictions.argmax(1)
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
width = 7
height = 7
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

#
# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix, without normalization'
#
#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     # classes = classes[unique_labels(y_true, y_pred)]
#
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')
#
#     print(cm)
#
#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')
#
#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")
#
#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax
#
#
# np.set_printoptions(precision=2)
#
# # Plot non-normalized confusion matrix
# plot_confusion_matrix(y_test, predictions, classes=LABELS, title='Confusion matrix, without normalization')
#
# # Plot normalized confusion matrix
# plot_confusion_matrix(y_test, predictions, classes=LABELS, normalize=True, title='Normalized confusion matrix')
#
# plt.show()
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
data_path = "/Users/babalia/Desktop/final_project/data_processing/Gesture_data/CNN_A"
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

# 进行logistic regression 计算

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1.0,penalty = 'l2')
lr.fit(x_train_rows, y_train)
lr_pred_prob = lr.predict_proba(x_test_rows) # 查看第一个测试样本属于各个类别的概率
predictions = lr.predict(x_test_rows)
accuracy = np.mean(y_test == predictions)
acc = lr.score(x_test_rows,y_test)
print('pred_prob',lr_pred_prob)
print('pred', predictions)
print('labels_test', y_test)
print(' Accuracy = %f' % accuracy,acc)


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


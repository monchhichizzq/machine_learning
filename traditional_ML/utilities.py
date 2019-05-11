import pandas as pd
import numpy as np
import os,sys
import cv2
import keras
from pandas import read_csv
import csv
from sklearn.model_selection import train_test_split

import os
import numpy as np
import sklearn
from sklearn import svm
import sklearn.metrics as sm
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import itertools
from sklearn.externals import joblib
from PIL import Image
import cv2
import time
import codecs
import math
import colorsys
from keras.preprocessing.sequence import pad_sequences



# 将中心点1固定于一点图的中心 如果图的比例是（240，120），中心点为（180，60）
def tranlate_to_center(X):
    dx = X[1][0]-180
    dy = X[1][1]-60
    for i in range(len(X)):
        # delta_X = g_txt[1][0]-180 / delta_y = g_txt[1][1]-60
        X[i][0] = X[i][0] - dx
        X[i][1] = X[i][1] - dy
    return X


def remove_zero_rows(X):
 # X is a scipy sparse matrix. We want to remove all zero rows from it
 nonzero_row_indice, _ = X.nonzero()
 unique_nonzero_indice = np.unique(nonzero_row_indice)
 return X[unique_nonzero_indice]

def zero_rows_complete(X,X_pre):
    nonzero_row_indice, _ = X.nonzero()
    nonzero_row_indice, _ = X.nonzero()
    unique_nonzero_indice = np.unique(nonzero_row_indice)
    # print('nonzero_row_indice', nonzero_row_indice)
    # print('unique_nonzero_indice', unique_nonzero_indice)
    for i in range(len(X)):
        if i not in nonzero_row_indice:
            X[i] = X_pre[i]
            if X[i][0] == 0 :
                X[i] = X[1]
    return X


# move legs and head
def remove_cols_in_gesture(X):
    # x = np.delete(X,[8,9,10,11,12,13,14,19,20,21,22,23,24], axis = 0)
    # body_25
    # x = np.delete(X, [8, 9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22, 23, 24], axis=0)
    # pose_18 remove legs [8, 9, 10, 11, 12, 13]
    X_short = []
    for i in range(len(X)):
        if i % 10 == 0:
            X_short.append(X[i])
    # print('X_short', np.array(X_short).shape)
    return np.array(X_short)

def distance(p1, p2):
    d = np.sqrt((p2[0]-p1[0])**2+(p2[1]-p1[1])**2)
    return d

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径

    # else:
    #     print("---  There is this folder!  ---")

def read_data(path):
    gesture_all_1D = []
    gesture_all_2D = []
    labels = []
    class_names = []
    label = 0
    # 每个类的名字
    n_class = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        filenames.sort()
        dirnames.sort()
        for dirname in dirnames:
            if len(os.path.splitext(dirname)[0])>3:
                p_1 = os.path.join(dirpath, dirname)
                class_names.append(dirname)
                for (dpath, dnames, dfiles) in os.walk(p_1):
                    n_class += 1
                    dnames.sort()
                    # gesture_all_1D = [] label
                    for dname in dnames:
                        p = os.path.join(dpath,dname)
                        # p /Users/babalia/Desktop/final_project/database/CAD_60/txt_gesture/brushing_teeth/00
                        # print('p',p)
                        for (dirpath_2, dirnames_2, filenames_2) in os.walk(p):
                            # 重新排序00。。。10.。。。100.。。1000.。。
                            filenames_2.sort()
                            filenames_2_rearrange = []
                            for filename_2 in filenames_2:
                                if len(os.path.splitext(filename_2)[0]) == 2:
                                    filenames_2_rearrange.append(filename_2)
                            for filename_2 in filenames_2:
                                if len(os.path.splitext(filename_2)[0]) == 3:
                                    filenames_2_rearrange.append(filename_2)
                            for filename_2 in filenames_2:
                                if len(os.path.splitext(filename_2)[0]) == 4:
                                    filenames_2_rearrange.append(filename_2)
                            for filename_2 in filenames_2:
                                if len(os.path.splitext(filename_2)[0]) == 5:
                                    filenames_2_rearrange.append(filename_2)
                            # print('filenames_2_rearrange', filenames_2_rearrange)

                            gesture = []
                            gesture_1D = []
                            t = 0
                            for file in filenames_2_rearrange:
                                p_file_txt = os.path.join(p,file)
                                # print('file', p_file_txt)
                                g_txt = np.loadtxt(open(p_file_txt, "rb"))
                                if gesture:
                                    g_txt = zero_rows_complete(g_txt, gesture[t-1])
                                g_txt_1D = []
                                for i in range(len(g_txt)):
                                    # 计算各点到中心的距离
                                    d = distance(g_txt[i], g_txt[1])
                                    g_txt_1D.append(d)
                                # print(np.array(g_txt_1D).T.shape)
                                # one completed gesture
                                gesture.append(g_txt)
                                gesture_1D.append(g_txt_1D)
                                t += 1
                            # gesture_1D = remove_cols_in_gesture(gesture_1D)
                            gesture_all_1D.append(gesture_1D)
                            gesture_all_2D.append(gesture)


                            # print('label', label)
                            # print('gesture', np.array(gesture).shape)
                            # print('gesture-1D', np.array(gesture_1D).shape)

                        # print('gesture_all_1D', np.array(gesture_all_1D).shape)
                        # pad sequence
                        gesture_all_1D_padded = pad_sequences(gesture_all_1D, padding='post')
                        gesture_all_2D_padded = pad_sequences(gesture_all_2D, padding='post')
                        # print('gesture_all_1D_padded', gesture_all_1D_padded.shape)
                        # 这里我们得到了rnn_input （batch, time_length, joints）
                        # 我们要把他们分类 然后label

                        labels.append(label)

                label += 1
                # print(label)

    # path_folder = '/Users/babalia/Desktop/final_project/database/CAD_60/rnn_gesture_all/'
    # mkdir(path_folder)
    # path_save = '/Users/babalia/Desktop/final_project/database/CAD_60/rnn_gesture_all/gesture_all_1D_padded.csv'
    # path_save_labels = '/Users/babalia/Desktop/final_project/database/CAD_60/rnn_gesture_all/gesture_all_1D_padded_labels.csv'
    # print('path_save',path_save)
    # np.savetxt(path_save, gesture_all_1D_padded, delimiter=',')
    # np.savetxt(path_save_labels, labels, delimiter=',')
    # print('labels', np.array(labels).shape)
    # print('gesture_all_1D_padded', gesture_all_1D_padded.shape)

    # n_classes = 14
    # n_channels = 18
    #
    # # Hyperparameters
    # lstm_size = 3*n_channels     # 3 times the amount of channels 3*18
    # print('lstm_size', lstm_size)
    # lstm_layers = 2        # Number of layers
    #
    # batch_size = 68      # Batch size 68个手势
    # seq_len = 1960       # Number of steps 1960
    # learning_rate = 0.0001  # Learning rate (default is 0.001)
    # epochs = 10000
    n_channels = gesture_all_1D_padded.shape[2]
    n_classes = label
    seq_len = gesture_all_1D_padded.shape[1]
    print('seq_len', seq_len, 'n_classes', n_classes, 'n_channels', n_channels, 'class_names', class_names)
    return gesture_all_1D_padded, np.array(labels), n_channels, n_classes, seq_len, class_names



def standardize(train, test):
    """ Standardize data """

    # Standardize train and test
    X_train = (train - np.mean(train, axis=0)[None,:,:]) / np.std(train, axis=0)[None,:,:]
    X_test = (test - np.mean(test, axis=0)[None,:,:]) / np.std(test, axis=0)[None,:,:]

    return X_train, X_test

def one_hot(labels, n_class):
    """ One-hot encoding """
    expansion = np.eye(n_class)
    y = expansion[:, labels-1].T
    assert y.shape[1] == n_class, "Wrong number of labels!"
    return y

def get_batches(X, y, batch_size ):
    """ Return a generator for batches """
    n_batches = len(X) // batch_size
    X, y = X[:n_batches*batch_size], y[:n_batches*batch_size]

    # Loop over batches and yield
    for b in range(0, len(X), batch_size):
        yield X[b:b+batch_size], y[b:b+batch_size]







	#

import tensorflow as tf
import numpy as np

import os
from utilities import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
# Imports
import tensorflow as tf


data_path = '/Users/babalia/Desktop/final_project/database/CAD_60/txt_gesture'
data, labels, n_channels, n_classes, seq_len = read_data(data_path)
# 这里我们得到了rnn_input （batch, time_length, joints）
# (68, 1960, 18)

#x为数据集的feature熟悉，y为label.
X_train, X_test, labels_train, labels_test = train_test_split(data, labels, test_size = 0.25)

print ("Training data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_train.shape[0],
                                                                             X_train.shape[1],
                                                                             X_train.shape[2]))
print ("Test data shape: N = {:d}, steps = {:d}, channels = {:d}".format(X_test.shape[0],
                                                                         X_test.shape[1],
                                                                         X_test.shape[2]))
y_train = one_hot(labels_train,n_classes)
y_test = one_hot(labels_test,n_classes)
print('y', y_train.shape, y_test.shape)

# #
# X_tr, X_vld, lab_tr, lab_vld = train_test_split(X_train, labels_train, stratify = labels_train, random_state = 123)
#
# # one_hot encoding
# y_tr = one_hot(lab_tr,n_classes)
# y_vld = one_hot(lab_vld, n_classes)
# y_test = one_hot(labels_test, n_classes)
# print('train_x',X_tr.shape, 'y_tr', y_tr.shape)

learning_rate = 0.0001
max_samples = 3000
display_size = 10
batch_size = 17

#实际上图的像素列数，每一行作为一个输入，输入到网络中。
n_input = n_channels
#LSTM cell的展开宽度，对于图像来说，也是图像的行数
#也就是图像按时间步展开是按照行来展开的。
n_step = seq_len
#LSTM cell个数   越多的hidden_size可以包含更多的细节，以及更丰富的表达能力，但是同时也会带来过拟合以及耗时间等缺点。
n_hidden = 256
n_class = n_classes


x = tf.placeholder(tf.float32, shape=[None, n_step, n_input])
y = tf.placeholder(tf.float32, shape =[None, n_class])

#这里的参数只是最后的全连接层的参数，调用BasicLSTMCell这个op，参数已经包在内部了，不需要再定义。
Weight = tf.Variable(tf.random_normal([2 * n_hidden, n_class]))   #参数共享力度比cnn还大
bias = tf.Variable(tf.random_normal([n_class]))


def BiRNN(x, weights, biases):
    #[1, 0, 2]只做第阶和第二阶的转置
    x = tf.transpose(x, [1, 0, 2])
    #把转置后的矩阵reshape成n_input列，行数不固定的矩阵。
    #对一个batch的数据来说，实际上有bacth_size*n_step行。
    x = tf.reshape(x, [-1, n_input])  #-1,表示样本数量不固定
    #拆分成n_step组
    x = tf.split(x, n_step)
    #调用现成的BasicLSTMCell，建立两条完全一样，又独立的LSTM结构
    lstm_qx = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    lstm_hx = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias = 1.0)
    #两个完全一样的LSTM结构输入到static_bidrectional_rnn中，由这个op来管理双向计算过程。
    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_qx, lstm_hx, x, dtype = tf.float32)
    #最后来一个全连接层分类预测
    return tf.matmul(outputs[-1], weights) + biases


pred = BiRNN(x, Weight, bias)
#计算损失、优化、精度（老套路）
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accurancy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

#run图过程。
with tf.Session() as sess:
    sess.run(init)
    step = 1
    train_loss_list = [0]
    train_acc_list = [0]
    val_loss_list = [0]
    val_acc_list = [0]
    while step * batch_size < max_samples:
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = X_train
        batch_y = y_train
        # batch_x = batch_x.reshape((batch_size, n_step, n_input))
        # print('batch_x',batch_x.shape,'batch_y',batch_y.shape)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        batch_x_v = X_test
        batch_y_v = y_test
        batch_x_v = batch_x_v.reshape((batch_size, n_step, n_input))
        sess.run(optimizer, feed_dict={x: batch_x_v, y: batch_y_v})

        acc = sess.run(accurancy, feed_dict={x: batch_x, y: batch_y})
        loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        acc_v = sess.run(accurancy, feed_dict={x: batch_x_v, y: batch_y_v})
        loss_v = sess.run(cost, feed_dict={x: batch_x_v, y: batch_y_v})
        train_loss_list.append(loss)
        train_acc_list.append(acc)
        val_loss_list.append(loss_v)
        val_acc_list.append(acc_v)
        if step % display_size == 0:

            print('Iter' + str(step * batch_size) + ',Train Minibatch Loss= %.6f' % (loss) + ', Train Accurancy= %.5f' % (acc))

            print('Iter' + str(step*batch_size) + ', Test Minibatch Loss= %.6f'%(loss_v) + ', Test Accurancy= %.5f'%(acc_v))


        step += 1
    print("Optimizer Finished!")

    fig_loss = plt.figure('loss')
    plt.plot(np.array(train_loss_list))
    plt.plot(np.array(val_loss_list))
    plt.title('Model loss by epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='right')

    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    fig_train = plt.figure('train')
    plt.plot(np.array(train_acc_list))
    plt.plot(np.array(val_acc_list))
    plt.title('Model accuracy by epoch')
    plt.ylabel('accuracy')
    plt.legend(['train', 'valid'], loc='right')
    plt.xlabel('epoch')
    plt.show()



    # plt.figure(figsize=(6, 6))
    # plt.plot(step * batch_size, np.array(loss), 'r-', step * batch_size, np.array(loss_v), 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Loss")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    #
    # # Plot Accuracies
    # plt.figure(figsize=(6, 6))
    #
    # plt.plot(step * batch_size, np.array(acc), 'r-', step * batch_size, np.array(acc_v), 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Accuray")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()


    # test_len = 10000
    # test_data = mnist.test.images[:test_len].reshape(-1, n_step, n_input)
    # test_label = mnist.test.labels[:test_len]
    test_data = X_test.reshape(-1, n_step, n_input)
    test_label = y_test
    print('Testing Accurancy:%.5f'%(sess.run(accurancy, feed_dict={x: test_data, y:test_label})))

    Coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=Coord)

    #
    # # Plot training and test loss
    # # t = np.arange(iteration-1)
    #
    # plt.figure(figsize = (6,6))
    # plt.plot(t, np.array(train_loss), step*batch_size, 'r-', , np.array(validation_loss), 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Loss")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()
    #
    # # Plot Accuracies
    # plt.figure(figsize = (6,6))
    #
    # plt.plot(t, np.array(train_acc), 'r-', t[t % 25 == 0], validation_acc, 'b*')
    # plt.xlabel("iteration")
    # plt.ylabel("Accuray")
    # plt.legend(['train', 'validation'], loc='upper right')
    # plt.show()

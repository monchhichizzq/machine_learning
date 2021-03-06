from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from keras import layers
from keras import models
from keras import optimizers
from keras import regularizers
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
from keras.regularizers import l2
from keras.layers.core import Dense, Dropout, Activation, Flatten
from sklearn.model_selection import StratifiedKFold
import cv2
import keras
import os
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import spline
from PIL import Image
# 引入Tensorboard
from keras.callbacks import TensorBoard

#

#
def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
#
# #读取图片
def load_data(data_path, img_rows, img_cols):
    class_names = []
    data = []
    labels = []
    label_index = 0
    data_new = []
    label_new = []
    for label_name in os.listdir(data_path):
        if os.path.splitext(label_name)[0] != 'Thumbs':
            class_names.append(label_name)
            for image_name in os.listdir(data_path + '/' + label_name):
                image = cv2.imread(data_path + '/' + label_name +'/' + image_name)
                if image is not None:
                    # print('image',image.shape)
                    image = cv2.resize(image, (img_rows, img_cols))
                    data.append(image)
                    labels.append(label_index)
            label_index = label_index + 1
    length_data = range(len(data))
    labels = keras.utils.to_categorical(labels, len(class_names))
    print('class_number:', len(class_names), 'class names:', str(class_names))
    data = np.array(data)
    data = data.astype('float32')
    return data, labels, len(class_names), class_names

img_rows = 227
img_cols = 227
# hyperparameter
keep_prob = 0.4
learning_rate = 1e-4
batch_size = 64
epoch = 200
r = 1e-4
K = 5

log_path = 'semi-alexnet-{}'.format(learning_rate)
model_path = 'model/my_mode_semi_alexnet.h5'
data_path = "C:\\Users\\10289005\\OneDrive - BD\\Desktop\\image_classification\\RollingBeadModel"
path_image = 'C:\\Users\\10289005\\OneDrive - BD\\Desktop\\image_classification\\image_save\\sq\\sq_alexnet_K'
data, labels, length, class_names = load_data(data_path, img_rows, img_cols)
# x为数据集的feature熟悉，y为label.

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# data normalization
x_train_rows = X_train.reshape(X_train.shape[0], img_rows * img_cols * 3)
x_test_rows = X_test.reshape(X_test.shape[0], img_rows * img_cols * 3)

minmax = MinMaxScaler()

x_train_rows = minmax.fit_transform(x_train_rows)
x_test_rows = minmax.fit_transform(x_test_rows)
# print('minmax',x_train_rows, x_test_rows)

# convert to 32 x 32 x 3
x_train = x_train_rows.reshape(x_train_rows.shape[0], img_rows, img_cols, 3)
x_test = x_test_rows.reshape(x_test_rows.shape[0], img_rows, img_cols, 3)
# print('x_train',x_train.shape, 'x_test',x_test.shape)

# y_train = labels_train
# y_test = labels_test
# print('y_train',y_train.shape,'y_test',y_test.shape)

# train, val, test data split
train_ratio = 0.8
x_train_, x_val, y_train_, y_val = train_test_split(x_train, y_train,train_size=train_ratio,random_state=123)

print('x_train_',np.array(x_train_).shape, 'y_train_',np.array(y_train_).shape)
print('x_val',np.array(x_val).shape, 'x_val',np.array(y_val).shape)
print('x_test',np.array(x_test).shape,'y_test',np.array(y_test).shape)



# length = 10

# fix random seed for reproducibility
seed = 7
# # define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=K, shuffle=True, random_state=seed)
cvscores = []
y = labels.argmax(1)
for train, test in kfold.split(data, y):
    # Listing 5.1. Instantiating a small convnet AlexNet
    # Instantiate an empty model
    model = models.Sequential()

    # 1st Convolutional Layer et maxpooling
    # 1st Convolutional Layer et maxpooling
    model.add(
    layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', input_shape=(227, 227, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size = (3, 3), strides = (2,2), padding = 'valid'))

    # 2nd Convolutional layer et maxpooling
    model.add(layers.Conv2D(filters = 256, kernel_size = (5,5), strides =(1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size = (3, 3), strides = (2,2), padding = 'valid' ))

    # 3rd Convolutional layer
    model.add(layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 4th Convolutional layer
    model.add(layers.Conv2D(filters = 384, kernel_size = (3, 3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 5th Convolutional layer et maxpooling
    model.add(layers.Conv2D(filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size = (3,3), strides = (2,2), padding = 'valid'))

    # passsing it to a Fully connected layer
    model.add(layers.Flatten())
    # 1st fully connected layer
    # model.add(layers.Dense(4096, activation = 'relu', kernel_regularizer=regularizers.l2(0.001)))
    model.add(layers.Dense(4096, activation = 'relu',kernel_regularizer=regularizers.l2(0.01)))
    # keras.layers.core.Dropout(rate, noise_shape=None, seed=None)
    model.add(layers.Dropout(1-keep_prob))
    # 2rd fully connected layer
    model.add(layers.Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(1-keep_prob))
    # 3rd fully connected layer
    model.add(layers.Dense(1000, activation = 'relu'))
    # dropout rate=0.8时，实际上，保留概率为0.2 rate = 1-keep_prob：0~1的浮点数，控制需要断开的神经元的比例
    model.add(layers.Dropout(1-keep_prob))

    # Output Layer
    model.add(layers.Dense(length, activation='softmax'))

    # Display the architecture of the convnet
    model.summary()
    #
    # model.compile(optimizer= optimizers.sgd(lr=learning_rate, momentum=0.9, decay=0.0005),
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])
    model.compile(optimizer= optimizers.adam(lr=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
    # # del model  # deletes the existing model
    # tensorborad
    tbCallBack = TensorBoard(log_dir='logs/{}'.format(log_path),  # log 目录
                             histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                             #                  batch_size=32,     # 用多大量的数据计算直方图
                             write_graph=True,  # 是否存储网络结构图
                             write_grads=True,  # 是否可视化梯度直方图
                             write_images=True,  # 是否可视化参数
                             embeddings_freq=0,
                             embeddings_layer_names=None,
                             embeddings_metadata=None)


    history = model.fit(data[train], labels[train], epochs=epoch, batch_size=batch_size, callbacks=[tbCallBack])
    # evaluate the model
    scores = model.evaluate(data[test], labels[test], verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
fig_acc = plt.gcf()
path_acc = path_image + 'acc.png'
mkdir(path_image)
fig_acc.savefig(path_acc,dpi=600)

plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
fig_loss = plt.gcf()
path_loss = path_image + 'loss.png'
fig_loss.savefig(path_loss,dpi=600)
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc',test_acc,'test_loss',test_loss)

# save
print('test before save: ', model.predict(x_test[0:2]))
model.save(model_path)   # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model

# load
model = load_model(model_path)
print('test after load: ', model.predict(x_test[0:2]))
test_loss, test_acc = model.evaluate(x_test, y_test)

# Results
predictions = model.predict(x_test).argmax(1)
y_test = y_test.argmax(1)
print('predictions', predictions.shape)

print("Testing Accuracy: {}%".format(100*test_acc))

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
height =7
plt.figure(figsize=(width, height))
plt.imshow(
    normalised_confusion_matrix,
    interpolation='nearest',
    cmap=plt.cm.rainbow
)
plt.title("Confusion matrix \n(normalised to % of total test data)")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=90)
plt.yticks(tick_marks, class_names)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = metrics.confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plot_confusion_matrix(y_test, predictions, classes=class_names, title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, predictions, classes=class_names, normalize=True, title='Normalized confusion matrix')
fig_cm = plt.gcf()
path_cm = path_image + 'cm.png'
fig_cm.savefig(path_cm,dpi=600)
plt.show()
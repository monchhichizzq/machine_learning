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

#
#
# #读取图片
def load_data(data_path,img_rows,img_cols):
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
                image = cv2.resize(image, (img_rows, img_cols))
                data.append(image)
                labels.append(label_index)
        label_index = label_index + 1
    print('labels', np.array(labels).shape)
    print('label_index',label_index)
    length_data = range(len(data))
    labels = keras.utils.to_categorical(labels, len(class_names))
    print('class_number:', len(class_names), 'class names:', str(class_names))
    data = np.array(data)
    data = data.astype('float32')
    return data, labels, len(class_names), class_names

img_rows = 60
img_cols = 60
# data_path = "/Users/babalia/Desktop/final_project/data_processing/Gesture_data/CNN_A"
data_path = '/Users/babalia/Desktop/final_project/data/action_data/CNN'
data, labels, length, class_names = load_data(data_path, img_rows, img_cols)

# x为数据集的feature熟悉，y为label.
X_train, X_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)

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

y_train = labels_train
y_test = labels_test
# print('y_train',y_train.shape,'y_test',y_test.shape)

# train, val, test data split
train_ratio = 0.8
x_train_, x_val, y_train_, y_val = train_test_split(x_train,y_train,train_size=train_ratio,random_state=123)

print('x_train_',np.array(x_train_).shape, 'y_train_',np.array(y_train_).shape)
print('x_val',np.array(x_val).shape, 'x_val',np.array(y_val).shape)
print('x_test',np.array(x_test).shape,'y_test',np.array(y_test).shape)



# hyperparameter
keep_prob = 0.4
# length = 10

#
# Listing 5.1. Instantiating a small convnet AlexNet
# Instantiate an empty model
model = models.Sequential()

# 1st Convolutional Layer et maxpooling
model.add(layers.Conv2D(filters = 96, kernel_size = (6, 6), strides = (1,1), padding = 'valid', input_shape=(60, 60, 3)))
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
model.add(layers.Dense(4096, activation = 'relu'))
model.add(layers.Dropout(1-keep_prob))
# 2rd fully connected layer
model.add(layers.Dense(4096, activation = 'relu'))
model.add(layers.Dropout(1-keep_prob))
# 3rd fully connected layer
model.add(layers.Dense(1000, activation = 'relu'))
# dropout rate=0.8时，实际上，保留概率为0.2
model.add(layers.Dropout(1-keep_prob))

# Output Layer
model.add(layers.Dense(length, activation='softmax'))

# Display the architecture of the convnet
model.summary()
#
model.compile(optimizer= optimizers.adam(lr=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
# # del model  # deletes the existing model


history = model.fit(x_train_, y_train_, epochs=100, batch_size=64, validation_data=(x_val, y_val))

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

plt.figure()

plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test)
print('test_acc',test_acc,'test_loss',test_loss)

# save
print('test before save: ', model.predict(x_test[0:2]))
model.save('model/my_model_1.h5')   # HDF5 file, you have to pip3 install h5py if don't have it
del model  # deletes the existing model

# load
model = load_model('model/my_model_1.h5')
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
width = 8
height =8
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


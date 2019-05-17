import gc
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from utilities import *
from sklearn import metrics
from sklearn import model_selection


# import warnings
# warnings.filterwarnings("ignore")
# print("TensorFlow Version: %s" % tf.__version__)



def rnn_input():
  data_path = 'C:\\dataset\\txt_aug_clean'
  data, labels, n_channels, n_classes, seq_len, class_names = read_data(data_path)
  print('Data', data.shape, 'labels', labels.shape)
# 这里我们得到了rnn_input （batch, time_length, joints）
  # (68, 1960, 18)
  labels = keras.utils.to_categorical(labels, len(class_names))
  print('class_number:', len(class_names), 'class names:', str(class_names))

  # x为数据集的feature熟悉，y为label.
  x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)

  print("Training data shape: N = {:d}, steps = {:d}, channels = {:d}".format(x_train.shape[0],
                                                                              x_train.shape[1],
                                                                              x_train.shape[2]))
  print("Test data shape: N = {:d}, steps = {:d}, channels = {:d}".format(x_test.shape[0],
                                                                          x_test.shape[1],
                                                                          x_test.shape[2]))
  return data, labels, n_channels, n_classes, seq_len, x_train, x_test, y_train, y_test, class_names


data, labels, n_channels, n_classes, seq_len, x_train, x_test, y_train, y_test, class_names = rnn_input()
print('x_tr', x_train.shape, 'y_tr', y_train.shape)
# print('x_val', x_val.shape, 'x_val', y_val.shape)
print('x_test', x_test.shape, 'y_test', y_test.shape)
#




# Input Data
training_data_count = len(x_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(x_test)  # 2947 testing series
n_steps = len(x_train[0])  # 128 timesteps per series
n_input = len(x_train[0][0])  # 9 input parameters per timestep

# LSTM Neural Network's internal structure

n_hidden = 64  # Hidden layer num of features
# n_classes = 6 # Total classes (should go up, or should go down)

lr = 1e-5
joints = x_train.shape[2]
window_len = 7
activation_function = 'relu'
# loss = 'categorical_crossentropy'
loss = 'categorical_crossentropy'
optimizer="adam"
dropout = 0.6
batch_size = 64
epochs = 1000
merge_date = '2016-01-01'

def build_model(inputs, output_size, neurons, activ_func=activation_function
, dropout=dropout, loss=loss, optimizer=optimizer, learning_rate = lr):
  """
  inputs: input data as numpy array
  output_size: number of predictions per input sample
  neurons: number of neurons/ units in the LSTM layer
  active_func: Activation function to be used in LSTM layers and Dense layer
  dropout: dropout ration, default is 0.25
  loss: loss function for calculating the gradient
  optimizer: type of optimizer to backpropagate the gradient
  This function will build 3 layered RNN model with LSTM cells with dripouts
  after each LSTM layer
  and finally a dense layer to produce the output using keras' sequential
  model.
  Return: Keras sequential model and model summary
  """
  model = Sequential()
  model.add(LSTM(neurons, return_sequences=True, input_shape=(inputs.shape[1], inputs.shape[2]), activation=activ_func))
  model.add(Dropout(dropout))
  model.add(LSTM(neurons, activation=activ_func))
  model.add(Dropout(dropout))
  # model.add(LSTM(neurons, activation=activ_func))
  # model.add(Dropout(dropout))
  model.add(Dense(units=output_size))
  model.add(Activation(activ_func))
  # model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
  model.compile(optimizer=optimizers.adam(lr=learning_rate),
                loss=loss,
                metrics=['accuracy'])
  model.summary()
  return model

# initialise model architecture
model = build_model(x_train, output_size=1, neurons=joints)
# train model on data
history = model.fit(x_train, y_train, epochs=epochs,
batch_size=batch_size, verbose=1, validation_data=(x_test, y_test),
shuffle=False)

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


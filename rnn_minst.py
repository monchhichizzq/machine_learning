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

import os
import warnings

warnings.filterwarnings("ignore")
print("TensorFlow Version: %s" % tf.__version__)


def rnn_input():
    data_path = '/Users/babalia/Desktop/final_project/data/action_data/txt_aug_clean'
    data, labels, n_channels, n_classes, seq_len, LABELS = read_data(data_path)

    print('Data', data.shape, 'labels', labels.shape)
    # 这里我们得到了rnn_input （batch, time_length, joints）
    # (68, 1960, 18)

    #x为数据集的feature熟悉，y为label.
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.3)

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


# Input Data


training_data_count = len(X_train)  # 7352 training series (with 50% overlap between each serie)
test_data_count = len(X_test)  # 2947 testing series
n_steps = len(X_train[0])  # 128 timesteps per series
n_input = len(X_train[0][0])  # 9 input parameters per timestep


# LSTM Neural Network's internal structure

n_hidden = 64 # Hidden layer num of features
# n_classes = 6 # Total classes (should go up, or should go down)


# Training

learning_rate = 1e-3
lambda_loss_amount = 1e-5
training_iters = training_data_count * 10000  # Loop 300 times on the dataset
print('training_iters',training_iters)
batch_size = 128
display_iter = 100  # To show test set accuracy during training
keep_prob = 0.4

# Some debugging info

print("Some useful info to get an insight on dataset's shape and normalisation:")
print("(X shape, y shape, every X's mean, every X's standard deviation)")
print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")


def LSTM_RNN(_X, _weights, _biases):
    # # **步骤1：RNN 的输入shape = (batch_size, timestep_size, input_size)
    # X = tf.reshape(_X, [-1, 28, 28])
    #
    # # **步骤2：定义一层 LSTM_cell，只需要说明 hidden_size, 它会自动匹配输入的 X 的维度
    # lstm_cell = rnn.BasicLSTMCell(num_units=hidden_size, forget_bias=1.0, state_is_tuple=True)
    #
    # # **步骤3：添加 dropout layer, 一般只设置 output_keep_prob
    # lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
    #
    # # **步骤4：调用 MultiRNNCell 来实现多层 LSTM
    # mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
    #
    # # **步骤5：用全零来初始化state
    # init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)

    # Function returns a tensorflow LSTM (RNN) artificial neural network from given parameters.
    # Moreover, two LSTM cells are stacked which adds deepness to the neural network.
    # Note, some code of this notebook is inspired from an slightly different
    # RNN architecture used on another dataset, some of the credits goes to
    # "aymericdamien" under the MIT license.

    # (NOTE: This step could be greatly optimised by shaping the dataset once
    # input shape: (batch_size, n_steps, n_input)
    _X = tf.transpose(_X, [1, 0, 2])  # permute n_steps and batch_size
    # Reshape to prepare input to hidden activation
    _X = tf.reshape(_X, [-1, n_input])
    # new shape: (n_steps*batch_size, n_input)

    # ReLU activation, thanks to Yu Zhao for adding this improvement here:
    _X = tf.nn.relu(tf.matmul(_X, _weights['hidden']) + _biases['hidden'])
    # Split data because rnn cell needs a list of inputs for the RNN inner loop
    _X = tf.split(_X, n_steps, 0)
    # new shape: n_steps * (batch_size, n_hidden)

    # Define two stacked LSTM cells (two recurrent layers deep) with tensorflow
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_1, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_2, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cell_3 = tf.contrib.rnn.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_3 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_3, input_keep_prob=1.0, output_keep_prob=keep_prob)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1], state_is_tuple=True)
    # Get LSTM cell output
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    # Get last time step's output feature for a "many-to-one" style classifier,
    # as in the image describing RNNs at the top of this page
    lstm_last_output = outputs[-1]

    # Linear activation
    return tf.matmul(lstm_last_output, _weights['out']) + _biases['out']


def extract_batch_size(_train, step, batch_size):
    # Function to fetch a "batch_size" amount of data from "(X|y)_train" data.

    shape = list(_train.shape)
    shape[0] = batch_size
    batch_s = np.empty(shape)

    for i in range(batch_size):
        # Loop index
        index = ((step - 1) * batch_size + i) % len(_train)
        batch_s[i] = _train[index]

    return batch_s


def one_hot(y_, n_classes=n_classes):
    # Function to encode neural one-hot output labels from number indexes
    # e.g.:
    # one_hot(y_=[[5], [0], [3]], n_classes=6):
    #     return [[0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]

    y_ = y_.reshape(len(y_))
    return np.eye(n_classes)[np.array(y_, dtype=np.int32)]  # Returns FLOATS

# Graph input/output
x = tf.placeholder(tf.float32, [None, n_steps, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])

# Graph weights
weights = {
    'hidden': tf.Variable(tf.random_normal([n_input, n_hidden])), # Hidden layer weights
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=1.0))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([n_hidden])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = LSTM_RNN(x, weights, biases)

# Loss, optimizer and evaluation
l2 = lambda_loss_amount * sum(
    tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables()
) # L2 loss prevents this overkill neural network to overfit the data
print('l2',l2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))  # Softmax loss + l2
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) # Adam Optimizer

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# To keep track of training's performance
test_losses = []
test_accuracies = []
train_losses = []
train_accuracies = []
train_losses_1batch = []
train_accuracies_1batch = []

# Launch the graph
sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
init = tf.global_variables_initializer()
sess.run(init)

# Perform Training steps with "batch_size" amount of example data at each loop
step = 1
while step * batch_size <= training_iters:
    batch_xs = extract_batch_size(X_train, step, batch_size)
    batch_ys = one_hot(extract_batch_size(y_train, step, batch_size))

    # Fit training using batch data
    _, loss, acc = sess.run(
        [optimizer, cost, accuracy],
        feed_dict={
            x: batch_xs,
            y: batch_ys
        }
    )
    train_losses_1batch.append(loss)
    train_accuracies_1batch.append(acc)

    # Evaluate network only at some steps for faster training:
    if (step * batch_size % display_iter == 0) or (step == 1) or (step * batch_size > training_iters):
        # To not spam console, show training accuracy/loss in this "if"
        # print("Training iter #" + str(step * batch_size) + \
        #       ":   Batch Loss = " + "{:.6f}".format(loss) + \
        #       ", Accuracy = {}".format(acc))
        print("Batch #" + str(step*batch_size / display_iter) + \
              ":   Batch Loss = " + "{:.6f}".format(loss) + \
              ", Accuracy = {}".format(acc))
        train_losses.append(loss)
        train_accuracies.append(acc)

        # Evaluation on the test set (no learning made here - just evaluation for diagnosis)
        loss, acc = sess.run(
            [cost, accuracy],
            feed_dict={
                x: X_test,
                y: one_hot(y_test)
            }
        )
        test_losses.append(loss)
        test_accuracies.append(acc)
        print("PERFORMANCE ON TEST SET: " + \
              "Batch Loss = {}".format(loss) + \
              ", Accuracy = {}".format(acc))

    step += 1

print("Optimization Finished!")

# Accuracy for test data
one_hot_predictions_train, accuracy_train, train_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_train,
        y: one_hot(y_train)
    }
)


one_hot_predictions, accuracy, final_loss = sess.run(
    [pred, accuracy, cost],
    feed_dict={
        x: X_test,
        y: one_hot(y_test)
    }
)

# test_losses.append(final_loss)
# test_accuracies.append(accuracy)

print("FINAL Train RESULT: " + \
      "Batch Loss = {}".format(train_loss) + \
      ", Accuracy = {}".format(accuracy_train))

print("FINAL RESULT: " + \
      "Batch Loss = {}".format(final_loss) + \
      ", Accuracy = {}".format(accuracy))

# plot
# (Inline plots: )
# %matplotlib inline
# plot accuracy
font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = 12
height = 12
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_accuracies), "b-", label="Train accuracies")

indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_train_axis, np.array(test_accuracies), "g-", label="Test accuracies")

plt.title("Training accuracy session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Accuracy values)')
plt.xlabel('Training iteration')

plt.show()


# plot loss
font = {
    'family' : 'Bitstream Vera Sans',
    'weight' : 'bold',
    'size'   : 18
}
matplotlib.rc('font', **font)

width = n_classes
height = n_classes
plt.figure(figsize=(width, height))

indep_train_axis = np.array(range(batch_size, (len(train_losses)+1)*batch_size, batch_size))
plt.plot(indep_train_axis, np.array(train_losses),     "b-", label="Train losses")


indep_test_axis = np.append(
    np.array(range(batch_size, len(test_losses)*display_iter, display_iter)[:-1]),
    [training_iters]
)
plt.plot(indep_train_axis, np.array(test_losses),     "g-", label="Test losses")


plt.title("Training loss session's progress over iterations")
plt.legend(loc='upper right', shadow=True)
plt.ylabel('Training Progress (Loss values)')
plt.xlabel('Training iteration')
plt.show()

# Results

predictions = one_hot_predictions.argmax(1)
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


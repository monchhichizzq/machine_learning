import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import warnings
warnings.filterwarnings('ignore')
# import keras.backend as K
# dtype='float16'
# K.set_floatx(dtype)
# # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
# K.set_epsilon(1e-4)
from CNN_architectures import models
import time
import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from keras import optimizers
from keras.utils import multi_gpu_model
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn import metrics
from CNN_architectures import data_generator_reader
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau


def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
def get_classes(data_path):
    class_names = []
    for label_name in os.listdir(data_path):
        if os.path.splitext(label_name)[0] != 'Thumbs':
            class_names.append(label_name)
    print('class_number:', len(class_names), 'class names:', str(class_names))
    return len(class_names), class_names

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 50:
        lr *= 1e-5
    elif epoch > 20:
        lr *= 1e-4
    elif epoch > 10:
        lr *= 1e-3
    print('Learning rate: ', lr)
    return lr


time_start = time.time()

img_rows = 224
img_cols = 224

gpu = 4
# hyperparameter
keep_prob = 0.5
# depth = 56
batch_size = 8
learning_rate = 1e-5
# epoch 150(0.90)
epoch = 100
r = 1e-3
decay =1e-4
momentum = 0.9

# optimzier
# adam = optimizers.adam(lr=lr_schedule(0))
# # sgd = optimizers.SGD(lr=learning_rate, decay=decay, momentum=momentum, nesterov=True)
# optimizers = adam



model_name = 'VGG_0_12_24'
log_path = 'semi-resnet-{}'.format(model_name)
model_path = 'model/'+model_name+'_800.h5'
data_path = 'D:\\data\\Time_0_12_24_training\\TOP_BLACK\\train'
data_train = 'D:\\data\\Time_0_12_24_training\\TOP_BLACK\\train'
data_test = 'D:\\data\\Time_0_12_24_training\\TOP_BLACK\\test'
data_val = 'D:\\data\\Time_0_12_24_training\\TOP_BLACK\\val'
path_image = 'D:\\AWS_test\\image_save\\9_channels\\'+ model_name + '_' + str(epoch)+'_drop'+str(keep_prob)+'_batch'+str(batch_size)
mkdir(path_image)

nb_train_samples = 4938
nb_validation_samples = 533
nb_test_samples = 603

length, class_names = get_classes(data_path)

train_generator = data_generator_reader.batch_generator_0_12_24(data_train, batch_size, img_cols, img_rows, length, shuffle=True, seperation=True)
test_generator =  data_generator_reader.batch_generator_0_12_24(data_test, batch_size, img_cols, img_rows, length, shuffle=False, seperation=True)
val_generator = data_generator_reader.batch_generator_0_12_24(data_val, batch_size, img_cols, img_rows, length, shuffle=False, seperation=True)



# Prepare model model saving directory.
save_dir = os.path.join('E:\models', 'resnet_top_saved_models')
print(save_dir)
model_name_save = 'CHROM-ORI_%s_model.{epoch:03d}.h5' % model_name
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name_save)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]
# callbacks = [checkpoint]

# model = models.vgg16_transfer(input_shape=(img_rows, img_cols, 3), keep_prob=keep_prob, classes=length, r=r)
# model = models.resnet_v2(input_shape=(img_rows, img_cols, 3), depth=depth, num_classes = length)
model = models.vgg16_0_12_24(input_shape=(img_rows, img_cols, 3), keep_prob=keep_prob, classes=length, r=r)
# model = multi_gpu_model(model, gpus=gpu)
model.compile(loss='categorical_crossentropy', optimizer=optimizers.adam(lr=learning_rate), metrics=['accuracy'])
history = model.fit_generator(
            train_generator,
            steps_per_epoch= nb_train_samples // batch_size,
            epochs=epoch,
            validation_data=val_generator,
            validation_steps= nb_validation_samples // batch_size,
            callbacks = callbacks)
model.save(model_path)

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
path_acc = path_image + '/acc.png'
mkdir(path_image)
fig_acc.savefig(path_acc,dpi=600)

plt.figure()
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
fig_loss = plt.gcf()
path_loss = path_image + '/loss.png'
fig_loss.savefig(path_loss,dpi=600)
plt.show()

test_loss, test_acc = model.evaluate_generator(test_generator, steps=round(nb_test_samples//batch_size)+1)
print('test_acc',test_acc,'test_loss',test_loss)

# save
model.save(model_path)   # HDF5 file, you have to pip3 install h5py if don't have it
# del model  # deletes the existing model

# load
# model = load_model(model_path)
# print('test after load: ', model.predict(x_test[0:2]))
# test_loss, test_acc = model.evaluate(x_test, y_test)



# Results

y_pred= model.predict_generator(test_generator, steps=round(nb_test_samples//batch_size)+1)
predictions = np.argmax(y_pred, axis=1)
y_test = test_generator.classes
print('predictions', predictions.shape)

print("Testing Accuracy: {}%".format(100*test_acc))

print("")
print("Precision: {}%".format(100*metrics.precision_score(y_test, predictions, average="weighted")))
print("Recall: {}%".format(100*metrics.recall_score(y_test, predictions, average="weighted")))
print("f1_score: {}%".format(100*metrics.f1_score(y_test, predictions, average="weighted")))





def plot_confusion_matrix_medical(y_true, y_pred, classes,
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
    cm = metrics.confusion_matrix(y_pred, y_true)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Predicted label',
           xlabel='True label')

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
plot_confusion_matrix_medical(y_test, predictions, classes=class_names, title='Confusion matrix, without normalization')
fig_cm = plt.gcf()
path_cm = path_image + '/cm.png'
fig_cm.savefig(path_cm, dpi=600)
plt.show()


# Plot normalized confusion matrix
plot_confusion_matrix_medical(y_test, predictions, classes=class_names, normalize=True, title='Normalized confusion matrix')
fig_cm_normal = plt.gcf()
path_cm_normal = path_image + '/cm_normal.png'
fig_cm_normal.savefig(path_cm_normal, dpi=600)
plt.show()

import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from keras.utils import to_categorical
import os
import cv2
import re
from PIL import Image

from sklearn import metrics

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:  # whether the folder exists
        os.makedirs(path)  # makedirs


def batch_generator_test(data_path, batch_size, img_cols, img_rows, interpolation, length, shuffle=True, seperation = True, test=False):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    image_folder_names = []
    image_folder_labels = []
    # for seperation in os.listdir(data_path):
    #     if seperation is 'train':
    #         shuffle = True
    path_split= data_path
    label = 0
    labels = []
    for label_name in os.listdir(path_split):
        label_path = path_split + '/' +label_name
        image_folder_name = os.listdir(label_path)
        number = len(image_folder_name)
        labels = np.ones(number)*label
        image_folder_names.extend(image_folder_name)
        image_folder_labels.extend(labels)
        label += 1
    # print(len(image_folder_names))
    # print(len(image_folder_labels))
    image_folder_names_np = np.array(image_folder_names).reshape(np.array(image_folder_names).shape[0], 1)
    image_folder_labels_np = np.array(image_folder_labels).reshape(np.array(image_folder_labels).shape[0], 1)
    data_name_with_labels = np.concatenate((image_folder_names_np, image_folder_labels_np), axis=1)
    # print(data_name_with_labels.shape)
    if shuffle is True:
        sample = random.sample(image_folder_names, len(image_folder_names))
        # print(image_folder_names)
        # print(sample)
    else:
        sample = image_folder_names
        # print(sample)
    new_labels = []
    for image in sample:
        image_label = image_folder_labels[image_folder_names.index(image)]
        new_labels.append(image_label)
    # print(new_labels)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(sample):
            start = batch_count * batch_size
            delta = len(sample) - batch_count * batch_size
            end = start +delta
            input_batch_name = sample[start:end]
            label_batch = new_labels[start:end]
            input_batch_bottom = []
            input_batch_top = []
            for i in range(delta):
                path_new = 'D:\\test_TOP_BOTTOM_Training\\data_24h_all_1600'
                image_path = path_new + '/' + input_batch_name[i]
                for image_name in os.listdir(image_path):
                    if 'BOTTOM' in image_name:
                        image_BOTTOM_name = image_path + '/' + image_name
                        image_BOTTOM = Image.open(image_BOTTOM_name)
                        image_BOTTOM = image_BOTTOM.resize((img_rows, img_cols), resample=Image.BICUBIC)
                        image_BOTTOM = np.multiply(image_BOTTOM, 1.0 / 255.0)
                        input_batch_bottom.append(image_BOTTOM)
                    if 'TOP' in image_name:
                        image_TOP_name = image_path + '/' + image_name
                        image_TOP = Image.open(image_TOP_name)
                        image_TOP = image_TOP.resize((img_rows, img_cols), resample=Image.BICUBIC)
                        image_TOP = np.multiply(image_TOP, 1.0 / 255.0)
                        input_batch_top.append(image_TOP)
            label_batch = to_categorical(label_batch, length)
            yield [np.array(input_batch_bottom), np.array(input_batch_top)], np.array(label_batch)
            if test ==True:
                break
        start = batch_count * batch_size
        end = start + batch_size
        input_batch_name = sample[start:end]
        label_batch = new_labels[start:end]
        input_batch = []
        input_batch_bottom = []
        input_batch_top =[]
        for i in range(batch_size):
            path_new = 'D:\data\TOP-BOTTOM-1folder\data_24h_all_1600'
            image_path= path_new + '/' + input_batch_name[i]
            for image_name in os.listdir(image_path):
                if 'BOTTOM' in image_name :
                    image_BOTTOM_name= image_path + '/' +image_name
                    image_BOTTOM = Image.open(image_BOTTOM_name)
                    image_BOTTOM = image_BOTTOM.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_BOTTOM = np.multiply(image_BOTTOM, 1.0 / 255.0)
                    input_batch_bottom.append(image_BOTTOM)
                if 'TOP' in image_name:
                    image_TOP_name = image_path + '/' + image_name
                    image_TOP = Image.open(image_TOP_name)
                    image_TOP = image_TOP.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_TOP = np.multiply(image_TOP, 1.0 / 255.0)
                    input_batch_top.append(image_TOP)
        label_batch = to_categorical(label_batch, length)
        batch_count += 1
        # print(np.array(input_batch).shape, np.array(label_batch).shape)
        # print(np.array(input_batch_bottom) .shape, np.array(input_batch_top).shape, np.array(label_batch).shape)
        if seperation == True:
            yield [np.array(input_batch_bottom), np.array(input_batch_top)], np.array(label_batch)
        else:
            # print(np.array(input_batch).shape, np.array(label_batch).shape)
            yield np.array(input_batch), np.array(label_batch)


def batch_generator_wrong(data_path, batch_size, img_cols, img_rows, interpolation, length, shuffle=True, seperation = True, test=False):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    image_folder_names = []
    image_folder_labels = []
    # for seperation in os.listdir(data_path):
    #     if seperation is 'train':
    #         shuffle = True
    path_split= data_path
    label = 0
    labels = []
    for label_name in os.listdir(path_split):
        label_path = path_split + '/' +label_name
        image_folder_name = os.listdir(label_path)
        number = len(image_folder_name)
        labels = np.ones(number)*label
        image_folder_names.extend(image_folder_name)
        image_folder_labels.extend(labels)
        label += 1
    # print(len(image_folder_names))
    # print(len(image_folder_labels))
    image_folder_names_np = np.array(image_folder_names).reshape(np.array(image_folder_names).shape[0], 1)
    image_folder_labels_np = np.array(image_folder_labels).reshape(np.array(image_folder_labels).shape[0], 1)
    data_name_with_labels = np.concatenate((image_folder_names_np, image_folder_labels_np), axis=1)
    # print(data_name_with_labels.shape)
    if shuffle is True:
        sample = random.sample(image_folder_names, len(image_folder_names))
        # print(image_folder_names)
        # print(sample)
    else:
        sample = image_folder_names
        # print(sample)
    new_labels = []
    for image in sample:
        image_label = image_folder_labels[image_folder_names.index(image)]
        new_labels.append(image_label)
    # print(new_labels)

    batch_count = 0
    while True:
        # if batch_count * batch_size + batch_size > len(sample):
        #     start = batch_count * batch_size
        #     delta = len(sample) - batch_count * batch_size
        #     end = start +delta
        #     input_batch_name = sample[start:end]
        #     label_batch = new_labels[start:end]
        #     input_batch_bottom = []
        #     input_batch_top = []
        #     for i in range(delta):
        #         path_new = 'D:\\test_TOP_BOTTOM_Training\\data_24h_all_1600'
        #         image_path = path_new + '/' + input_batch_name[i]
        #         for image_name in os.listdir(image_path):
        #             if 'BOTTOM' in image_name:
        #                 image_BOTTOM_name = image_path + '/' + image_name
        #                 image_BOTTOM = Image.open(image_BOTTOM_name)
        #                 image_BOTTOM = image_BOTTOM.resize((img_rows, img_cols), resample=Image.BICUBIC)
        #                 image_BOTTOM = np.multiply(image_BOTTOM, 1.0 / 255.0)
        #                 input_batch_bottom.append(image_BOTTOM)
        #             if 'TOP' in image_name:
        #                 image_TOP_name = image_path + '/' + image_name
        #                 image_TOP = Image.open(image_TOP_name)
        #                 image_TOP = image_TOP.resize((img_rows, img_cols), resample=Image.BICUBIC)
        #                 image_TOP = np.multiply(image_TOP, 1.0 / 255.0)
        #                 input_batch_top.append(image_TOP)
        #     label_batch = to_categorical(label_batch, length)
        #     yield [np.array(input_batch_bottom), np.array(input_batch_top)], np.array(label_batch)
        #     if test ==True:
        #         break
        start = batch_count * batch_size
        end = start + batch_size
        input_batch_name = sample[start:end]
        label_batch = new_labels[start:end]
        input_batch = []
        input_batch_bottom = []
        input_batch_top =[]
        for i in range(batch_size):
            path_new = 'D:\\test_TOP_BOTTOM_Training\\data_24h_all_1600'
            print(i, batch_size*batch_count)
            image_path= path_new + '/' + input_batch_name[i]
            for image_name in os.listdir(image_path):
                if 'BOTTOM' in image_name :
                    image_BOTTOM_name= image_path + '/' +image_name
                    image_BOTTOM = Image.open(image_BOTTOM_name)
                    image_BOTTOM = image_BOTTOM.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_BOTTOM = np.multiply(image_BOTTOM, 1.0 / 255.0)
                    input_batch_bottom.append(image_BOTTOM)
                if 'TOP' in image_name:
                    image_TOP_name = image_path + '/' + image_name
                    image_TOP = Image.open(image_TOP_name)
                    image_TOP = image_TOP.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_TOP = np.multiply(image_TOP, 1.0 / 255.0)
                    input_batch_top.append(image_TOP)
        label_batch = to_categorical(label_batch, length)
        batch_count += 1
        # print(np.array(input_batch).shape, np.array(label_batch).shape)
        # print(np.array(input_batch_bottom) .shape, np.array(input_batch_top).shape, np.array(label_batch).shape)
        if seperation == True:
            yield [np.array(input_batch_bottom), np.array(input_batch_top)], np.array(label_batch)
        else:
            # print(np.array(input_batch).shape, np.array(label_batch).shape)
            yield np.array(input_batch), np.array(label_batch)

def batch_generator(data_path, batch_size, img_cols, img_rows, interpolation, length, shuffle=True, seperation = True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    image_folder_names = []
    image_folder_labels = []
    # for seperation in os.listdir(data_path):
    #     if seperation is 'train':
    #         shuffle = True
    path_split= data_path
    label = 0
    labels = []
    for label_name in os.listdir(path_split):
        label_path = path_split + '/' +label_name
        image_folder_name = os.listdir(label_path)
        number = len(image_folder_name)
        labels = np.ones(number)*label
        image_folder_names.extend(image_folder_name)
        image_folder_labels.extend(labels)
        label += 1
    # print(len(image_folder_names))
    # print(len(image_folder_labels))
    image_folder_names_np = np.array(image_folder_names).reshape(np.array(image_folder_names).shape[0], 1)
    image_folder_labels_np = np.array(image_folder_labels).reshape(np.array(image_folder_labels).shape[0], 1)
    data_name_with_labels = np.concatenate((image_folder_names_np, image_folder_labels_np), axis=1)
    # print(data_name_with_labels.shape)
    if shuffle is True:
        sample = random.sample(image_folder_names, len(image_folder_names))
        # print(image_folder_names)
        # print(sample)
    else:
        sample = image_folder_names
        # print(sample)
    new_labels = []
    for image in sample:
        image_label = image_folder_labels[image_folder_names.index(image)]
        new_labels.append(image_label)
    # print(new_labels)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(sample):
            batch_count = 0
            start = batch_count * batch_size
            end = len(sample)
            input_batch_name = sample[start:end]
            label_batch = new_labels[start:end]
        start = batch_count * batch_size
        end = start + batch_size
        input_batch_name = sample[start:end]
        label_batch = new_labels[start:end]
        input_batch = []
        input_batch_bottom = []
        input_batch_top = []
        for i in range(batch_size):
            path_new = 'D:\data\TOP-BOTTOM-1folder\data_24h_all_1600'
            image_path = path_new + '/' + input_batch_name[i]
            for image_name in os.listdir(image_path):
                if 'BOTTOM' in image_name:
                    image_BOTTOM_name = image_path + '/' + image_name
                    image_BOTTOM = Image.open(image_BOTTOM_name)
                    image_BOTTOM = image_BOTTOM.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_BOTTOM = np.multiply(image_BOTTOM, 1.0 / 255.0)
                    input_batch_bottom.append(image_BOTTOM)
                if 'TOP' in image_name:
                    image_TOP_name = image_path + '/' + image_name
                    image_TOP = Image.open(image_TOP_name)
                    image_TOP = image_TOP.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_TOP = np.multiply(image_TOP, 1.0 / 255.0)
                    input_batch_top.append(image_TOP)
        label_batch = to_categorical(label_batch, length)
        batch_count += 1
        # print(np.array(input_batch).shape, np.array(label_batch).shape)
        # print(np.array(input_batch_bottom) .shape, np.array(input_batch_top).shape, np.array(label_batch).shape)
        if seperation == True:
            yield [np.array(input_batch_bottom), np.array(input_batch_top)], np.array(label_batch)
        else:
            # print(np.array(input_batch).shape, np.array(label_batch).shape)
            yield np.array(input_batch), np.array(label_batch)


def batch_generator_0_12_24(data_path, batch_size, img_cols, img_rows, length, shuffle=True, seperation=True):
    """Generate batches of data.

    Given a list of array-like objects, generate batches of a given
    size by yielding a list of array-like objects corresponding to the
    same slice of each input.
    """
    image_folder_names = []
    image_folder_labels = []
    # for seperation in os.listdir(data_path):
    #     if seperation is 'train':
    #         shuffle = True
    path_split= data_path
    label = 0
    labels = []
    for label_name in os.listdir(path_split):
        label_path = path_split + '/' +label_name
        image_folder_name = os.listdir(label_path)
        number = len(image_folder_name)
        labels = np.ones(number)*label
        image_folder_names.extend(image_folder_name)
        image_folder_labels.extend(labels)
        label += 1
    print(len(image_folder_names))
    print(len(image_folder_labels))
    image_folder_names_np = np.array(image_folder_names).reshape(np.array(image_folder_names).shape[0], 1)
    image_folder_labels_np = np.array(image_folder_labels).reshape(np.array(image_folder_labels).shape[0], 1)
    data_name_with_labels = np.concatenate((image_folder_names_np, image_folder_labels_np), axis=1)

    if shuffle is True:
        sample = random.sample(image_folder_names, len(image_folder_names))
        print(image_folder_names)
        print(sample)
    else:
        sample = image_folder_names
        print(sample)
    new_labels = []
    for image in sample:
        image_label = image_folder_labels[image_folder_names.index(image)]
        new_labels.append(image_label)
    # print(new_labels)

    batch_count = 0
    while True:
        if batch_count * batch_size + batch_size > len(sample):
            start = batch_count * batch_size
            end = len(sample)
            input_batch_name = sample[start:end]
            label_batch = new_labels[start:end]
            batch_count = 0
        start = batch_count * batch_size
        end = start + batch_size
        input_batch_name = sample[start:end]
        label_batch = new_labels[start:end]
        input_batch = []
        input_batch_0h = []
        input_batch_12h = []
        input_batch_24h = []
        for i in range(batch_size):
            path_new = 'D:\\data\\Time_0_12_24_training\\TOP_BLACK_all_1600'
            image_path= path_new + '/' + input_batch_name[i]
            # print(image_path)
            # print(os.listdir(image_path))
            for image_name in os.listdir(image_path):
                incubation = re.search(r'(?<=-)\w+', os.path.splitext(image_name)[0][-10:-1]).group(0)
                incubation = int(incubation)
                if incubation<=4 :
                    image_0 = image_name
                    image_0h_name= image_path + '/' +image_name
                    image_0h = Image.open(image_0h_name)
                    image_0h = image_0h.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_0h = np.multiply(image_0h, 1.0 / 255.0)
                if 10 <= incubation <= 14:
                    image_12 = image_name
                    image_12h_name = image_path + '/' + image_name
                    image_12h = Image.open(image_12h_name)
                    image_12h = image_12h.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_12h = np.multiply(image_12h, 1.0 / 255.0)
                if 23 <= incubation <= 25:
                    image_24 = image_name
                    image_24h_name = image_path + '/' + image_name
                    image_24h = Image.open(image_24h_name)
                    image_24h = image_24h.resize((img_rows, img_cols), resample=Image.BICUBIC)
                    image_24h = np.multiply(image_24h, 1.0 / 255.0)
            if seperation == True:
                # print(image_0, image_12, image_24)
                input_batch_0h.append(image_0h)
                input_batch_12h.append(image_12h)
                input_batch_24h.append(image_24h)
            else:
                image = np.concatenate((image_0h, image_12h, image_24h),axis=-1)
                input_batch.append(image)
        label_batch = to_categorical(label_batch, length)
        batch_count += 1
        if seperation == True:
            # yield np.array(input_batch_0h), np.array(input_batch_12h), np.array(input_batch_24h), np.array(label_batch)
            yield [np.array(input_batch_0h), np.array(input_batch_12h), np.array(input_batch_24h)], np.array(label_batch)
            # print(np.array(input_batch_0h).shape, np.array(input_batch_12h).shape, np.array(input_batch_24h).shape, np.array(label_batch).shape)
        else:
            yield np.array(input_batch), np.array(label_batch)
            print(np.array(input_batch).shape, np.array(label_batch).shape)




# train_path = 'E:\\Time_0_12_24_one_folder\\TOP_BLACK\\train'
# test_path = 'E:\\Time_0_12_24_one_folder\\TOP_BLACK\\test'
# val_path = 'E:\\Time_0_12_24_one_folder\\TOP_BLACK\\val'
# # input_batch, label_batch = batch_generator(data_path, batch_size=4, img_cols=800, img_rows=800, interpolation=cv2.INTER_CUBIC, shuffle=True)
#
# batch_generator_0_12_24(test_path, batch_size=4, img_cols=800, img_rows=800, interpolation=cv2.INTER_CUBIC, length=5, shuffle=False, seperation=True)
# batch_generator_0_12_24(val_path, batch_size=4, img_cols=800, img_rows=800, interpolation=cv2.INTER_CUBIC, length=5,shuffle=False, seperation=True)
# batch_generator_0_12_24(train_path, batch_size=4, img_cols=800, img_rows=800, interpolation=cv2.INTER_CUBIC, length=5, shuffle=True, seperation=True)
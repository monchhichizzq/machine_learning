from __future__ import print_function


from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import concatenate
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import regularizers
from keras.layers import AveragePooling2D, add, ZeroPadding2D
from keras.applications import vgg16, resnet50
from keras.models import load_model,Model
from keras.initializers import VarianceScaling
# import keras.backend as K
# dtype='float16'
# K.set_floatx(dtype)
# # default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
# K.set_epsilon(1e-4)


def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), r = 1e-2, padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    # x = Conv2D(nb_filter, kernel_size, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(r),
    #            use_bias=True, bias_initializer='zero', padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = Conv2D(nb_filter, kernel_size, kernel_initializer='he_normal', use_bias=True, bias_initializer='zero', padding=padding, strides=strides,
               name=conv_name)(x)
    # x = Conv2D(nb_filter, kernel_size, kernel_initializer= VarianceScaling(scale=1.0, mode='fan_avg', distribution='uniform', seed=None), kernel_regularizer=regularizers.l2(r),
    #            bias_initializer='zero', use_bias=True, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    x = Activation('relu')(x)
    return x


def VGG16(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2, top_bottom = 'TOP'):
    Inpt = Input(shape=input_shape, name='Input_'+top_bottom)
    # Block 1
    x = Conv2d_BN(Inpt, 64, (3, 3), (1, 1), padding='same', name='block1_conv1' + top_bottom)
    x = Conv2d_BN(x, 64, (3, 3), (1, 1),  padding='same', name='block1_conv2'+ top_bottom)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'+ top_bottom)(x)

    # Block 2
    x = Conv2d_BN(x, 128, (3, 3), (1, 1), padding='same', name='block2_conv1'+ top_bottom)
    x = Conv2d_BN(x, 128, (3, 3), (1, 1), padding='same', name='block2_conv2'+ top_bottom)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'+ top_bottom)(x)

    # Block 3
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), padding='same', name='block3_conv1'+ top_bottom)
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), padding='same', name='block3_conv2'+ top_bottom)
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), padding='same', name='block3_conv3'+ top_bottom)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'+ top_bottom)(x)

    # Block 4
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block4_conv1'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block4_conv2'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block4_conv3'+ top_bottom)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'+ top_bottom)(x)

    # Block 5
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block5_conv1'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block5_conv2'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block5_conv3'+ top_bottom)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'+ top_bottom)(x)

    # Input 50*50*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv1'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv2'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv3'+ top_bottom)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool'+ top_bottom)(x)
    # output 25*25*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv1'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv2'+ top_bottom)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv3'+ top_bottom)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool'+ top_bottom)(x)
    # output 13*13*512

    # Classification block
    x = Flatten(name='flatten_'+top_bottom)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r), name='fc1'+ top_bottom)(x)
    x = Dropout(1 - keep_prob, name='dropout_1_'+top_bottom)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r), name='fc2'+ top_bottom)(x)
    x = Dropout(1 - keep_prob, name='dropout_2_'+top_bottom)(x)
    x = Dense(classes, activation='softmax', name='predictions'+top_bottom)(x)

    # Create model.
    model = Model(Inpt, x, name='vgg16_'+top_bottom)
    model.summary()
    return model


def vgg16_transfer(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2, side='top_0'):
    Inpt = Input(shape=input_shape, name='Input_' + side)
    model = vgg16.VGG16(include_top=False, weights='imagenet', input_tensor= Inpt)
    # Freeze the layers except the last 4 layers
    # for layer in model.layers:
    #     layer.trainable = False
    #     print(layer, layer.trainable)
    #     print(layer.get_config())

    for layer in model.layers:
        layer.trainable = True
        # print(layer, layer.trainable)
    model_vgg = Model(input=Inpt, output=model.output, name='VGG16_'+side)
    x = model_vgg(Inpt)
    # Input 50*50*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block6_conv1_' + side)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block6_conv2_' + side)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block6_conv3_' + side)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool_'+side)(x)
    # output 25*25*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block7_conv1_' + side)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block7_conv2_' + side)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block7_conv3_' + side)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool_' + side)(x)
    # output 13*13*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block8_conv1_' + side)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block8_conv2_' + side)
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r=r, padding='same', name='block8_conv3_' + side)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block_pool_' + side)(x)
    x = Flatten()(x)  # 展平
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc1_' + side)(x)
    x = Dropout(1 - keep_prob, name='dropout1_' + side)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc2_' + side)(x)
    x = Dropout(1-keep_prob, name='dropout2_' + side)(x)
    predictions = Dense(classes, activation='softmax',  name='predictions_' + side)(x)
    model = Model(input=Inpt, output=predictions)
    model.summary()
    return model



def vgg16_24h_BOTTOM_TOP(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2):
    Inpt = Input(shape=input_shape)
    # Block 1 800*800*6
    x = Conv2d_BN(Inpt, 64, (3, 3), (1, 1), padding='same', name='block1_conv1')
    x = Conv2d_BN(x, 64, (3, 3), (1, 1),  padding='same', name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2 400*400*64
    x = Conv2d_BN(x, 128, (3, 3), (1, 1), padding='same', name='block2_conv1')
    x = Conv2d_BN(x, 128, (3, 3), (1, 1), padding='same', name='block2_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3 200*200*128
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), padding='same', name='block3_conv1')
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), padding='same', name='block3_conv2')
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), padding='same', name='block3_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4 100*100*256
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block4_conv1')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block4_conv2')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block4_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5 50*50*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block5_conv1')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block5_conv2')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block5_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Block 6 25*25*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv1')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv2')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)

    # Block 7 12*12*512
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv1')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv2')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool')(x)
    # output 6*6*512
    x = Flatten()(x)  # 展平
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc1')(x)
    x = Dropout(1 - keep_prob)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc2')(x)
    x = Dropout(1-keep_prob)(x)
    predictions = Dense(classes, activation='softmax',  name='predictions')(x)
    model = Model(input=Inpt, output=predictions)
    model.summary()
    return model


def vgg16_0_12_24(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2):
    Inpt_0h = Input(shape=input_shape)
    Inpt_12h = Input(shape=input_shape)
    Inpt_24h = Input(shape=input_shape)
    Inpt = concatenate([Inpt_0h, Inpt_12h, Inpt_24h],axis=-1)
    # Block 1
    x = Conv2d_BN(Inpt, 64, (3, 3), (1, 1), r,padding='same', name='block1_conv1')
    x = Conv2d_BN(x, 64, (3, 3), (1, 1), r, padding='same', name='block1_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2d_BN(x, 128, (3, 3), (1, 1), r, padding='same', name='block2_conv1')
    x = Conv2d_BN(x, 128, (3, 3), (1, 1), r, padding='same', name='block2_conv2')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), r, padding='same', name='block3_conv1')
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), r, padding='same', name='block3_conv2')
    x = Conv2d_BN(x, 256, (3, 3), (1, 1), r, padding='same', name='block3_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r, padding='same', name='block4_conv1')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r, padding='same', name='block4_conv2')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r, padding='same', name='block4_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r, padding='same', name='block5_conv1')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r, padding='same', name='block5_conv2')
    x = Conv2d_BN(x, 512, (3, 3), (1, 1), r, padding='same', name='block5_conv3')
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # # Input 50*50*512
    # x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv1')
    # x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv2')
    # x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block6_conv3')
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(x)
    #
    # # output 25*25*512
    # x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv1')
    # x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv2')
    # x = Conv2d_BN(x, 512, (3, 3), (1, 1), padding='same', name='block7_conv3')
    # x = MaxPooling2D((2, 2), strides=(2, 2), name='block7_pool')(x)

    # output 13*13*512
    x = Flatten()(x)  # 展平
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc1')(x)
    x = Dropout(1 - keep_prob)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc2')(x)
    x = Dropout(1-keep_prob)(x)
    predictions = Dense(classes, activation='softmax',  name='predictions')(x)
    model = Model(input=[Inpt_0h, Inpt_12h, Inpt_24h], output=predictions)
    model.summary()
    return model







def vgg16_0_12_24_V2(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2):
    Inpt_0h = Input(shape=input_shape)
    Inpt_12h = Input(shape=input_shape)
    Inpt_24h = Input(shape=input_shape)

    output = concatenate([Inpt_0h, Inpt_12h, Inpt_24h],axis=-1)
    x = output

    # Input 50*50*1536
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # output 25*25*512
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # output 12*12*512
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(output)  # 6*6*512
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc1')(x)
    x = Dropout(1 - keep_prob)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc2')(x)
    x = Dropout(1-keep_prob)(x)
    predictions = Dense(classes, activation='softmax',  name='predictions')(x)
    model = Model(input=[Inpt_0h, Inpt_12h, Inpt_24h], output=predictions)
    model.summary()
    return model


def vgg16_stacked(Inpt, r):
    # Input 50*50*1536
    x = Conv2d_BN(Inpt, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # output 25*25*512
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # output 12*12*512
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 512, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # output 6*6*512
    return x



def vgg16_TOP_BOTTOM_V3(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2):
    Inpt_TOP = Input(shape=input_shape)
    Inpt_BOTTOM = Input(shape=input_shape)
    # output_TOP = vgg16_stacked(Inpt_TOP, r)
    # output_BOTTOM = vgg16_stacked(Inpt_BOTTOM, r)
    # output = concatenate([output_BOTTOM, output_TOP],axis=-1)
    # Input 50*50*1536
    Inpt = concatenate([Inpt_BOTTOM, Inpt_TOP], axis=-1)
    x = Conv2d_BN(Inpt, 1048, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # output 25*25*512
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # output 12*12*512
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = Conv2d_BN(x, 1048, (3, 3), r, (1, 1), padding='same')
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    # output 6*6*512
    x = Flatten()(x)  # 6*6*1024
    x = Dense(8192, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc1')(x)
    x = Dropout(1 - keep_prob)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc2')(x)
    x = Dropout(1-keep_prob)(x)
    x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r), name='fc3')(x)
    x = Dropout(1 - keep_prob)(x)
    predictions = Dense(classes, activation='softmax',  name='predictions')(x)
    model = Model(input=[Inpt_BOTTOM, Inpt_TOP], output=predictions)
    model.summary()
    return model



def Conv_Block(inpt, nb_filter, kernel_size, r, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter[0], kernel_size=(1, 1), r=r, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[1], kernel_size=(3, 3), r=r, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter[2], kernel_size=(1, 1),r=r, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter[2], strides=strides, kernel_size=kernel_size, r=r)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x

def resnet50_transfer(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2):
    model = resnet50.ResNet50(weights='imagenet', include_top=False,input_shape=input_shape)
    # Freeze the layers except the last 4 layers
    for layer in model.layers:
        layer.trainable = True
        # print(layer, layer.trainable)
    x = model.output
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r, strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)

    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r, strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dropout(1 - keep_prob)(x)
    predictions = Dense(classes, activation='softmax',  name='predictions')(x)
    model = Model(input=model.input, output=predictions)
    model.summary()
    return model


def resnet_24h_BOTTOM_TOP(input_shape=None, keep_prob=0.5, classes=1000, r=1e-2):
    Inpt_bottom = Input(shape=input_shape)
    Inpt_top = Input(shape=input_shape)
    output_bottom = resnet_architecture(Inpt_bottom, r)
    output_top = resnet_architecture(Inpt_top, r)
    x = concatenate([output_bottom, output_top], axis=-1)
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    # x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc1')(x)
    # x = Dropout(1 - keep_prob)(x)
    # x = Dense(4096, activation='relu', kernel_regularizer=regularizers.l2(r),name='fc2')(x)
    x = Dropout(1-keep_prob)(x)
    predictions = Dense(classes, activation='softmax',  name='predictions')(x)
    model = Model(input=[Inpt_bottom, Inpt_top], output=predictions)
    model.summary()
    return model

def resnet_architecture(Inpt, r):
    # 0 stage
    x = ZeroPadding2D((3, 3))(Inpt)
    # 1st stage, input: 806*806*3
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid', r=r)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    # 2nd stage, input: 400*400*64
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), r=r, strides=(1, 1), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[64, 64, 256], kernel_size=(3, 3), r=r)
    # 3nd stage, input: 200*200*256
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), r=r, strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[128, 128, 512], kernel_size=(3, 3), r=r)
    # 4nd stage, input: 100*100*512
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), r=r, strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[256, 256, 1024], kernel_size=(3, 3), r=r)
    # 5nd stage, input: 50*50*1024
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r, strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    # 6nd stage, input: 25*25*2048
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r, strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    # 7nd stage, input: 13*13*2048
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r, strides=(2, 2), with_conv_shortcut=True)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    x = Conv_Block(x, nb_filter=[512, 512, 2048], kernel_size=(3, 3), r=r)
    # output 7*7*2048
    return x

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder

    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)

    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=regularizers.l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

    Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
    bottleneck layer
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filter maps is
    doubled. Within each stage, the layers have the same number filters and the
    same filter map sizes.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)

    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # Start model definition.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # Instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # bottleneck residual unit
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])

        num_filters_in = num_filters_out

    # Add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
"""
Library code to build and use a Tiramisu CNN using the Keras functional API
Tiramisu: a specific U-Net architecture described by https://arxiv.org/abs/1611.09326
"""

import numpy as np
from keras import layers, regularizers
from keras.models import Model

####################################################
# the model
####################################################

# shared CNN parameters
REGULARIZER_L = 1e-4
DROPOUT_RATE = 0.2
PADDING ='same'
INITIALIZER = 'he_uniform'
POOLING = (2, 2)


def tiramisu(blocks=[4, 5, 7, 10, 12], bottleneck=15,  # architecture of the tiramisu
    n_classes=12, input_shape=(224, 224, 3)):  # properties of the data

    ##########################
    # image input
    _input = layers.Input(shape=input_shape)
    ##########################
    # conv layer
    x = layers.Convolution2D(48, (3, 3), strides=(1, 1),
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(_input)
    ##########################
    # down path
    skips = []
    for nb in blocks:
        x = _dense_block(nb, x, end2end=True)
        skips.append(x)
        x = _transition_down(x)
    ##########################
    # bottleneck
    x = _dense_block(bottleneck, x)
    ##########################
    # up path
    for nb in blocks[::-1]:
        x = layers.concatenate([_transition_up(x), skips.pop()])
        x = _dense_block(nb, x)
    ##########################
    # conv layer
    x = layers.Convolution2D(n_classes, (1, 1), strides=(1, 1),
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    ##########################
    # segmented image output
    x = layers.Activation('softmax')(x)
    _output = layers.Reshape((-1, n_classes))(x)
    ####################################################
    # put it together
    model = Model(inputs=_input, outputs=_output)
    return model


def _layer(x):
    filters = 16  # the growth rate (# of feature maps added per layer)
    kernel = (3, 3)
    stride = (1, 1)
    x = layers.BatchNormalization(beta_regularizer=regularizers.l2(REGULARIZER_L),
                                  gamma_regularizer=regularizers.l2(REGULARIZER_L))(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(filters, kernel, strides=stride,
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    return x


def _dense_block(n_layers, x, end2end=False):
    # if end2end, will provide a path from input to output, as well as internal paths
    _these_layer_outputs = []
    _start = x
    # n-1 layers with their outputs concatted to their inputs
    for i in range(n_layers-1):
        lyr = _layer(x)
        _these_layer_outputs.append(lyr)
        x = layers.concatenate([x, lyr])
    # one more layer, then concatenate all of the layer outputs
    _these_layer_outputs.append(_layer(x))
    if end2end:
        _these_layer_outputs.append(_start)
    x = layers.concatenate(_these_layer_outputs)
    return x


def _transition_down(x):
    filters = x.get_shape().as_list()[-1]  # same number of output feature maps as input
    kernel = (1, 1)
    stride = (1, 1)
    x = layers.BatchNormalization(beta_regularizer=regularizers.l2(REGULARIZER_L),
                                  gamma_regularizer=regularizers.l2(REGULARIZER_L))(x)
    x = layers.Activation('relu')(x)
    x = layers.Convolution2D(filters, kernel, strides=stride,
                             padding=PADDING, kernel_initializer=INITIALIZER,
                             kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.MaxPooling2D(pool_size=POOLING, strides=POOLING)(x)
    return x


def _transition_up(x):
    filters = x.get_shape().as_list()[-1]  # same number of output feature maps as input
    kernel = (3, 3)
    x = layers.Conv2DTranspose(filters, kernel, strides=POOLING,
                               padding=PADDING, kernel_initializer=INITIALIZER,
                               kernel_regularizer=regularizers.l2(REGULARIZER_L))(x)
    return x

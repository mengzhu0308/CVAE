#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/20 11:35
@File:          decoder.py
'''

from keras.layers import *

def decoder(x):
    x = Dense(7 * 7 * 128)(x)
    x = Reshape((7, 7, 128))(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same')(x)
    x = ReLU()(x)

    x = UpSampling2D()(x)
    x = Conv2D(32, 3, padding='same')(x)
    x = ReLU()(x)

    x = Conv2D(1, 3, padding='same')(x)
    x = Activation('sigmoid')(x)

    return x

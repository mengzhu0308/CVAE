#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/20 11:34
@File:          encoder.py
'''

from keras.layers import *

def encoder(x, latent_dim=50):
    x = Conv2D(32, 3, padding='same')(x)
    x = ReLU()(x)

    x = Conv2D(64, 4, strides=2, padding='same')(x)
    x = ReLU()(x)

    x = Conv2D(128, 4, strides=2, padding='same')(x)
    x = ReLU()(x)

    x = Flatten()(x)
    return [Dense(latent_dim)(x) for _ in range(2)]

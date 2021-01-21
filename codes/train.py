#! -*- coding:utf-8 -*-

'''
@Author:        ZM
@Date and Time: 2021/1/1 20:12
@File:          train.py
'''

import math
import numpy as np
import cv2
from keras.layers import Input, Lambda
from keras import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import Callback

from Loss import Loss
from Dataset import Dataset
from mnist_dataset import get_mnist
from generator import generator
from encoder import encoder
from decoder import decoder

class VAELoss(Loss):
    def compute_loss(self, inputs):
        z_mean, z_log_var, x, z_decoded = inputs
        x = K.batch_flatten(x)
        z_decoded = K.batch_flatten(z_decoded)
        cross_loss = K.mean(K.binary_crossentropy(x, z_decoded), axis=-1)
        kl_loss = -0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(cross_loss + kl_loss)

if __name__ == '__main__':
    batch_size = 512
    img_size = (28, 28, 1)
    dst_img_size = (140, 140)
    latent_dim = 50

    (X_train, Y_train), _ = get_mnist()
    X_train = X_train[Y_train == 8]
    X_train = X_train.astype('float32') / 255
    X_train = np.expand_dims(X_train, 3)

    dataset = Dataset(X_train)
    generator = generator(dataset, batch_size=batch_size, shuffle=True)

    encoder_input = Input(shape=img_size, dtype='float32')
    encoder_model = Model(encoder_input, encoder(encoder_input, latent_dim=latent_dim))
    decoder_input = Input(shape=(latent_dim, ), dtype='float32')
    decoder_model = Model(decoder_input, decoder(decoder_input))
    train_input = Input(shape=img_size, dtype='float32')
    z_mean, z_log_var = encoder_model(train_input)
    def sample(inputs):
        z_mean, z_log_var = inputs
        epsilon = K.random_normal(K.shape(z_log_var))
        return z_mean + K.exp(z_log_var / 2) * epsilon
    z = Lambda(sample)([z_mean, z_log_var])
    z_decoded = decoder_model(z)
    z_decoded = VAELoss(output_axis=-1)([z_mean, z_log_var, train_input, z_decoded])
    train_model = Model(train_input, z_decoded)
    train_model.compile(Adam(learning_rate=1e-4))

    def evaluate():
        random_latent_vector = np.random.normal(size=(1, latent_dim))
        generated_image = decoder_model.predict_on_batch(random_latent_vector)[0]

        img = cv2.resize(np.round(generated_image * 255).astype('uint8'), dst_img_size)
        cv2.imwrite('generated_image.png', img)

    class Evaluator(Callback):
        def __init__(self):
            super(Evaluator, self).__init__()

        def on_epoch_end(self, epoch, logs=None):
            evaluate()

    evaluator = Evaluator()

    train_model.fit_generator(
        generator,
        steps_per_epoch=math.ceil(len(X_train) / batch_size),
        epochs=150,
        callbacks=[evaluator],
        shuffle=False,
        initial_epoch=0
    )

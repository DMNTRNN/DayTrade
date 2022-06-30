#!/usr/bin/python
# -*- coding: utf-8 -*-
from tensorflow import math
from tensorflow import zeros_initializer
from tensorflow import random_normal_initializer
from tensorflow import keras
from tensorflow import Variable
from tensorflow.keras import layers
import numpy as np

tr = 145000

# coinlist=['XRP-USDT','SOL-USDT','ADA-USDT','DOT-USDT']

coinlist = ['ETH-USDT']
dtst = []
lbls = []

# algorithm can compile the dataset fro numerous processed files

for name in coinlist:
    d = np.load('D ' + name + '.npy')
    ll = np.load('L ' + name + '.npy')
    dataset = d[-tr:-5000]
    dataset = dataset.astype('float32')
    labelRaw = ll[-tr:-5000]
    label = np.zeros((len(labelRaw), 7))
    label[:, 1] = labelRaw[:, 1]
    label[labelRaw[:, 1] > 1.004, 4] = 1  # here we choose lower surge limit. In this particular exampel occasions where the surge is

    # greater than 0.4% compared to the last close prise of a dataset instance  will be labeled as 1

    dtst.append(dataset)
    lbls.append(label)

dataset = np.concatenate(dtst)
labels = np.concatenate(lbls)


# this class initially was used for ensempling models, but here it works as a filter

class PASS(keras.layers.Layer):

    def __init__(
        self,
        initializer,
        kernel_regularizer=None,
        **kwargs
        ):
        super(PASS, self).__init__(**kwargs)
        self.initializer = initializer
        self.kernel_regularizer = kernel_regularizer

    def build(self, input_shape):
        self.w = self.add_weight(shape=input_shape[-1],
                                 initializer=self.initializer,
                                 trainable=True,
                                 regularizer=self.kernel_regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'initializer': self.initializer,
                      'kernel_regularizer': self.kernel_regularizer})
        return config

    def call(self, inputs):
        return math.multiply(inputs, self.w)


dataset = dataset[-140000:]
labels = labels[-140000:]

i0 = dataset[:, 0, :]
i1 = dataset[:, 1, :]
i2 = dataset[:, 2, :]
i3 = dataset[:, 3, :]
i4 = dataset[:, 4, :]
i5 = dataset[:, 5, :]
i6 = dataset[:, 6, :]
i7 = dataset[:, 7, :]
i8 = dataset[:, 8, :]
i9 = dataset[:, 9, :]
i10 = dataset[:, 10, :]
i11 = dataset[:, 11, :]
label = labels[:, 4]


def weight(a, t, k):
    rez = np.zeros(len(a))
    for i in range(len(a)):
        rez[i] = (k * (a[i] - t)) ** 1 + 1
    return rez


# adding class weights to mitigate imbalance

sweights = np.ones(len(label))
sweights[label == 1] = 1.5

W = 256

# the model itself is casual TCN-like model. Because no sophisticated training techniques
# were applied whole model is written directly to the global variables fopr the sake of simplicity  
I0 = keras.Input(shape=W, name='i0')
I1 = keras.Input(shape=W, name='i1')
I2 = keras.Input(shape=W, name='i2')
I3 = keras.Input(shape=W, name='i3')
I4 = keras.Input(shape=W, name='i4')
I5 = keras.Input(shape=W, name='i5')
I6 = keras.Input(shape=W, name='i6')
I7 = keras.Input(shape=W, name='i7')
I8 = keras.Input(shape=W, name='i8')
I9 = keras.Input(shape=W, name='i9')
I10 = keras.Input(shape=W, name='i10')
I11 = keras.Input(shape=W, name='i11')

Ir0 = layers.Reshape((W, 1))(I0)
Ir1 = layers.Reshape((W, 1))(I1)
Ir2 = layers.Reshape((W, 1))(I2)
Ir3 = layers.Reshape((W, 1))(I3)
Ir4 = layers.Reshape((W, 1))(I4)
Ir5 = layers.Reshape((W, 1))(I5)
Ir6 = layers.Reshape((W, 1))(I6)
Ir7 = layers.Reshape((W, 1))(I7)
Ir8 = layers.Reshape((W, 1))(I8)
Ir9 = layers.Reshape((W, 1))(I9)
Ir10 = layers.Reshape((W, 1))(I10)
Ir11 = layers.Reshape((W, 1))(I11)


def scheduler(x):
    if x < 1:
        return 0.001
    if x < 6:
        return 0.0002
    else:
        return 0.0001


# u is number of units in conv layers

u = 16

callback = keras.callbacks.LearningRateScheduler(scheduler)

# because of hyperparameter tuning the input is separated above and here also:

vol1 = layers.concatenate([Ir4, Ir7, Ir0, Ir6, Ir11])
kline = layers.concatenate([Ir1, Ir2, Ir3, Ir10])
total = layers.concatenate([vol1, kline], axis=-1)

total = PASS('ones', kernel_regularizer='l2')(total)

v110 = layers.Conv1D(
    u,
    1,
    strides=1,
    padding='causal',
    data_format='channels_last',
    dilation_rate=1,
    )(total)
v110 = layers.BatchNormalization()(v110)
v110 = layers.Dropout(0.25)(v110)

v111 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=1)(total)
v111 = layers.BatchNormalization()(v111)
v111 = layers.LeakyReLU()(v111)
v111 = layers.Dropout(0.25)(v111)

v112 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=2)(v111)
v112 = layers.BatchNormalization()(v112)
v112 = layers.LeakyReLU()(v112)
v112 = layers.Dropout(0.25)(v112)

v113 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=4)(v112)
v113 = layers.BatchNormalization()(v113)
v113 = layers.LeakyReLU()(v113)
v113 = layers.Dropout(0.25)(v113)

v114 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=8)(v113)
v114 = layers.BatchNormalization()(v114)
v114 = layers.LeakyReLU()(v114)
v114 = layers.Add()([v114, v110])
v114 = layers.Dropout(0.25)(v114)

v115 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=16)(v114)
v115 = layers.BatchNormalization()(v115)
v115 = layers.LeakyReLU()(v115)
v115 = layers.Dropout(0.25)(v115)

v116 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=32)(v115)
v116 = layers.BatchNormalization()(v116)
v116 = layers.LeakyReLU()(v116)
v116 = layers.Dropout(0.25)(v116)

v117 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=64)(v116)
v117 = layers.BatchNormalization()(v117)
v117 = layers.LeakyReLU()(v117)
v117 = layers.Dropout(0.25)(v117)

v118 = layers.Conv1D(u, 2, strides=1, padding='causal',
                     dilation_rate=128)(v117)
v118 = layers.BatchNormalization()(v118)
v118 = layers.LeakyReLU()(v118)
v118 = layers.Add()([v118, v114])

rez11 = PASS('zeros', kernel_regularizer='l2')(v118[:, -1, :])

rez11 = layers.Dense(12)(rez11)
rez11 = layers.LeakyReLU()(rez11)

catN2 = layers.Dense(1, activation='sigmoid', use_bias=False,
                     name='output')(rez11)

m0 = keras.Model(inputs=[
    I0,
    I1,
    I2,
    I3,
    I4,
    I5,
    I6,
    I7,
    I8,
    I9,
    I10,
    I11,
    ], outputs=catN2)

opt = keras.optimizers.Adam()

m0.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False,
           label_smoothing=0.3, reduction='auto',
           name='binary_crossentropy'), optimizer=opt,
           weighted_metrics=[], metrics=[keras.metrics.Precision(),
           keras.metrics.Recall()])

m0.fit(
    {
        'i0': i0,
        'i1': i1,
        'i2': i2,
        'i3': i3,
        'i4': i4,
        'i5': i5,
        'i6': i6,
        'i7': i7,
        'i8': i8,
        'i9': i9,
        'i10': i10,
        'i11': i11,
        },
    {'output': label},
    epochs=10,
    validation_split=4 / 14,
    callbacks=[callback],
    batch_size=256,
    shuffle=True,
    sample_weight=sweights,
    )

m0.save('model.h5')










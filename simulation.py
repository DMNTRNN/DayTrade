#!/usr/bin/python
# -*- coding: utf-8 -*-

from tensorflow import math
from tensorflow import keras
import numpy as np
import pandas as pd

W = 256


def drawing(A):
    rez = np.zeros((13, W - 10))
    HZ = np.zeros(W - 10)
    poz = np.zeros(W - 10)
    neg = np.zeros(W - 10)
    for t in range(W - 10):
        HZ[t] = A[t, 4] * (A[t, 3] - A[t, 0]) / (2 * (A[t, 1] - A[t,
                2]) - abs(A[t, 3] - A[t, 0]))
    M = np.max(np.abs(HZ))
    for t in range(W - 10):
        HZ[t] = np.sum(HZ[t:W - 10])
    m1 = np.median(A[:W - 10, 4])
    m2 = np.median(A[:W - 10, 6])
    for t in range(W - 10):
        rez[7, t] = A[t, 7] / A[t, 4]
    for t in range(W - 10):
        rez[4, t] = min(np.log(A[t, 4] / m1 / 2), 1)
        rez[6, t] = min(np.log(A[t, 6] / m2 / 2), 1)
        rez[12, t] = 10 * A[t, 11] / A[W - 11, 11] - 10
        poz[t] = max(A[t, 3] - A[t, 0], 0)
        neg[t] = max(A[t, 0] - A[t, 3], 0)
    for t in range(10, W - 10):
        rez[11, t] = 10 * (2 * (A[t, 1] - A[t, 2]) - abs(A[t, 3] - A[t,
                           0])) / A[W - 11, 3] - 10
        rez[0, t] = 1 - 1 / (1 + np.sum(poz[t - 10:t])
                             / max(np.sum(neg[t - 10:t]), 0.000000001))
        rez[5, t] = HZ[t] / M
    for channel in (
        1,
        2,
        3,
        8,
        9,
        10,
        ):
        for t in range(W - 10):
            rez[channel, t] = 10 * A[t, channel] / A[W - 11, 3] - 10
    return rez


coinlist = ['ETH-USDT']

D = {}
for coin in coinlist:
    filename = coin + 'processed.csv'
    ADATA = pd.read_csv(filename)
    ADATA = pd.read_csv(filename)
    a = ADATA.to_numpy(copy=True)
    DATA = a[-20000:]
    D[coin] = DATA


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


model = keras.models.load_model('model.h5',
                                custom_objects={'PASS': PASS})

# here we set initial trading value

USD = 10000

# algorithm allows to simulate trading on a number of crypto pairs

inps = {}
for coin in coinlist:
    net_input = np.zeros((len(DATA) + 1, 13, W - 10))
    for i in range(W - 10, len(D[coin])):
        sample = (D[coin])[i - W + 10:i]
        net_input[i] = drawing(sample)
    inps[coin] = net_input

Predictions = {}

for coin in coinlist:
    pred = model([
        inps[coin][:, 0],
        inps[coin][:, 1],
        inps[coin][:, 2],
        inps[coin][:, 3],
        inps[coin][:, 4],
        inps[coin][:, 5],
        inps[coin][:, 6],
        inps[coin][:, 7],
        inps[coin][:, 8],
        inps[coin][:, 9],
        inps[coin][:, 10],
        inps[coin][:, 11],
        inps[coin][:, 12],
        ], training=False)

    Predictions[coin] = pred.numpy()


# the following function moves through the dataset ( the part that was not involved in training)
# and at each step checks nn prediction. If prediction is above 0.5 it simulates  opening position (it is set to work for
# long positions). It counts for 0.05% comission. As prediction window equals to 5, it waits for 5 steps to close position
# using predetermined rule, i.e. closing on reaching OCOhigh limit,
# or closes at a current price (also checks the nn prediction to be lower than0.5)

def sim(OCOhigh, coin):
    lenghts = []
    USD = 10000
    regime = 0
    counter = 0
    counterw = 0
    lenn = 0
    DATA = D[coin]
    Prediction = Predictions[coin]
    for i in range(40, len(DATA)):
        preds = Prediction[i]
        if preds >= 0.5 and regime == 0:
            regime = 1
        if regime == 1:
            buy = DATA[i - 1, 3]
            BTC = USD * 0.9995 / buy
            regime = 2
            counter += 1
            start = 0
        if regime == 2 and DATA[i, 1] >= buy * OCOhigh:
            lenghts.append(lenn)
            sell = buy * OCOhigh * 0.9995
            counterw += 1
            USD = BTC * sell
            regime = 0
        if regime == 2 and DATA[i, 1] < buy * OCOhigh and Prediction[i
                + 1] < 0.5 and start >= 5:
            lenghts.append(lenn)
            sell = DATA[i, 3] * 0.9995
            USD = BTC * sell
            regime = 0
        if regime == 2:
            start += 1

    return USD


# function, that shows accumulated result after running through the dataset. Variable entries is a list of
# OCOhigh thresholds

def RESULT(entries):
    REZ = {}
    for coin in coinlist:
        r = []
        for entry in entries:
            r.append(sim(entry, coin)[0])
        REZ[coin] = r
    return pd.DataFrame(data=REZ, index=entries)


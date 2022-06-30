#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

# W - is window size, that is 256+10, where 256 - data window and P=10 is steps of prediction window.
# It is important to avoid data leak from label window to data window. We do not use any values from
# label window to calculate  features and avoid the leak.

P = 10
W = 2256 + P


# function for extracting features

def drawing(A):
    rez = np.zeros((13, W - P))  # all features data array
    poz = np.zeros(W - P)
    neg = np.zeros(W - P)
    poz1 = np.zeros(W - P)
    neg1 = np.zeros(W - P)
    m1 = np.median(A[:W - P, 4])
    m2 = np.median(A[:W - P, 6])
    for t in range(W - P):
        rez[7, t] = A[t, 7] / A[t, 4]  # relative taker volume
    for t in range(W - P):
        rez[4, t] = min(np.log(max(A[t, 4] / m1, 1 / 2.72 ** 2)), 2)  # scaled overall volume
        rez[6, t] = min(np.log(max(A[t, 6] / m2, 1 / 2.72 ** 2)), 2)  # scaled trades number
        rez[12, t] = 10 * A[t, 11] / A[W - P - 1, P - 1] - P  # scaled average price
        poz[t] = max(A[t, 3] - A[t, 0], 0)  # RSI  formula neg
        neg[t] = max(A[t, 0] - A[t, 3], 0)  # RSI formula pos
        poz1[t] = (1 if A[t, 3] - A[t, 0] >= 0 else 0)
        neg1[t] = (1 if A[t, 3] - A[t, 0] < 0 else 0)
    for t in range(20, W - P):
        rez[11, t] = min(np.log(max(A[t, 4] * (1 - (A[t, 1] - A[t, 2])
                         / (2 * (A[t, 1] - A[t, 2]) - abs(A[t, 3]
                         - A[t, 0]))) / m1, 1 / 2.72 ** 2)), 2)
        rez[0, t] = 1 - 1 / (1 + np.sum(poz[t - 10:t])
                             / max(np.sum(neg[t - P:t]), 0.000000001))  # RSI
        rez[5, t] = 1 - 1 / (1 + np.sum(poz1[t - 10:t])
                             / max(np.sum(neg1[t - P:t]), 0.000000001))  # RSI-like feature
    for channel in (
        1,
        2,
        3,
        8,
        9,
        10,
        ):
        for t in range(W - P):  # this loop adds scaled candle values (open, high, low, close) and averages
            rez[channel, t] = 10 * A[t, channel] / A[W - P - 1, 3] - 10
    return rez


# This function adds label to a dataset instance. Basically, it calculates
# the surge which takes plase in the following 5 timesteps (label[0]). The max drop is label[1]

t_steps = 5


def LABELING(A):
    label = np.zeros(2)
    start = A[W - P - 1, 3]
    label[0] = np.max(A[W - P:W - t_steps, 1]) / start
    label[1] = np.min(A[W - P:W - t_steps, 2]) / start
    return label


# We start from 300 to calculate averages

Start = 300

# processing can be performed on a number of candlestick data files, for example:
# coinlist=['XRP-USDT','SOL-USDT','ADA-USDT','DOT-USDT','DOGE-USDT']

coinlist = ['ETH-USDT']
for name in coinlist:

    BTC = pd.read_csv('BTC-USDT.csv')  # file produced by grabber1.py for BTC-USDT pair
    ETH = pd.read_csv('ETH-USDT.csv')  # file produced by grabber1.py for ETH-USDT pair
    b = ETH.drop(columns=['open_time', 'close_time', 'ignore'])
    b1 = BTC.drop(columns=['open_time', 'close_time', 'ignore'])
    a = b.to_numpy(copy=True)
    a1 = b1.to_numpy(copy=True)
    a = np.concatenate([a, a1], axis=-1)
    suma = np.sum(a, axis=-1)
    a = a[suma != 0]
    Rez = np.zeros((len(a) - Start, 12))

    #  Rez array contains the following values (BASE: 'value', array index): ETH:'open',0; 'high',1; 'low',2;
    # 'close',3; 'volume',4; 'quote_asset_volume',5; 'number_of_trades',6;
    # 'taker_buy_base_asset_volume',7; 'ma100',8; 'ma25',9; 'average price', 10;
    # BTC: 'open',12

    for i in range(Start - 1, len(a)):
        if i == Start:
            ma100 = np.sum(a[i - 99:i + 1, 3]) / 100
            ma25 = np.sum(a[i - 24:i + 1, 3]) / 25
            vwap = a[i, 5] / a[i, 4]
            ii = i - Start
            Rez[ii, 0] = a[i, 0]
            Rez[ii, 1] = a[i, 1]
            Rez[ii, 2] = a[i, 2]
            Rez[ii, 3] = a[i, 3]
            Rez[ii, 4] = a[i, 4]
            Rez[ii, 5] = a[i, 5]
            Rez[ii, 6] = a[i, 6]
            Rez[ii, 7] = a[i, 7]
            Rez[ii, 8] = ma100
            Rez[ii, 9] = ma25
            Rez[ii, 10] = vwap
            Rez[ii, 11] = a[i, 12]
        if i > Start:
            ma100 = ma100 + a[i, 3] / 100 - a[i - 100, 3] / 100
            ma25 = ma100 + a[i, 3] / 25 - a[i - 25, 3] / 25
            vwap = a[i, 5] / a[i, 4]
            ii = i - Start
            Rez[ii, 0] = a[i, 0]
            Rez[ii, 1] = a[i, 1]
            Rez[ii, 2] = a[i, 2]
            Rez[ii, 3] = a[i, 3]
            Rez[ii, 4] = a[i, 4]
            Rez[ii, 5] = a[i, 5]
            Rez[ii, 6] = a[i, 6]
            Rez[ii, 7] = a[i, 7]
            Rez[ii, 8] = ma100
            Rez[ii, 9] = ma25
            Rez[ii, 10] = vwap
            Rez[ii, 11] = a[i, 12]

    DATA = pd.DataFrame(Rez, columns=[
        'open',
        'high',
        'low',
        'close',
        'volumeB',
        'volumeQ',
        'trades',
        'takerVol',
        'ma100',
        'ma25',
        'vwap',
        'BTCopen',
        ])
    DATA.dropna()
    DATA.to_csv(name + 'processed.csv', index=False)
    N = len(DATA)

    LIB = np.zeros((N - W, W, 12))
    for i in range(W, N):
        LIB[i - W] = DATA.values[i - W:i]

    # dataset and labels are written in separate files

    DATASET = np.zeros((len(LIB), 13, W - 10), dtype='float32')
    LABELS = np.zeros((len(LIB), 2))
    i = -1
    for timestep in LIB:
        i += 1
        dd = drawing(timestep)
        ll = LABELING(timestep)
        if np.sum(np.isnan(dd)) != 0:
            continue
        DATASET[i] = dd
        LABELS[i] = ll
    np.save('D ' + name, DATASET)
    np.save('L ' + name, LABELS)

                        
                    


        
 

        
        


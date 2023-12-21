# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import random
import numpy as np
import scipy.signal as sig
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
default_f = [0.603661872, 0.47138388, 0.52861612]


def preprocess_signal(x, info, data_cfg, preprocess_cfg, preprocessed=False, split_idx=None, save_path=None):
    """ resample, filter, scale ecg signal and get features """
    lead_idxs, gains, baselines, sample_rate, age, sex = info

    # select leads and preprocess with gains and baselines
    #x = x[lead_idxs, :]
    if not preprocessed:
        x = x[lead_idxs, :]
        for i in range(len(lead_idxs)):
            x[i, :] = (x[i, :] - baselines[i]) / gains[i]

        if sample_rate != preprocess_cfg.sample_rate:
            num = x.shape[1] // sample_rate * preprocess_cfg.sample_rate
            x = sig.resample(x, num, axis=1)
            sample_rate = preprocess_cfg.sample_rate
        x = filter_signal(x, sample_rate, preprocess_cfg)
        x = scale_signal(x, preprocess_cfg)

        if save_path is not None:
            np.save(save_path, x)

    if split_idx == "train":
        if x.shape[1] > data_cfg.chunk_length:
            pos = random.randint(0, x.shape[1] - data_cfg.chunk_length)
            x = x[:, pos:pos+data_cfg.chunk_length]
        elif x.shape[1] < data_cfg.chunk_length:
            pad_l = (data_cfg.chunk_length - x.shape[1]) // 2
            pad_r = data_cfg.chunk_length - x.shape[1] - pad_l
            x = np.pad(x, ((0, 0), (pad_l, pad_r)), 'constant')

    # features (age, one-hot encoded sex, and missing_flags)s
    f = np.zeros((5), dtype=np.float32)
    if not np.isnan(age):
        f[0] = age
    else:
        f[0] = default_f[0]
        f[3] = 1

    if sex in ('Female', 'female', 'F', 'f'):
        f[1] = 1
    elif sex in ('Male', 'male', 'M', 'm'):
        f[2] = 1
    else:
        f[1] = default_f[1]
        f[2] = default_f[2]
        f[4] = 1

    return x, f


def filter_signal(x, sample_rate, preprocess_cfg):
    """ filter ecg signal """
    nyq = sample_rate * 0.5
    for i in range(len(x)):
        for cutoff in preprocess_cfg.filter_highpass:
            x[i] = sig.filtfilt(*sig.butter(2, cutoff / nyq, btype='highpass'), x[i])
        for cutoff in preprocess_cfg.filter_lowpass:
            if cutoff >= nyq: cutoff = nyq - 0.05
            x[i] = sig.filtfilt(*sig.butter(2, cutoff / nyq, btype='lowpass'), x[i])
        for cutoff in preprocess_cfg.filter_notch:
            x[i] = sig.filtfilt(*sig.iirnotch(cutoff, cutoff, sample_rate), x[i])

    return x


def scale_signal(x, preprocess_cfg):
    """ scale ecg signal """
    for i in range(len(x)):
        if preprocess_cfg.scaler is None: continue
        elif "minmax" in preprocess_cfg.scaler:   scaler = MinMaxScaler(feature_range=(-1, 1))
        elif "standard" in preprocess_cfg.scaler: scaler = StandardScaler()
        elif "robust" in preprocess_cfg.scaler:   scaler = RobustScaler()
        scaler.fit(np.expand_dims(x[i], 1))
        x[i] = scaler.transform(np.expand_dims(x[i], 1)).squeeze()

    return x


def preprocess_label(labels, scored_classes, equivalent_classes):
    """ convert string labels to binary labels """
    y = np.zeros((len(scored_classes)), np.float32)
    for label in labels:
        if label in equivalent_classes:
            label = equivalent_classes[label]

        if label in scored_classes:
            y[scored_classes.index(label)] = 1

    return y

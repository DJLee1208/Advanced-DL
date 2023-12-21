# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)
# Some parts of the code were referenced from or inspired by below
# PhysioNet Challenge (https://github.com/physionetchallenges/python-classifier-2021)

import os
import time
import numpy as np
from scipy.io import loadmat

import torch

from src.preprocess import preprocess_signal, preprocess_label
from src.evaluate import load_weights


class ECG_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for ECG dataset """
    def __init__(self, X, F, Y, filenames=None, headers=None, data_cfg=None, preprocess_cfg=None, split_idx=None):
        self.X = X
        self.F = F
        self.Y = Y
        self.filenames = filenames
        self.headers = headers
        self.data_cfg = data_cfg
        self.preprocess_cfg = preprocess_cfg
        self.split_idx = split_idx

        self.data_preprocessed_path = './dataset_preprocessed'

        if not os.path.exists(self.data_preprocessed_path):
            os.mkdir(self.data_preprocessed_path)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, i):
        if len(self.X) > 0:
            x = self.X[i]
            f = self.F[i]
        else:
            # stime = time.time()
            preprocessed_path = os.path.join(self.data_preprocessed_path,
                                             self.filenames[i].split('/')[-1] + "-" + self.preprocess_cfg.preprocess_idx + ".npy")
            preprocessed = os.path.isfile(preprocessed_path)
            x, f = preprocess_signal(load_data(self.filenames[i], preprocessed, self.preprocess_cfg.preprocess_idx, preprocessed_filename=preprocessed_path),
                                     get_info(self.headers[i], self.data_cfg.leads),
                                     self.data_cfg, self.preprocess_cfg, preprocessed, self.split_idx, save_path=preprocessed_path)
            # print(time.time() - stime)
            x = torch.from_numpy(x)
            f = torch.from_numpy(f)
        y = self.Y[i]

        return x, f, y


def get_dataset_from_configs(data_cfg, preprocess_cfg, data_directory, dataset_idx="all", split_idx=None, pre_loading=False, sanity_check=False):
    """ load ecg dataset from config files """
    if data_cfg.data is not None:
        x, f = preprocess_signal(data_cfg.data, get_info(data_cfg.header, data_cfg.leads), data_cfg, preprocess_cfg)
        y = np.zeros((1), np.float32)  # dummy label for code compatibility
        X = [torch.from_numpy(x).float()]
        F = [torch.from_numpy(f).float()]
        Y = [torch.from_numpy(y)]
        dataset = ECG_dataset(X, F, Y)
    else:
        if data_cfg.filenames is not None: 
            filenames_all = data_cfg.filenames
        else:                              
            filenames_all = get_filenames_from_split(data_cfg, dataset_idx, split_idx, data_directory)
        X, F, Y, filenames, headers = [], [], [], [], []
        for filename in filenames_all:
            header = load_header(filename)
            y = preprocess_label(get_labels(header), data_cfg.scored_classes, data_cfg.equivalent_classes)
            if np.sum(y) == 0 and not sanity_check: continue

            Y.append(torch.from_numpy(y))
            if pre_loading:
                preprocessed = os.path.exists(filename + "-" + preprocess_cfg.preprocess_idx + ".npy")
                x, f = preprocess_signal(load_data(filename, preprocessed, preprocess_cfg.preprocess_idx),
                                         get_info(header, data_cfg.leads), preprocess_cfg, preprocessed)
                X.append(torch.from_numpy(x))
                F.append(torch.from_numpy(f))
            else:
                filenames.append(filename)
                headers.append(header)

        dataset = ECG_dataset(X, F, Y, filenames, headers, data_cfg, preprocess_cfg, split_idx)

    return dataset


def load_data(filename, preprocessed=False, preprocess_idx=None, preprocessed_filename=None):
    """ load data from WFDB files """
    if not preprocessed:
        data = np.asarray(loadmat(filename + ".mat")['val'], dtype=np.float32)
    else:
        if preprocessed_filename is not None:
            data = np.load(preprocessed_filename).astype(np.float32)
        else:
            data = np.load(filename + "-" + preprocess_idx + ".npy").astype(np.float32)

    return data


def load_header(filename):
    """ load header from WFDB files """
    HEADER = open(filename + ".hea", 'r')
    header = HEADER.readlines()
    HEADER.close()

    return header


def get_info(header, leads):
    """ get lead_idxs, gains, baselines, sample_rate, age, sex from header """
    available_leads = get_leads(header)
    info = [
        [available_leads.index(lead) for lead in leads],
        *get_gains_baselines(header, leads),
        get_sample_rate(header),
        get_age(header),
        get_sex(header)
    ]

    return info


def get_sample_rate(header):
    """ get sample rate from header """
    sample_rate = int(header[0].strip().split()[2])

    return sample_rate


def get_leads(header):
    """ get leads from header """
    leads = []
    for i, line in enumerate(header):
        tokens = line.strip().split(' ')
        if i == 0:
            num_leads = int(tokens[1])
        elif i <= num_leads:
            leads.append(tokens[-1])

    return leads


def get_gains_baselines(header, leads):
    """ get analog-to-digital converter gains and baselines from header """
    gains = np.zeros(len(leads), dtype=np.float32)
    baselines = np.zeros(len(leads), dtype=np.float32)
    for i, line in enumerate(header):
        tokens = line.strip().split(' ')
        if i == 0:
            num_leads = int(tokens[1])
        elif i <= num_leads:
            lead = tokens[-1]
            if lead in leads:
                gains[leads.index(lead)] = float(tokens[2].split('/')[0])
                baselines[leads.index(lead)] = float(tokens[4].split('/')[0])

    return gains, baselines


def get_age(header):
    """ get age from header """
    for line in header:
        if line.startswith('#Age'):
            try:
                age = float(line.split(': ')[1].strip()) / 100.0
            except:
                age = float('nan')

    return age


def get_sex(header):
    """ get sex from header """
    for line in header:
        if line.startswith('#Sex'):
            sex = line.split(': ')[1].strip()

    return sex


def get_labels(header):
    """ get labels from header """
    labels = []
    for line in header:
        if line.startswith('#Dx'):
            labels = [label.strip() for label in line.split(': ')[1].split(',')]

    return labels


def get_filenames_from_split(data_cfg, dataset_idx, split_idx, data_directory):
    """ get filenames from config file and split index """
    filenames_all = []
    for d, dataset in enumerate(data_cfg.datasets):
        if dataset_idx not in ["all", dataset]: continue
        # elif data_cfg.fold is None and split_idx in ["train", "test"]: continue
        # elif data_cfg.fold is not None and split_idx in ["train", "val"] and data_cfg.fold == d: continue
        # elif data_cfg.fold is not None and split_idx in ["test"] and data_cfg.fold != d: continue

        if split_idx in ["train", "val"]:
            filenames = data_cfg.split[dataset][split_idx]
        # elif split_idx in ["test"] or data_cfg.fold in [None, d]:
        #     filenames = data_cfg.split[dataset]["all"]
        # elif data_cfg.fold != d:
        #     filenames = data_cfg.split[dataset]["val"]

        # if sanity_check and len(filenames) > 3:
        #     filenames = filenames[:3]

        filenames_all += [data_directory + "/%s" % filename for filename in filenames]
    # filenames_all = []
    # if (split_idx == "train"):
    #     for file in os.listdir(data_directory):
    #         path = data_directory
    #         if os.path.isfile(os.path.join(path, file)):
    #             if not file.lower().startswith('.') and file.lower().endswith('mat'):
    #                 filename = os.path.splitext(file)[0]
    #                 filenames_all.append(os.path.join(path, filename))
    #         else:
    #             path = os.path.join(path, file)
    #             for file in os.listdir(path):
    #                 if os.path.isfile(os.path.join(path, file)):
    #                     if not file.lower().startswith('.') and file.lower().endswith('mat'):
    #                         filename = os.path.splitext(file)[0]
    #                         filenames_all.append(os.path.join(path, filename))
    # elif (split_idx == "val"):
    #     for file in os.listdir(data_directory):
    #         path = data_directory
    #         if os.path.isfile(os.path.join(path, file)):
    #             if not file.lower().startswith('.') and file.lower().endswith('mat'):
    #                 filename = os.path.splitext(file)[0]
    #                 filenames_all.append(os.path.join(path, filename))
    #         else:
    #             path = os.path.join(path, file)
    #             for file in os.listdir(path):
    #                 if os.path.isfile(os.path.join(path, file)):
    #                     if not file.lower().startswith('.') and file.lower().endswith('mat'):
    #                         filename = os.path.splitext(file)[0]
    #                         filenames_all.append(os.path.join(path, filename))


    return filenames_all


def get_loss_weights_and_flags(data_cfg, dataset_train=None):
    """ get class and confusion weights for training """
    if dataset_train is not None:
        Y = np.stack(dataset_train.Y, 0)
        class_weight = np.sum(Y, axis=0).astype(np.float32)
        for i in range(len(class_weight)):
            class_weight[i] = class_weight[i] / len(Y)
    else:
        class_weight = None

    confusion_weight = load_weights(data_cfg.path + "weights.csv", data_cfg.scored_classes)[1]

    return class_weight, confusion_weight


def collate_into_list(args):
    """ collate variable-length ecg signals into list """
    X = [a[0] for a in args]
    F = torch.stack([a[1] for a in args], 0)
    Y = torch.stack([a[2] for a in args], 0)
    return X, F, Y


def collate_into_block(batch, l, stride):
    """
    collate variable-length ecg signals into block
    for those longer than chunk_length, divide them into l-point chunks with (overlapping) stride
    """
    X, F, Y = batch
    if stride is None:
        # assume all ecg signals have same length
        X_block = torch.stack(X, 0)
        X_flag  = X[0].new_ones((len(X)), dtype=torch.bool)
    else:
        # collate variable-length ecg signals
        b, c, = 0, X[0].shape[0]
        for x in X:
            if x.shape[1] <= l: b += 1
            else:               b += int(np.ceil((x.shape[1] - l) / float(stride) + 1))

        X_block = X[0].new_zeros((b, c, l))
        X_flag  = X[0].new_zeros((b), dtype=torch.bool)
        idx = 0
        for x in X:
            if x.shape[1] <= l:
                X_block[idx, :, :x.shape[1]] = x[:, :]
                X_flag[idx] = True
                idx += 1
            else:
                num_chunks = int(np.ceil((x.shape[1] - l) / float(stride) + 1))
                for i in range(num_chunks):
                    if i != num_chunks - 1:
                        X_block[idx] = x[:, i*stride:i*stride + l]
                        X_flag[idx] = False
                    elif x.shape[1] > l:
                        X_block[idx] = x[:, -l:]
                        X_flag[idx] = True
                    else:
                        X_block[idx, :, :x.shape[1]] = x[:, :]
                        X_flag[idx] = True
                    idx += 1

    return X_block, X_flag, F, Y

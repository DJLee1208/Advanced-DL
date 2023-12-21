# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import json

import torch

from src.utils import Print


class DataConfig():
    def __init__(self, file=None, idx="data_config"):
        """ data configurations """
        self.idx = idx
        self.path = None
        self.datasets = None
        self.fold = None
        self.leads = None
        self.chunk_length = None
        self.chunk_stride = None

        self.split = None
        self.scored_classes = None
        self.equivalent_classes = None
        self.normal_class = None

        self.data = None
        self.header = None
        self.filenames = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("data-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if   key == "path":                         self.path = value
                elif key == "datasets":                     self.datasets = value
                elif key == "fold":                         self.fold = value
                elif key == "leads":                        self.leads = value
                elif key == "chunk_length":                 self.chunk_length = value
                elif key == "chunk_stride":                 self.chunk_stride = value
                else: sys.exit("# ERROR: invalid key [%s] in data-config file" % key)

        # load information
        if self.path is not None:
            self.split = json.load(open(self.path + "split.json", "r"))
            info = json.load(open(self.path + "info.json", "r"))
            self.equivalent_classes = info["equivalent_classes"]
            scored_classes = info["scored_classes"]
            for class1, class2 in self.equivalent_classes.items():
                if class1 in scored_classes:
                    scored_classes[scored_classes.index(class1)] = class2
            self.scored_classes = sorted(set(scored_classes))
            self.normal_class = info["normal_class"]
            dataset_classes = info["datasets"]
            # for d in dataset_classes.keys():
            #     scored_classes = []
            #     for cls in dataset_classes[d]:
            #         if cls in self.equivalent_classes: scored_classes.append(self.equivalent_classes[cls])
            #         else:                              scored_classes.append(cls)
            #     dataset_classes[d] = scored_classes
            # self.dataset_classes = dataset_classes

    def get_config(self):
        configs = []
        configs.append(["path", self.path])
        configs.append(["datasets", self.datasets])
        configs.append(["fold", self.fold])
        configs.append(["leads", self.leads])
        configs.append(["chunk_length", self.chunk_length])
        configs.append(["chunk_stride", self.chunk_stride])
        configs.append(["scored_classes", len(self.scored_classes)])
        return configs


class PreprocessConfig():
    def __init__(self, file=None, idx="preprocess_config"):
        """ preprocess configurations """
        self.idx = idx
        self.preprocess_idx = None
        self.sample_rate = None
        self.filter_notch = []
        self.filter_lowpass = []
        self.filter_highpass = []
        self.scaler = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("preprocess-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key == "idx":                            self.preprocess_idx = value
                elif key == "sample_rate":                  self.sample_rate = value
                elif key == "filter_highpass":              self.filter_highpass = value
                elif key == "filter_lowpass":               self.filter_lowpass = value
                elif key == "filter_notch":                 self.filter_notch = value
                elif key == "scaler":                       self.scaler = value
                else: sys.exit("# ERROR: invalid key [%s] in preprocess-config file" % key)

    def get_config(self):
        configs = []
        configs.append(["idx", self.preprocess_idx])
        configs.append(["sample_rate", self.sample_rate])
        if len(self.filter_highpass) > 0: configs.append(["filter_highpass", self.filter_highpass])
        if len(self.filter_lowpass) > 0:  configs.append(["filter_lowpass", self.filter_lowpass])
        if len(self.filter_notch) > 0:    configs.append(["filter_notch", self.filter_notch])
        if self.scaler is not None:       configs.append(["scaler", self.scaler])

        return configs


class ModelConfig():
    def __init__(self, file=None, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.num_blocks = None
        self.width_factor = None
        self.kernel_size = None
        self.stride = None
        self.se = None
        self.dropout_rate = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("model-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if key == "num_blocks":                     self.num_blocks = value
                elif key == "width_factor":                 self.width_factor = value
                elif key == "kernel_size":                  self.kernel_size = value
                elif key == "stride":                       self.stride = value
                elif key == "se":                           self.se = value
                elif key == "dropout_rate":                 self.dropout_rate = value
                else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

    def get_config(self):
        configs = []
        configs.append(["num_blocks", self.num_blocks])
        configs.append(["width_factor", self.width_factor])
        configs.append(["kernel_size", self.kernel_size])
        configs.append(["stride", self.stride])
        configs.append(["se", self.se])
        configs.append(["dropout_rate", self.dropout_rate])

        return configs


class RunConfig():
    def __init__(self, file=None, idx="run_config", eval=False, sanity_check=False):
        """ run configurations """
        self.idx = idx
        self.eval = eval
        self.batch_size = None
        self.num_epochs = None
        self.learning_rate = None
        self.weight_decay = None
        self.mixup = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("run-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            for key, value in cfg.items():
                if   key == "batch_size":                   self.batch_size = value
                elif key == "num_epochs":                   self.num_epochs = value
                elif key == "learning_rate":                self.learning_rate = value
                elif key == "weight_decay":                 self.weight_decay = value
                elif key == "mixup":                        self.mixup = value
                else: sys.exit("# ERROR: invalid key [%s] in run-config file" % key)

        if sanity_check:
            self.batch_size = 32
            self.num_epochs = 4

    def get_config(self):
        configs = []
        configs.append(["batch_size", self.batch_size])
        if not self.eval:
            configs.append(["num_epochs", self.num_epochs])
            configs.append(["learning_rate", self.learning_rate])
            configs.append(["weight_decay", self.weight_decay])
            configs.append(["mixup", self.mixup])

        return configs


def print_configs(args, cfgs, device, output):
    if args["sanity_check"]: Print(" ".join(['##### SANITY_CHECK #####']), output)
    Print(" ".join(['##### arguments #####']), output)
    for cfg in cfgs:
        Print(" ".join(['%s:' % cfg.idx, str(args[cfg.idx])]), output)
        for c, v in cfg.get_config():
            Print(" ".join(['-- %s: %s' % (c, v)]), output)
    if "checkpoint" in args and args["checkpoint"] is not None:
        Print(" ".join(['checkpoint: %s' % (args["checkpoint"])]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['log_file:', str(output.name)]), output, newline=True)
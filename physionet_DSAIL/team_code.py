#!/usr/bin/env python

# Edit this script to add your team's training code.
# Some functions are *required*, but you can edit most parts of the required functions, remove non-required functions, and add your own functions.

################################################################################
#
# Imported functions and variables
#
################################################################################
import sys
import os
import numpy as np
from collections import OrderedDict
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import time
import torch

import src.config as config
from src.data import get_dataset_from_configs, collate_into_list, get_loss_weights_and_flags
from src.model.model_utils import get_model, get_profile
from src.train import Trainer
from src.utils import set_seeds


################################################################################
#
# Training model function
#
################################################################################

# Train your model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def training_code(data_directory, model_directory):
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

    for num_leads in [2,3,4,6,12]:
        if os.path.exists('./dataset_preprocessed'):
            os.system("rm -rf ./dataset_preprocessed")
        print('Training for %dleads...' % num_leads)
        data_cfg = config.DataConfig("config/cv-%dleads.json" % num_leads)
        preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
        model_cfg = config.ModelConfig("config/model.json")
        run_cfg = config.RunConfig("config/run.json")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print('Loading data...')
        set_seeds(2021)
        #data_cfg.filenames = get_filenames_from_split(data_cfg, "all", "train", data_directory)
        dataset_train = get_dataset_from_configs(data_cfg, preprocess_cfg, data_directory, split_idx = "train")
        #data_cfg.filenames = get_filenames_from_split(data_cfg, "all", "val", data_directory)
        dataset_val = get_dataset_from_configs(data_cfg, preprocess_cfg, data_directory, split_idx="val")
        iterator_train = torch.utils.data.DataLoader(dataset_train, run_cfg.batch_size, #collate_fn=collate_into_list,
                                                     shuffle=True, pin_memory=True, num_workers=4)
        iterator_val = torch.utils.data.DataLoader(dataset_val, run_cfg.batch_size, collate_fn=collate_into_list,
                                                   shuffle=False, pin_memory=True, num_workers=4)
        print("training samples: %d" % len(dataset_train))
        print("evaluation samples: %d" % len(dataset_val))

        ## initialize a model
        model, params = get_model(model_cfg, len(data_cfg.leads), len(data_cfg.scored_classes))
        get_profile(model, len(data_cfg.leads), data_cfg.chunk_length)

        ## setup trainer configurations
        loss_weights_and_flags = get_loss_weights_and_flags(data_cfg, dataset_train)
        trainer = Trainer(model, data_cfg, loss_weights_and_flags)
        trainer.set_device(device)
        trainer.set_optim_scheduler(run_cfg, params, len(iterator_train))

        ## train a model
        print('Training model...')
        PNC_list = []
        for epoch in range(run_cfg.num_epochs):
            stime = time.time()
            for B, batch in enumerate(iterator_train):
                trainer.train(batch, device)
                if B % 5 == 0: print('# epoch [{}/{}] train {:.1%}'.format(
                    epoch + 1, run_cfg.num_epochs, B / len(iterator_train)), end='\r', file=sys.stderr)
            print(' ' * 150, end='\r', file=sys.stderr)
            print('TIME: ', time.time() - stime)

            for B, batch in enumerate(iterator_val):
                trainer.evaluate(batch, device, "val")
                if B % 5 == 0: print('# epoch [{}/{}] val {:.1%}'.format(
                    epoch + 1, run_cfg.num_epochs, B / len(iterator_val)), end='\r', file=sys.stderr)
            print(' ' * 150, end='\r', file=sys.stderr)

            ## print log and save models
            trainer.epoch += 1
            torch.save(trainer.model.state_dict(),
                       os.path.join(model_directory, '%dleads_model_epoch%d.pt' % (len(data_cfg.leads), epoch)))
            trainer.logger_val.evaluate(trainer.scored_classes, trainer.normal_class, trainer.confusion_weight)
            PNC_list.append(float(trainer.logger_val.log[-1]))
            print("Epoch: %03d   PNC: %s" %(epoch, trainer.logger_val.log[-1]))
            trainer.log_reset()

        ## save a model
        print('Saving model...')
        model = os.path.join(model_directory, '%dleads_model_epoch*.pt' % (len(data_cfg.leads)))
        best_model = os.path.join(model_directory, '%dleads_model_epoch%d.pt' %(len(data_cfg.leads), int(np.argmax(PNC_list))))
        final_model = os.path.join(model_directory, '%dleads_model.pt' % (len(data_cfg.leads)))
        os.system("cp %s %s" % (best_model, final_model))
        os.system("rm -rf %s" % model)
        if (len(data_cfg.leads)==12):
            os.system("rm -rf ./dataset_preprocessed")

################################################################################
#
# Running trained model function
#
################################################################################

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def run_model(model, header, recording):
    data_cfg, preprocess_cfg, run_cfg, models = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg.data = recording.astype('float32')
    data_cfg.header = header.split("\n")
    dataset_val = get_dataset_from_configs(data_cfg, preprocess_cfg, "")
    iterator_val = torch.utils.data.DataLoader(dataset_val, 1, collate_fn=collate_into_list)

    outputs_list = []
    for model_ in models:
        loss_weights = get_loss_weights_and_flags(data_cfg)
        trainer = Trainer(model_, data_cfg, loss_weights)
        trainer.set_device(device)

        for B, batch in enumerate(iterator_val):
            trainer.evaluate(batch, device)

        outputs = trainer.logger_test.scalar_outputs[0][0]
        outputs_list.append(outputs)

    classes = data_cfg.scored_classes
    num_classes = len(classes)
    labels = np.zeros(num_classes, dtype=int)
    probabilities = np.zeros(num_classes)
    for i in range(num_classes):
        for j in range(len(outputs_list)):
            probabilities[i] += outputs_list[j][i]
        probabilities[i] = probabilities[i] / len(outputs_list)
        if probabilities[i] > 0.5: labels[i] = 1

    return classes, labels, probabilities

################################################################################
#
# File I/O functions
#
################################################################################

# Load a trained model. This function is *required*. You should edit this function to add your code, but do *not* change the arguments of this function.
def load_model(model_directory, leads):
    # load the model from disk
    data_cfg = config.DataConfig("config/cv-%dleads.json" % len(leads))
    preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
    model_cfg = config.ModelConfig("config/model.json")
    run_cfg = config.RunConfig("config/run.json")

    models = []
    model, _ = get_model(model_cfg, len(data_cfg.leads), len(data_cfg.scored_classes))
    checkpoint = torch.load(os.path.join(model_directory, '%dleads_model.pt' % len(data_cfg.leads)),
                            map_location=torch.device("cpu"))
    state_dict = OrderedDict()
    for k, v in checkpoint.items():
        if k.startswith("module."): k = k[7:]
        state_dict[k] = v
    model.load_state_dict(state_dict, strict=False)
    models.append(model)

    eval_list = [data_cfg, preprocess_cfg, run_cfg, models]

    return eval_list

# Define the filename(s) for the trained models. This function is not required. You can change or remove it.

def get_filenames_from_split(data_cfg, dataset_idx, split_idx, data_directory):
    """ get filenames from config file and split index """
    filenames_all = []
    path = data_directory
    for d, dataset in enumerate(data_cfg.datasets):
        if dataset_idx not in ["all", dataset]: continue

        if split_idx in ["train", "val"]:
            filenames = data_cfg.split[dataset][split_idx]

        filenames_all += [path + "/%s/%s" % (dataset, filename) for filename in filenames]
    return filenames_all


################################################################################
#
# Feature extraction function
#
################################################################################

# Extract features from the header and recording. This function is not required. You can change or remove it.


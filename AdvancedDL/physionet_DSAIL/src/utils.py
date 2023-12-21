# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

""" Utility functions """

import os
import sys
import time
import random
import datetime
import subprocess
import numpy as np
from gpuinfo import GPUInfo

import torch
from torch.utils.tensorboard import SummaryWriter


def Print(string, output, newline=False, timestamp=True):
    """ print to stdout and a file (if given) """
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else: 
        time = None
        line = string

    print(line, file=sys.stderr)
    if newline: print("", file=sys.stderr)

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)

    output.flush()
    return time


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_output(args, string):
    """ set output configurations """
    output, writer, save_prefix = sys.stdout, None, None
    if args["output_path"] is not None:
        save_prefix = args["output_path"]
        if not os.path.exists(save_prefix):
            os.makedirs(save_prefix, exist_ok=True)
        output = open(args["output_path"] + "/" + string + ".txt", "a")
        if "eval" not in string:
            tb = args["output_path"] + "/tensorboard/"
            if not os.path.exists(tb):
                os.makedirs(tb, exist_ok=True)
            writer = SummaryWriter(tb)

    return output, writer, save_prefix


def count_gpu_process(num_gpu):
    """ count running gpu process """
    info = GPUInfo.get_info()
    gpu_process = [0] * num_gpu
    for k, v in info[0].items():
        for g in v: gpu_process[int(g)] += 1
    return gpu_process


def run_commands(commands, gpu_process_limits, conda, wait_seconds=3):
    """ run commands in que whenever required_gpu is available """
    num_gpu = len(gpu_process_limits)
    gpu_runs_buffer = [0] * num_gpu
    gpu_runs_info = []
    for c, (idx, command, required_gpu) in enumerate(commands):
        while 1:
            run, available_gpu, device = False, 0, ""

            # check available_gpu
            gpu_process = count_gpu_process(num_gpu)
            for d in range(num_gpu):
                if gpu_process[d] + gpu_runs_buffer[d] < gpu_process_limits[d]:
                    if device == "": device += "%s"  % str(d)
                    else:            device += ",%s" % str(d)
                    available_gpu += 1
                    if available_gpu == required_gpu: break

            # run command if required_gpu is available
            if available_gpu == required_gpu:
                for d in device.split(","):
                    gpu_runs_buffer[int(d)] += 1
                gpu_runs_info.append([datetime.datetime.now(), device])

                print("\t".join(["%4d" % c, idx, "GPU%s" % device]))
                file = open("run.sh", "w")
                file.write("#!/bin/bash\n")
                file.write("source %s\n" % conda)
                file.write("CUDA_VISIBLE_DEVICES=%s " % device + command)
                file.close()
                subprocess.Popen(["./run.sh"], shell=True)
                time.sleep(3)
                break

            # check gpu_runs and sleep
            else:
                gpu_runs_info_new = []
                for run_time, device in gpu_runs_info:
                    # remove from the buffer if running_time is over than wait_seconds
                    if datetime.datetime.now() - run_time > datetime.timedelta(seconds=wait_seconds):
                        for d in device.split(","):
                            gpu_runs_buffer[int(d)] -= 1
                    else:
                        gpu_runs_info_new.append([run_time, device])
                gpu_runs_info = gpu_runs_info_new
                time.sleep(wait_seconds)

    print("DONE")


def get_training_results(file_idx, oracle=False, delete_checkpoints=False):
    """ get results (best validation) """
    if not os.path.exists(file_idx): return -1, -1, -1, -1, False

    FILE = open(file_idx, "r")
    lines = FILE.readlines()
    FILE.close()

    VAL_F1, VAL_CS, TEST_F1, TEST_CS, DONE = [], [], [], [], False
    for line in lines:
        tokens = line.strip().split("\t")
        if len(tokens) > 5 and tokens[5] == "val":
            VAL_F1.append(float(tokens[8]))
            VAL_CS.append(float(tokens[11]))
        if len(tokens) > 13 and tokens[13] == "test":
            TEST_F1.append(float(tokens[16]))
            TEST_CS.append(float(tokens[19]))
        if len(tokens) > 1 and tokens[1] == "end training a model":
            DONE = True

    if not oracle:
        best_epoch = np.argmax(VAL_CS) if len(VAL_CS) > 0 else -1
    else:
        best_epoch = np.argmax(TEST_CS) if len(TEST_CS) > 0 else -1
    val_f1 = VAL_F1[best_epoch] if best_epoch != -1 and len(VAL_F1) > best_epoch else -1
    val_cs = VAL_CS[best_epoch] if best_epoch != -1 and len(VAL_CS) > best_epoch else -1
    test_f1 = TEST_F1[best_epoch] if best_epoch != -1 and len(TEST_F1) > best_epoch else -1
    test_cs = TEST_CS[best_epoch] if best_epoch != -1 and len(TEST_CS) > best_epoch else -1
    if DONE:
        path = os.path.split(file_idx)[0]
        if not oracle:
            if not os.path.exists("%s/best.pt" % (path)):
                os.system("cp %s/checkpoints/%d.pt %s/best.pt" % (path, best_epoch + 1, path))
        else:
            if not os.path.exists("%s/oracle_best.pt" % (path)):
                os.system("cp %s/checkpoints/%d.pt %s/oracle_best.pt" % (path, best_epoch + 1, path))
        if delete_checkpoints:
            os.system("rm %s/checkpoints/*.pt" % (path))
            os.system("mv %s/best.pt %s/checkpoints/." % (path, path))

    return val_f1, val_cs, test_f1, test_cs, DONE

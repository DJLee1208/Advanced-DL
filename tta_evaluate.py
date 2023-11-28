import torch, os, shutil
import numpy as np
from tqdm import tqdm

#import data_aggregation
import src.config as config
from src.data import get_dataset_from_configs, collate_into_list, get_loss_weights_and_flags
from src.model.model_utils import get_model, get_profile
from team_code import load_model
from src.model.model import ECG_model
from tent import tent, norm
import torch.nn.functional as F
from src.evaluate import *


def evaluate_tta(source, target):
    """
    Evaluate the performance of the model trained on source data using TTA method.
    method : source, 
    """
    print(f"Source: {source}, Target: {target}")
    if not os.path.exists(f'results/{source}-{target}'):
        os.mkdir(f'results/{source}-{target}')
    
    # leads configuration
    #leads_cfg = [2, 3, 4, 6, 12]
    leads_cfg = [12]
    
    for num_leads in leads_cfg:
        print(f" ### Leads: {num_leads}")
        # set target dataset
        fdir = 'data_aggr'
        if os.path.exists(fdir):
            shutil.rmtree(fdir)
        os.mkdir(fdir)

        # Copy files of the target data to the dataset folder
        print('Copying data...', end='')
        copy_files(source_dir = f"./../dataset/{target}/", goal_dir = f"{fdir}/", file_extensions = [".hea", ".mat"])
        print('Done.')
        
        # load configurations
        print('Loading configurations...', end='')
        data_cfg = config.DataConfig(f"config/{target}/cv-{num_leads}leads.json")
        preprocess_cfg = config.PreprocessConfig("config/preprocess.json")
        model_cfg = config.ModelConfig("config/model.json")
        run_cfg = config.RunConfig("config/run.json")
        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
        print('Done.') 
        
        # load train set
        print('Loading train set & model...', end='')
        data_directory = fdir
        dataset_train = get_dataset_from_configs(data_cfg, preprocess_cfg, data_directory, split_idx = "train")

        # Iterator
        iterator_train = torch.utils.data.DataLoader(dataset_train, run_cfg.batch_size, #collate_fn=collate_into_list,
                                                            shuffle=True, pin_memory=True, num_workers=4)
        
        
        ## initialize a model
        model, params = get_model(model_cfg, len(data_cfg.leads), len(data_cfg.scored_classes))
        get_profile(model, len(data_cfg.leads), data_cfg.chunk_length)

        state_dict = torch.load(f'model/{source}/{num_leads}leads_model.pt')
        model.load_state_dict(state_dict)

        model = tent.configure_model(model)
        params, param_names = tent.collect_params(model)
        optimizer = torch.optim.Adam([{'params': params[0], 'weight_decay': run_cfg.weight_decay},
                                    {'params': params[1], 'weight_decay': 0}])
        print('Done')
        
        ### source method : use base model trained by source dataset
        print('1. Baseline...')
        fpath = f'{source}-{target}/base_{num_leads}_leads'
        evaluate_test(model, iterator_train, fpath)
        
        ### norm method : adjusting batch normalization on test batch
        # norm model
        print('2. Norm method...')
        norm_model = norm.Norm(model)
        fpath = f'{source}-{target}/norm_{num_leads}_leads'
        evaluate_test(norm_model, iterator_train, fpath)

        ### tent method (episodic) : tent method episodic adaptation
        # https://github.com/DequanWang/tent
        # model is reset after each test batch
        print('3. Tent - episodic method...')
        tent_episodic = True
        tented_model = tent.Tent(model, optimizer, episodic=tent_episodic)
        fpath = f'{source}-{target}/tent-episodic_{num_leads}_leads'
        evaluate_test(tented_model, iterator_train, fpath)

        ### tent method (online) : tent method online adaptation
        print('4. Tent - online method...')
        tent_episodic = False
        tented_model = tent.Tent(model, optimizer, episodic=tent_episodic)
        fpath = f'{source}-{target}/tent-online_{num_leads}_leads'
        evaluate_test(tented_model, iterator_train, fpath)
        
        ### memo method
        #print('5. Memo method...')
        
    
# Inference
def evaluate_test(model, iterator_train, fpath): 
    scalars, binaries, true_labels = [], [], []
    for B, batch in enumerate(tqdm(iterator_train)):
        inputs, features, labels = batch
        
        # prediction
        outputs = model(inputs, features)
        
        scalar_outputs = torch.sigmoid(outputs)
        binary_outputs = scalar_outputs > 0.5
        
        # change outputs to numpy
        scalar_outputs = scalar_outputs.detach().numpy()
        binary_outputs = binary_outputs.detach().numpy()
        
        # ground-truth labels
        labels = labels.detach().numpy()    
        
        scalars.extend(scalar_outputs)
        binaries.extend(binary_outputs)
        true_labels.extend(labels)
       
    # save outputs 
    if not os.path.exists('results'):
        os.mkdir('results')
    np.savez(f'results/{fpath}.npz', scalar = np.array(scalars), binary = np.array(binaries), labels = np.array(true_labels))

    # evaluate and save results
    evaluate_score(np.array(scalars), np.array(binaries), np.array(true_labels), fpath)
    
    
# Evaluate the performance and save the results
def evaluate_score(scalar_outputs, binary_outputs, labels, fpath):
    weights_file = 'config/weights.csv'
    sinus_rhythm = set(['426783006'])
    classes, weights = load_weights(weights_file)

    # Evaluate the model by comparing the labels and outputs.
    print('Evaluating model...')

    print('- AUROC and AUPRC...')
    auroc, auprc, auroc_classes, auprc_classes = compute_auc(labels, scalar_outputs)
    print(f'auroc {auroc:.3f}, auprc {auprc:.3f}')

    print('- Accuracy...')
    accuracy = compute_accuracy(labels, binary_outputs)
    print(f'acc {accuracy:.3f}')

    print('- F-measure...')
    f_measure, f_measure_classes = compute_f_measure(labels, binary_outputs)
    print(f'f score {f_measure:.3f}')

    print('- Challenge metric...')
    challenge_metric = compute_challenge_metric(weights, labels, binary_outputs, classes, sinus_rhythm)
    print(f'challenge metric: {challenge_metric}')
    
    with open(f"results/{fpath}_{challenge_metric:.3f}.txt", 'w') as f:
        f.write(f'auroc {auroc:.3f}, auprc {auprc:.3f}')
        f.write(f'acc {accuracy:.3f}')
        f.write(f'f score {f_measure:.3f}')
        f.write(f'challenge metric: {challenge_metric}')
    
    print('Done.')
    
    
def copy_files(source_dir, goal_dir, file_extensions):
    """
    Aggregates files with specified extensions from source_dir (including subfolders)
    and moves them to goal_dir. It does not copy files that already exist in goal_dir.

    :param source_dir: Directory to search for files.
    :param goal_dir: Directory to move the files to.
    :param file_extensions: List of file extensions to look for.
    """
    
    os.makedirs(goal_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(tuple(file_extensions)):
                source_file_path = os.path.join(root, file)
                goal_file_path = os.path.join(goal_dir, file)

                if not os.path.exists(goal_file_path):
                    shutil.copy2(source_file_path, goal_file_path)
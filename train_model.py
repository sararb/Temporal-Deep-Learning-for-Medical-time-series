import os
import sys 
from functools import partial
import numpy as np
import torch
import argparse
from pathlib import Path
import datetime
import glob
from sklearn.model_selection import KFold

from data.dataset import SeqDataset, FastDataBunch, collate_seq

from fastai.basic_train import Learner
from fastai.train import fit_one_cycle
from fastai.callbacks import CSVLogger

import wandb
from wandb.fastai import WandbCallback

from models.model import Hba1cModel
from optimization.radam import RAdam
from metrics import *
from loss import weighted_ner_loss
from callbacks import CustomSaveModelCallback
from do_test import get_predictions
from utils import load_cfg

# import wandb
# from wandb.fastai import WandbCallback

import random
import pickle 


def set_seeds(bag_number=1, seed=42, seed_np=12345, seed_torch=1234):
    random.seed(bag_number+seed)
    np.random.seed(bag_number+seed_np)
    torch.manual_seed(bag_number+seed_torch)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data(cfg, trunc_max_len, use_valid, do_cv=False, cv_fold=5):
    ####################################################################################################################
    #                                       Setting data path parameters                                               #
    ####################################################################################################################
    # set cache directory
    data_dir = cfg['data_directory']
    ####################################################################################################################
    #                                         Setting Fast.ai Databunch object                                         #
    ####################################################################################################################
    # configure dataset object
    train_file = cfg['train_file_name']
    test_file = cfg['test_file_name']
    valid_file = cfg['validation_file_name']
    train_data_file = os.path.join(data_dir, train_file)
    test_data_file = os.path.join(data_dir, test_file)
    valid_data_file = os.path.join(data_dir, valid_file)
    # batch size 
    batch_size_per_gpu = cfg['batch_size']
    # single Train / Test 
    if not do_cv: 
        if use_valid: 
            train_data_file = [train_data_file, valid_data_file]
        # Train dataset
        train_dataset = SeqDataset([train_data_file])
        # test dataset
        test_dataset = SeqDataset([valid_data_file])
        # Fast.ai databunch
        collate = partial(collate_seq, pad_value = -0.5, trunc_max_len = trunc_max_len,
                        side_info = cfg['side_info'])
        data = FastDataBunch.create(train_dataset,
                                    test_dataset,
                                    num_workers = cfg['n_workers'],
                                    bs = batch_size_per_gpu,
                                    val_bs = cfg['val_bs'],
                                    collate_fn = collate,
                                    device = cfg['device'])
        return [data]
    else: 
        # load and merge train and validation data if use_valid set to True  
        with open(train_data_file, "rb") as handle:
                train_patients = pickle.load(handle)
                handle.close()
        if use_valid: 
            with open(valid_data_file, "rb") as handle:
                    train_patients += pickle.load(handle)
                    handle.close()
        # load test data 
        with open(test_data_file, "rb") as handle:
            test_patients = pickle.load(handle)
            handle.close()
        # create k-fold split of train data 
        kf = KFold(n_splits=cv_fold, random_state=42)
        train_patients = np.array(train_patients)
        # build list of databunch 
        kfold_data = []
        for train_index, valid_index in kf.split(np.array(train_patients)):
            # Train dataset
            train_dataset = SeqDataset(list(train_patients[train_index]))
            # test dataset
            valid_dataset = SeqDataset(list(train_patients[valid_index]))
            # Fast.ai databunch
            collate = partial(collate_seq, pad_value = -0.5, trunc_max_len = trunc_max_len,
                            side_info = cfg['side_info'])
            data = FastDataBunch.create(train_dataset,
                                        valid_dataset,
                                        num_workers = cfg['n_workers'],
                                        bs = batch_size_per_gpu,
                                        val_bs = cfg['val_bs'],
                                        collate_fn = collate,
                                        device = cfg['device'])
            kfold_data.append(data)
        return kfold_data
            

def run_train(cfg, data, fold, trunc_max_len, bag_number=1,  do_test=False, find_best_lr=False):
    """
    Run training for specific fold

    Arguments:
        trunc_max_len:
        cfg: setting parameters loaded from yaml file
        do_test: whether to build prediction for test data or not
    Saves :
        - Train log file
        - model checkpoint
        - CSV history track
    """

    run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

    ####################################################################################################################
    #                                             Setting the model                                                    #
    ####################################################################################################################

    model = Hba1cModel(cfg).cuda()

    ####################################################################################################################
    #                                             Setting Optimizer                                                    #
    ####################################################################################################################
    # Set optimizer
    if cfg['optimizer'] == 'radam':
        optimizer = partial(RAdam, weight_decay = cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = partial(torch.optim.SGD, weight_decay = cfg['weight_decay'])
    elif cfg['optimizer'] == 'adam':
        optimizer = partial(torch.optim.Adam, weight_decay = cfg['weight_decay'])
    else:
        raise ValueError("optimizer should be one of the following: [sgd, radam, adam]")
    print('\tOptimizer: %s\n' % cfg['optimizer'])

    ####################################################################################################################
    #                                 Create directories where to save results                                           #
    ####################################################################################################################
    experiment_name = "algorithm_%s_min_meas_%s_max_meas_%s_side_features_%s_" % (cfg['name'], cfg['min_measurement'], trunc_max_len,
                                                                '_'.join(cfg['side_info']))
    model_dir = os.path.join(cfg['result_dir'],
                      cfg['model_type'])
    Path(model_dir).mkdir(exist_ok = True)
    
    Path(os.path.join(model_dir,experiment_name)).mkdir(exist_ok = True)
    Path(os.path.join(model_dir,experiment_name, str(bag_number))).mkdir(exist_ok = True)
    experiment_dir = os.path.join(model_dir,
                                experiment_name, 
                                str(bag_number),
                                str(fold))
    Path(experiment_dir).mkdir(exist_ok = True)

    checkpoint_dir=os.path.join(experiment_dir, 'checkpoints')
    Path(checkpoint_dir).mkdir(exist_ok = True)
    
    ####################################################################################################################
    #                                                 Setting the Learner                                              #
    ####################################################################################################################
    # set wandb callback 

    # Set additional wandb config
    #wandb_config = cfg
    #wandb_config['trunc_max_len'] = trunc_max_len
    #wandb_config['fold'] = fold
    #wandb_config['time_mode'] = cfg['temporal_model']['model_time']
    # Init wandb
    #wandb.init(project = "final_results_experiment",
    #            entity = "sararb",
    #            name = f"fold_%s" %fold,
    #            group = "algorithm_%s_min_meas_%s_max_meas_%s_side_features_%s_" % (cfg['name'], cfg['min_measurement'], trunc_max_len,
    #                                                            '_'.join(cfg['side_info'])),
    #            config = wandb_config,
    #            reinit = True)
    # wandb.init(project = "temporal_hba1c", entity = "sararb")
    
    
    # Set Fast.ai learner
    learn = Learner(data = data,
                    model = model,
                    opt_func = optimizer,
                    metrics = [flat_accuracy, recall_m, precision, f1_score_m, auc_score],
                    silent = False,
                    model_dir = checkpoint_dir,
                    #callback_fns=partial(WandbCallback, log="parameters", save_model=False)
                    )

    ####################################################################################################################
    #                                             Setting Loss function                                                #
    ####################################################################################################################
    loss_name = 'cross_entropy'
    print('\tCriterion: %s\n' % loss_name)
    class_weight = np.array(cfg['class_weight'])
    class_weight = torch.tensor(class_weight, device = cfg['device']).float()
    learn.loss_func = weighted_ner_loss(weights = class_weight).mlm_loss

    ####################################################################################################################
    #                                               Seting Callbacks                                                   #
    ####################################################################################################################
    # set result directory for saving .csv metrics history and model checkpoints 
    callbacks = [CSVLogger(learn,
                           filename = os.path.join(experiment_dir,
                                                'history_track'),

                           append = True),
                 CustomSaveModelCallback(learn,
                                         every = 'improvement',
                                         monitor = cfg['monitor'],
                                         logger = None,
                                         mode = cfg['mode'],
                                         name = cfg['model_type']),
                 ]

    ####################################################################################################################
    #                                                 Getting Best LR                                                  #
    ####################################################################################################################
    print('find best lr')
    if find_best_lr:
        learn.lr_find()
        fig = learn.recorder.plot(return_fig=True)
        fig.savefig(os.path.join(model_dir, 'learning_rate_finder.png'))
        sys.exit(0)

    ####################################################################################################################
    #                                             Launch one cycle learning                                            #
    ####################################################################################################################
    fit_one_cycle(learn,
                  cyc_len = cfg['cycle_len'],
                  max_lr = cfg['max_lr'],
                  callbacks = callbacks)

    ####################################################################################################################
    #                                            Computing Test set predictions                                        #
    ####################################################################################################################
    if do_test:
        get_predictions(cfg, cfg['model_type'], learn.model, save_dir = experiment_dir,
                        trunc_max_len = trunc_max_len)


def main(config_directory, trunc_max_len, bag_number=1,  use_valid=False, do_cv=False, cv_fold=10, do_test=True):
    config_files = glob.glob(os.path.join(config_directory, '*.yaml'))
    for config_file in config_files:
        config = load_cfg(config_file)
        print("\n Gather results for model %s with time_representation %s: \n" %(config['model_type'], config['name']))
        fold_data = get_data(config, trunc_max_len, use_valid=use_valid, do_cv=do_cv, cv_fold=cv_fold)
        for i, data in enumerate(fold_data):
            run_train(config, data, i+1, trunc_max_len, bag_number=bag_number, do_test = do_test)


if __name__ == '__main__':
    print("Start final test....")
    parser = argparse.ArgumentParser(description = 'Launch model training.')
    parser.add_argument('--config', type = str, help = 'The path to the config directory')
    parser.add_argument('--trunc_max_len', type = int, default=151,
                        help = 'The maximum number of measurements to take into account for each patient')
    parser.add_argument('--test', default = False, action = 'store_true',
                        help = 'whether to build the test predictions or not',
                        required = False)
    parser.add_argument('--use_valid', default = False, action = 'store_true',
                        help = 'whether to use validation data in train or not',
                        required = False)  
    parser.add_argument('--do_cv', default = False, action = 'store_true',
                        help = 'whether to use k-fold training or not',
                        required = False)   
    parser.add_argument('--kfolds', type=int,
                        help = 'number of splits for k-fold training ',
                        required = False)  
    parser.add_argument('--bag_number', type=int, default=1,
                        help = 'random seed when kfold is impossible',
                        required = False)   

    args = parser.parse_args()
    set_seeds(bag_number=args.bag_number)

    main(config_directory=args.config,
        bag_number=args.bag_number,
        trunc_max_len = args.trunc_max_len,
        use_valid=args.use_valid,
        do_cv=args.do_cv,
        cv_fold=args.kfolds,
        do_test = args.test)

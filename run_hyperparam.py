import optuna
import copy
import os
from functools import partial
import torch
import wandb
from wandb.fastai import WandbCallback
import argparse


from data.dataset import SeqDataset, FastDataBunch, collate_seq

from fastai.basic_train import Learner
from fastai.train import fit_one_cycle

from optimization.radam import RAdam
from metrics import *
from loss import weighted_ner_loss
from utils import load_cfg
from models.model import Hba1cModel

import random
import numpy as np
min_measurements = 5


data_paths = {3: ["965_patients_min_3_seq_infos_Train.sav", "121_patients_min_3_seq_infos_Valid.sav", 121],
              5: [ "836_patients_min_5_seq_infos_Train.sav"  ,"105_patients_min_5_seq_infos_Valid.sav", 105],
              10: [ "689_patients_min_10_seq_infos_Train.sav"  ,"87_patients_min_10_seq_infos_Valid.sav", 87],
              15: ["552_patients_min_15_seq_infos_Train.sav", "69_patients_min_15_seq_infos_Valid.sav", 69]  
}

data_paths_1 = {3: ["1026_patients_min_3_seq_infos_Train.sav", "129_patients_min_3_seq_infos_Valid.sav", 129],
              5: [ "892_patients_min_5_seq_infos_Train.sav"  ,"112_patients_min_5_seq_infos_Valid.sav", 112],
              10: [ "732_patients_min_10_seq_infos_Train.sav"  ,"92_patients_min_10_seq_infos_Valid.sav", 92],
              15: ["609_patients_min_15_seq_infos_Train.sav", "77_patients_min_15_seq_infos_Valid.sav", 77]  
}

def set_seeds():
    random.seed(42)
    np.random.seed(12345)
    torch.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_data_config(config, min_measurement, max_len,  side_info_mode):
    # set how to leverage side info 
    if side_info_mode == 'seq_concat': 
        config['use_patient_info'] = True
        config['classifier']['use_patient_info'] = False 

    if side_info_mode == 'classifier_concat': 
        config['use_patient_info'] = False
        config['classifier']['use_patient_info'] = True 

    elif side_info_mode == 'not_used': 
        config['use_patient_info'] = False
        config['classifier']['use_patient_info'] = False 

    # set data paths 
    config['train_file_name'] = data_paths[min_measurement][0]
    config['validation_file_name'] = data_paths[min_measurement][1]
    config['val_bs']  = data_paths[min_measurement][2]
    config['min_measurement'] = min_measurement
    config['max_len'] = max_len
    return config 
    

def get_param_from_trial(trial, cfg): 
    # log the params to optimize to wandb
    params = copy.deepcopy(cfg)
    wandb_config = {}
    for param_name, values in params.items():
        if param_name in ['patient_config', 'time_config', 'event_config', 'temporal_model', 'classifier']:
            for param, nested_value in values.items():
                if isinstance(nested_value, list) and (len(nested_value) > 1) and (nested_value[0] == 'optimize'):  
                        if nested_value[1] == 'float': 
                            params[param_name][param] = trial.suggest_discrete_uniform(param, nested_value[2], nested_value[3], nested_value[4])
                        
                        elif nested_value[1] == 'int': 
                            params[param_name][param] = trial.suggest_int(param, nested_value[2], nested_value[3], step = nested_value[4])
                        
                        elif nested_value[1] == 'uniform': 
                            params[param_name][param] = trial.suggest_uniform(param, nested_value[2], nested_value[3])

                        elif nested_value[1] == 'categorical': 
                            params[param_name][param] = trial.suggest_categorical(param, nested_value[2:])
                        wandb_config[param_name+'_'+param] = params[param_name][param]

        elif isinstance(values, list) and (len(values) > 1) and (values[0] == 'optimize'): 
            if values[1] == 'float': 
                params[param_name] = trial.suggest_discrete_uniform(param_name, values[2], values[3], values[4])
            
            elif values[1] == 'int': 
                params[param_name] = trial.suggest_int(param_name, values[2], values[3], step = values[4])
            
            elif values[1] == 'uniform': 
                params[param_name] = trial.suggest_loguniform(param_name, values[2], values[3])

            elif values[1] == 'categorical': 
                params[param_name] = trial.suggest_categorical(param_name, values[2:])
            
            wandb_config[param_name] = params[param_name]

    params['event_config']['continuous_hidden_dims'] = [params['event_config']['continuous_hidden_dims']]
    params['time_config']['hidden_dims'] = [params['time_config']['hidden_dims']]
    params['patient_config']['hidden_dims'] = [params['patient_config']['hidden_dims']]
    return params, wandb_config


def load_data(cfg, trunc_max_len):
    # set cache directory
    data_dir = cfg['data_directory']
    # configure dataset object
    train_file = cfg['train_file_name']
    val_file = cfg['validation_file_name']
    train_data_file = os.path.join(data_dir, train_file)
    val_data_file = os.path.join(data_dir, val_file)
    # break mode to build tokens blocs
    batch_size_per_gpu = cfg['batch_size']
    # Train dataset
    train_dataset = SeqDataset(train_data_file)
    # Validation dataset
    val_dataset = SeqDataset(val_data_file)
    # Fast.ai databunch
    collate = partial(collate_seq, pad_value = cfg['temporal_model']['pad_value'], trunc_max_len=trunc_max_len,
                      side_info = cfg['side_info'])

    data = FastDataBunch.create(train_dataset,
                                val_dataset,
                                num_workers = cfg['n_workers'],
                                bs = batch_size_per_gpu,
                                val_bs = cfg['val_bs'],
                                collate_fn = collate,
                                device = cfg['device'])
    return data


def get_learner(trial, cfg, min_measurement, trunc_max_len, side_info_mode):
    """

    :return:
    """
    # set params of config from trial 
    cfg = set_data_config(cfg, min_measurement, trunc_max_len,  side_info_mode)
    cfg, wandb_config = get_param_from_trial(trial, cfg)
    # load data
    data = load_data(cfg, trunc_max_len)
    # Generate the model architecture
    model = Hba1cModel(cfg).cuda()
    # Set optimizer
    if cfg['optimizer'] == 'radam':
        optimizer = partial(RAdam, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
        optimizer = partial(torch.optim.SGD, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adam':
        optimizer = partial(torch.optim.Adam, weight_decay=cfg['weight_decay'])
    else:
        raise ValueError("optimizer should be one of the following: [sgd, radam, adam]")
    # Set training params
    max_lr = cfg['max_lr']
    cycle_len = cfg['cycle_len']
    weight_1 = cfg['class_weight_1']
    class_weight = torch.tensor([1-weight_1, weight_1], device=cfg['device'], dtype = torch.float)

    # Set additional wandb config
    wandb_config['min_measurement'] = cfg['min_measurement']
    wandb_config['trunc_max_len'] = trunc_max_len
    wandb_config['model_type'] = cfg['model_type']
    wandb_config['side_info'] = '_'.join(cfg['side_info'])
    wandb_config['time_mode'] = cfg['temporal_model']['model_time']
    wandb_config['side_info_mode'] = side_info_mode

    # Init wandb
    wandb.init(project = "hyper_parameter_experiments_def_v2_final_version",
                entity = "sararb",
                name = f"trial_%s" % trial.number,
                group = "%s_%s_min_meas_%s_side_features_%s_mode_%s" % (cfg['model_type'],
                                                                    cfg['name'],
                                                                    cfg['min_measurement'],
                                                                    '_'.join(cfg['side_info']), 
                                                                    side_info_mode
                                                                   ),
                config = wandb_config,
                reinit = True)
    # set learner 
    learn = Learner(data = data,
                    model = model,
                    opt_func = optimizer,
                    metrics = [flat_accuracy, recall_m, precision, f1_score_m, auc_score],
                    silent = False,
                    model_dir = 'config_files',
                    callback_fns=partial(WandbCallback, log="parameters", save_model=False))
    # Set loss criterion
    loss_name = 'cross_entropy'
    print('\tCriterion: %s\n' % loss_name)
    learn.loss_func = weighted_ner_loss(weights = class_weight).mlm_loss
    return learn, max_lr, cycle_len


def objective(trial, cfg, min_measurement, trunc_max_len, side_info_mode):
    # Generate the fast.ai learner object
    learn, max_lr, cycle_len = get_learner(trial, cfg, min_measurement, trunc_max_len, side_info_mode)
    # train the model
    fit_one_cycle(learn,
                  cyc_len = cycle_len,
                  max_lr = max_lr)
    # optimize the auc score
    return learn.recorder.metrics[-1][-2]


def main(cfg_file, min_measurement, side_info_mode,  trunc_max_len=151, n_trials=100):
    set_seeds()
    cfg = load_cfg(cfg_file)
    study = optuna.create_study(direction="maximize")
    study.optimize(partial(objective, cfg = cfg, trunc_max_len=trunc_max_len,
                        min_measurement = min_measurement, side_info_mode = side_info_mode),
                    n_trials=n_trials)
    print("Best trial:")
    trial = study.best_trial
    print("  positive F1-Score: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == '__main__':
    set_seeds()
    parser = argparse.ArgumentParser(description = 'Launch model training.')
    parser.add_argument('--config', type = str,  help = 'The path to the config file specifying params for hyper-param tuning')
    parser.add_argument('--trunc_max_len', type = int,
                        help = 'The maximum number of measurements to take into account for each patient')
    parser.add_argument('--min_measurement', type = int,
                        help = 'The minimum number of measurements to take into account for each patient, possible values are: [3 , 5, 10, 15]')
    parser.add_argument('--side_info_mode', type = str,
                        help = 'How to leverage patient information: [not_used, seq_concat, classifier_concat]')
    args = parser.parse_args()
    main(cfg_file=args.config, min_measurement=args.min_measurement, side_info_mode=args.side_info_mode, trunc_max_len=args.trunc_max_len)

from timeit import default_timer as timer
import warnings
from fastai.torch_core import model_type
import numpy as np
import pandas as pd
import torch
import pickle
from pathlib import Path
import argparse

import os
from data.dataset import SeqDataset, collate_seq
from functools import partial
from torch.utils.data import (
    DataLoader,
    RandomSampler)
import sys 
from models.model import Hba1cModel

from utils import load_cfg, time_to_str

warnings.filterwarnings("ignore")



def do_test(model, test_loader, test_len, use_gpu=True, save_dir='.', save_self_attention=False):
    """
    do_test -> Pre-compute test predictions probabilities and save the resulting dataframe 
    in a .csv file 

    Parameters
    ----------
        model (nn.module) : the trained pytorch model
        test_loader (torch.dataloader):  Test data loader
        use_gpu: whether to use GPU for testing of CPU
        test_len (int):  total number of test observations
        save_dir: path to the directory where to save .csv prediction file 
        save_attention: whether to save attention vectors self attention matrices
    """
    ####################################################################################################################
    #                                                  Init output objects                                             #
    ####################################################################################################################
    outputs, targets, lengths, patient_ids, attentions_vector, attentions_matrix = [], [], [], [], [], []
    probabilities = []
    hidden_rep = []

    num_batches = 0
    start = timer()
    test_num = 0
    if use_gpu:
        device = 'cuda'
    else:
        device = 'cpu'
    print("\nGather patient-level labels predictions...")

    ####################################################################################################################
    #                                                 Gather predictions                                               #
    ####################################################################################################################
    for b, (temporal_input, target, patients) in enumerate(test_loader):

        temporal_input, target = temporal_input.to(device), target.to(device)

        model.eval()
        with torch.no_grad():
            predict, weights = model(temporal_input)
            x_rep = model.patient_embedding(temporal_input)
            patient_representations, _ = model.time_model(x_rep, mask=None)
            print(patient_representations.shape)

        if save_self_attention:
            # Save the self attention matrix of transformer model : save latest transformer bloc output
            attentions_matrix.append(dict(zip(patients, weights[1][-1].data.cpu().numpy())))
            
        attentions_vector.append(dict(zip(patients, weights[0].data.cpu().numpy())))
        hidden_rep.append(dict(zip(patients, patient_representations.data.cpu().numpy())))

        batch_size = test_loader.batch_size

        proba = torch.softmax(predict, dim = 1).data.cpu().numpy()
        outputs.append(np.argmax(proba, axis = 1))
        probabilities.append(proba)

        #inputs.append(X.data.cpu().numpy())
        targets.append(target.data.cpu().numpy())
        patient_ids.append(patients)
        lengths.append(temporal_input.sequence_lengths.data.cpu().numpy())

        test_num += len(temporal_input.sequence_lengths)
        num_batches += batch_size
        print('\r %8d/%8d     %0.2f  %s' % (test_num, test_len, test_num / test_len,
                                            time_to_str(timer() - start, 'min')), end = '', flush = True)

    ####################################################################################################################
    #                                            Save predictions dataframe                                            #
    ####################################################################################################################
    print('\nSave predictions to .csv file...')
    targets = [target_ for batch in targets for target_ in batch]
    outputs = [out for batch in outputs for out in batch]
    probabilities = [out for batch in probabilities for out in batch]
    patients = [patient for batch in patient_ids for patient in batch]
    lengths = [len_ for batch in lengths for len_ in batch]
    result_frame = pd.DataFrame(targets, columns = ['true_labels'])
    result_frame['predictions'] = outputs
    result_frame['probability_1'] = probabilities
    result_frame['patient_id'] = patients
    result_frame['length'] = lengths
    result_frame.to_csv(os.path.join(save_dir, 'prediction_results.csv'), sep = ';', index = False)

    ####################################################################################################################
    #                                               Save attention weights                                             #
    ####################################################################################################################
    print("\nSave attention vector weights to pkl file...")
    attention_dict = {k: v for d in attentions_vector for k, v in d.items()}
    pickle.dump(attention_dict, open(os.path.join(save_dir, 'attention_weights.pkl'), 'wb'))

    print("\nSave patient hidden representation to pkl file...")
    patient_hidden_rep = {k: v for d in hidden_rep  for k, v in d.items()}
    pickle.dump(patient_hidden_rep, open(os.path.join(save_dir, 'patient_hidden_rep.pkl'), 'wb'))

    print('\nResults saved at %s' %save_dir)

    if save_self_attention:
        print("\nSave self attention matrix weights to pkl file...")
        self_attention = {k: v for d in attentions_matrix for k, v in d.items()}
        pickle.dump(self_attention, open(os.path.join(save_dir, 'self_attention_weights.pkl'), 'wb'))
        print('\nResults saved at %s' %save_dir)


def get_predictions(cfg, model_type, trained_model, save_dir, trunc_max_len):
    """x
        get_predictions -> Save predictions after training 

    Parameters: 
    -----------
    :param save_dir:
    :param cfg:
    :param trained_model:
    :param model_type:
    :param trunc_max_len:

    """
    print('=' * 10)
    print('\t Build predictions for %s model...\n' % model_type)
    print('=' * 10)

    data_dir = cfg['data_directory']
    test_file = cfg['test_file_name']
    test_data_file = os.path.join(data_dir, test_file)
    
    # Test dataset
    test_dataset = SeqDataset([test_data_file])

    collate = partial(collate_seq, pad_value = 0, side_info = cfg['side_info'],
                      trunc_max_len=trunc_max_len,  test = True)
    test_loader = DataLoader(test_dataset,
                             sampler = RandomSampler(test_dataset),
                             batch_size = cfg['val_bs'],
                             collate_fn = collate,
                             pin_memory = True)


    if model_type == 'attention':
        do_test(trained_model, test_loader, test_len = len(test_dataset),
                use_gpu = cfg['device'] == 'cuda',
                save_dir = save_dir,
                save_self_attention = True)
    else:
        do_test(trained_model, test_loader, test_len = len(test_dataset),
                use_gpu = cfg['device'] == 'cuda',
                save_dir = save_dir,
                save_self_attention = False)


def main(config_file, model_type, model_checkpoint, savedir, trunc_max_len=151):
    """
    :param config_file:
    :param model_type:
    :param model_checkpoint:
    :return:
    """
    cfg = load_cfg(config_file)
    # Define model's architecture
    model = Hba1cModel(cfg)
    # Load from checkpoint
    model.load_state_dict(torch.load(model_checkpoint))
    get_predictions(cfg, model_type, model.cuda(), savedir , trunc_max_len)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Build predictions from pre-trained model.')
    parser.add_argument('--config', type = str,  help = 'the path to the config file')
    parser.add_argument('--model_type', type = str, help = 'the model type to use')
    parser.add_argument('--checkpoint', type = str,  help = 'the trained model checkpoint')
    parser.add_argument('--save_dir', type = str, help = 'the path to the directory where to store predictions')
    parser.add_argument('--trunc_max_len', type = int, help = 'maximum length of the sequence')
    args = parser.parse_args()
    main(args.config, args.model_type, args.checkpoint, args.save_dir)


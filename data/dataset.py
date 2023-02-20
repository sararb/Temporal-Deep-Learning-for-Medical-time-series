import os
import torch
from pathlib import Path
import pickle
from typing import List, Dict

import numpy as np

from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
    RandomSampler,
    Dataset,
)


from fastai.core import Callable, defaults, listify, PathOrStr, List, ifnone
from fastai.basic_data import DeviceDataLoader
from torch.nn.utils.rnn import pad_sequence
from fastai.basic_data import DataBunch

from models.patient_embedding import TemporalInputs
#####################################################################################################################
#                                                                                                                   #
#                                           Tensor Text Dataset                                                     #
#                                                                                                                   #
#####################################################################################################################


class SeqDataset(Dataset):
    def __init__(self, data):
        """
        :param data: either path to pickle file or 
        """
        if isinstance(data, list) and isinstance(data[0], str): 
            self.patients = []
            for file_path in data:
                assert os.path.isfile(file_path)
                print("Loading patients features from cached file %s", file_path)
                with open(file_path, "rb") as handle:
                    self.patients += pickle.load(handle)
                    handle.close()
        elif isinstance(data, str):
            file_path = data 
            assert os.path.isfile(file_path)
            print("Loading patients features from cached file %s", file_path)
            with open(file_path, "rb") as handle:
                self.patients = pickle.load(handle)
                handle.close()
        elif isinstance(data, list) and isinstance(data[0], dict):
            self.patients = data
        else: 
            raise ValueError("Not supported type %s for argument data"%type(data))

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        # return torch.tensor(self.examples[item]), torch.tensor(self.lengths[item])
        return self.patients[item]


#####################################################################################################################
#                                                                                                                   #
#                                              Collate functions                                                    #
#                                                                                                                   #
#####################################################################################################################
# Collate_function for carediab retino
map_sex = {'F': 0, 'M': 1}
def collate_seq(x, pad_value, trunc_max_len,  side_info, test=False):
    """
    :param x:
    :param pad_value:
    :param add_delta_time:
    :param trunc_max_len:
    :param side_info:
    :param test:
    :return:
    """
    seq_hba1c = [torch.tensor(patient['seq_hba1c_norm'][:trunc_max_len]) for patient in x]
    timedelta = [torch.tensor(patient['timedelta_norm'][:trunc_max_len]) for patient in x]
    lengths = torch.tensor([len(s) for s in seq_hba1c], dtype=torch.float)

    if len(side_info) > 1:
        side_tensor = []
        for info in side_info:
            if info == 'sexe': 
                side_tensor.append(torch.tensor([map_sex[patient[info]] for patient in x]).view(-1, 1))
            else: 
                side_tensor.append(torch.tensor([patient[info] for patient in x]).view(-1, 1))
        side_tensor = torch.cat(side_tensor, dim=1)
    else:
        side_tensor = torch.tensor([patient[side_info[0]] for patient in x]).view(-1, 1)

    max_len = np.min((np.max([patient['seq_length'] for patient in x]), trunc_max_len))
    seq_hba1c = pad_sequence(seq_hba1c, batch_first = True, padding_value = pad_value).view(-1, max_len, 1).float()
    timedelta = pad_sequence(timedelta, batch_first = True, padding_value = pad_value).view(-1, max_len, 1).float()

    # target variable
    y = torch.tensor([patient['statut'] for patient in x])

    # Build temporal input class 
    temporal_input = TemporalInputs(event_sequences= {'seq_hba1c': seq_hba1c},
                                    time_gap_sequence = timedelta,
                                    patient_information = side_tensor,
                                    sequence_lengths = lengths)

    if test:
        patient_ids = [patient['patient_id'] for patient in x]
        sequence_length = [np.min((patient['seq_length'], trunc_max_len)) for patient in x]
        return temporal_input, y, patient_ids
    return temporal_input, y

# collate function for mimiciii IHM  
def collate_seq_ihm(x, test=False):
    inputs = {}
    inputs['seq_ts'] = torch.cat([torch.tensor(patient['ts']).unsqueeze(0) for patient in x], dim=0)
    inputs['delta_t'] = torch.cat([torch.tensor(patient['episodes_delta']).unsqueeze(0) for patient in x], dim=0)
    inputs['lengthofstays'] = torch.cat([torch.tensor(patient['episodes_los']).unsqueeze(0) for patient in x], dim=0)
    
    inputs['diag_codes'] = torch.cat([torch.tensor(patient['diag_codes']).unsqueeze(0) for patient in x], dim=0)
    inputs['proc_codes'] = torch.cat([torch.tensor(patient['proc_codes']).unsqueeze(0) for patient in x], dim=0)

    inputs['gender'] = torch.tensor([patient['gender_'] for  patient in x])
    inputs['ethnicity'] = torch.tensor([patient['ethnicity'] for  patient in x])
    inputs['age'] = torch.tensor([patient['age'] for  patient in x])

    y = torch.tensor([patient['label'] for  patient in x])

    if test:
        patient_ids = [patient['id'] for patient in x]
        return inputs, y, patient_ids
    
    return  inputs, y


#####################################################################################################################
#                                                                                                                   #
#                                                Fast.ai DataBunch                                                  #
#                                                                                                                   #
#####################################################################################################################


class FastDataBunch(DataBunch):
    @classmethod
    def remove_tfm(cls, tfm: Callable) -> None:
        """Remove `tfm` from `self.tfms`."""
        if tfm in cls.tfms:
            cls.tfms.remove(tfm)

    @classmethod
    def add_tfm(cls, tfm: Callable) -> None:
        """Add `tfm` to `self.tfms`."""
        cls.tfms.append(tfm)

    @classmethod
    def create(cls, train_ds: SeqDataset, valid_ds: SeqDataset, test_ds=None,
               path: PathOrStr = '.', bs: int = 64, n_gpu: int = 0, val_bs=None,
               num_workers: int = defaults.cpus, device: torch.device = None,
               collate_fn: Callable = None, tfms: List[Callable] = None,
               size: int = None, args=None, **kwargs):
        cls.tfms = listify(tfms)
        bs = bs * max(1, n_gpu)
        val_bs = ifnone(val_bs, bs)
        # get  weighted sampler w.r.t to target distribution
        targets = np.array([int(patient['statut']) for patient in train_ds.patients])
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight
        train_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_dl = DataLoader(train_ds,
                              sampler = train_sampler,
                              batch_size = bs,
                              collate_fn = collate_fn,
                              pin_memory = True)

        val_dl = DataLoader(valid_ds,
                            # sampler = val_sampler,
                            sampler = RandomSampler(valid_ds),
                            batch_size = val_bs,
                            collate_fn = collate_fn,
                            pin_memory = True)
        if valid_ds is not None:
            cls.empty_val = False
        else:
            cls.empty_val = True
        cls.device = defaults.device if device is None else device
        # Convert data-loaders to device loaders ?
        dls = [DeviceDataLoader(train_dl, device = device, collate_fn = collate_fn),
               DeviceDataLoader(val_dl, device = device, collate_fn = collate_fn)]
        # load batch in device
        if test_ds is not None:
            cls.train_dl, cls.valid_dl, cls.test_dl = dls
        else:
            cls.train_dl, cls.valid_dl = dls
        # set data path
        cls.path = Path(path)
        return cls

    def empty_val(self):
        self.empty_val = True

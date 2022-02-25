from typing import Optional, Callable, Any, Dict, List 
from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
from fastai.core import ItemBase

from .utils.utils import ACT2FN

#####################################
#                                   #
#          Tensors class            #
#                                   #
#####################################

class TemporalInputs(ItemBase): 
    def __init__(self, event_sequences, time_gap_sequence, patient_information, sequence_lengths): 
        self.event_sequences = event_sequences
        self.time_gap_sequence = time_gap_sequence
        self.patient_information = patient_information
        self.sequence_lengths = sequence_lengths
        self.data = [event_sequences, time_gap_sequence, patient_information, sequence_lengths]
    def to(self, device, non_blocking=True): 
        for name in self.event_sequences.keys():
            self.event_sequences[name] = self.event_sequences[name].to(device)
        return TemporalInputs(self.event_sequences, self.time_gap_sequence.to(device),
                    self.patient_information.to(device), 
                    self.sequence_lengths.to(device))



class TemporalOutput(object): 
    def __init__(self, event_representation, raw_time_gap, time_gap_representation, patient_information_representation, sequence_lengths): 
        self.event_representation = event_representation
        self.time_gap_representation = time_gap_representation
        self.patient_information_representation = patient_information_representation
        self.raw_time_gap = raw_time_gap
        self.sequence_lengths = sequence_lengths

    
#####################################
#                                   #
#      Representation modules       #
#                                   #
#####################################

class MLPProjection(nn.Module):
    """
    Project inputs into a dense MLP network with n layers 
    """
    def __init__(self, input_dim, hidden_dims, output_dim, tf_activation): 
        super(MLPProjection, self).__init__()
        self.mlp_layers=[nn.Linear(input_dim, hidden_dims[0])]
        for i in range(1, len(hidden_dims)):
            self.mlp_layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))


        self.mlp_layers = nn.ModuleList(self.mlp_layers)   
        self.output = nn.Linear(hidden_dims[-1], output_dim)
        self.activation = tf_activation

    def forward(self, inputs): 
        for layer in self.mlp_layers: 
            inputs = layer(inputs.float())
        x = self.activation(self.output(inputs))
        return x


class SoftOneHotEmbedding(nn.Module):
    def __init__(self, projection_size, time_embedding, embeddings_init_std):
        super(SoftOneHotEmbedding, self).__init__()
        self.projection_size = projection_size
        self.time_embedding = time_embedding
        self.projection_layer = nn.Linear(1, self.projection_size, bias=True)
        self.embedding_matrix =  nn.Embedding(self.projection_size, self.time_embedding)
        with torch.no_grad():
            self.embedding_matrix.weight.normal_(0.0, embeddings_init_std)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_numeric):
        weights = self.softmax(self.projection_layer(input_numeric.float()))
        soft_one_hot_embeddings = (
            weights.unsqueeze(-1) * self.embedding_matrix.weight
        ).sum(-2)
        return soft_one_hot_embeddings

class TimeMask(nn.Module):
    """
    Learn time representayion as a mask embedding 
    """
    def __init__(self,projection_size, event_embedding_dim): 
        super(TimeMask, self).__init__()

        self.time_embedding_1 = nn.Linear(1, projection_size)
        self.relu_time = nn.ReLU()
        self.time_embedding_2 = nn.Linear(projection_size, event_embedding_dim)
        self.sigmoid_time = nn.Sigmoid()
    
    def forward(self, inputs):
        # first projection 
        time_emb = self.relu_time(self.time_embedding_1(inputs))
        # second project and sigmoid mask 
        time_mask = self.sigmoid_time(self.time_embedding_2(time_emb))
        return time_mask


class Positional_time_encoding(nn.Module):
    """
    Learn a positional time representation using functional kernel 
    """
    def __init__(self, time_dim, expand_dim, device):
        super(Positional_time_encoding, self).__init__()

        # Init the basis frequencies (omegas)
        freq_var_torch = torch.nn.Parameter(torch.tensor(1 / 10 ** np.linspace(0, 9, time_dim), device=device, dtype=torch.float))
        freq_var_torch = freq_var_torch.unsqueeze(-1).expand([time_dim, expand_dim])
        self.freq_var_torch = freq_var_torch
        # expand_coef = torch.arange(start = 1, end = expand_dim + 1).unsqueeze(0).float()
        # self.freq_var_torch = freq_var_torch * expand_coef

        # constant (ci) and 0-order constant (bias)
        basis_expand_var_torch = torch.empty([time_dim, 2 * expand_dim],
                                             device = device)
        self.basis_expand_var_torch = torch.nn.Parameter(torch.nn.init.xavier_uniform_(basis_expand_var_torch))
        self.basis_expand_var_bias_torch = torch.nn.Parameter(torch.zeros(time_dim,
                                                                          device = device, dtype= torch.float))

        self.time_dim = time_dim
        self.expand_dim = expand_dim

    def forward(self, inputs):
        bs, len_ = inputs.shape
        expand_input = inputs.unsqueeze(-1).expand([bs, len_, self.time_dim])
        sin_enc_torch = torch.sin(torch.mul(expand_input.view(bs, len_, self.time_dim, 1),
                                            self.freq_var_torch.view(1, 1, self.time_dim, self.expand_dim)))
        cos_enc_torch = torch.cos(torch.mul(expand_input.view(bs, len_, self.time_dim, 1),
                                            self.freq_var_torch.view(1, 1, self.time_dim, self.expand_dim)))
        time_enc_torch = torch.cat([sin_enc_torch, cos_enc_torch], axis = -1)
        # time_enc_torch = torch.mul(time_enc_torch,
        #                           self.basis_expand_var_torch.view(1, 1, self.time_dim, self.expand_dim * 2))
        time_enc_torch = torch.sum(time_enc_torch, -1) + self.basis_expand_var_bias_torch.view(1, 1, self.time_dim)

        return time_enc_torch


class TimeEncode(torch.nn.Module):
    """
    Learn a cyclic time presentation 
    """
    def __init__(self, expand_dim, device='cuda'):
        super(TimeEncode, self).__init__()
        # init_len = np.array([1e8**(i/(time_dim-1)) for i in range(time_dim)])
        time_dim = expand_dim
        self.basis_freq = torch.nn.Parameter(torch.tensor(1 / 10 ** np.linspace(0, 9, time_dim), device=device, dtype=torch.float))
        self.phase = torch.nn.Parameter(torch.zeros(time_dim,  device = device, dtype = torch.float))

    def forward(self, inputs):
        batch_size = inputs.size(0)
        seq_len = inputs.size(1)

        inputs = inputs.view(batch_size, seq_len, 1)  # [N, L, 1]
        map_ts = inputs * self.basis_freq.view(1, 1, -1)  # [N, L, time_dim]
        map_ts += self.phase.view(1, 1, -1)

        harmonic = torch.cos(map_ts)

        return harmonic  # self.dense(harmonic)

class CategoricalEncoder(torch.nn.Module): 
    """
    Learn embedding representation of categorical variable
    """
    def __init__(self, embed_dim):
        super(CategoricalEncoder, self).__init__()
        self.embed_table = nn.Embedding(embedding_dim=embed_dim)
    
    def forward(self, inputs): 
        return self.embed_table(inputs)

#####################################
#                                   #
#         Aggregation class         #
#                                   #
#####################################

class Aggregation(nn.Module): 
    def __init__(self, aggregation_type, use_patient_info, use_time_info):
        super(Aggregation, self).__init__()
        self.aggregation_type = aggregation_type
        self.use_patient_info = use_patient_info
        self.use_time_info = use_time_info

    
    def forward(self, temporal_output): 
        # use only time info 
        if (not self.use_patient_info) and (self.use_time_info): 
            if self.aggregation_type == 'concat': 
                sequence_representation = torch.cat([temporal_output.event_representation, temporal_output.time_gap_representation], -1)
            elif self.aggregation_type == 'element-wise-multiplication':
                sequence_representation = temporal_output.event_representation * temporal_output.time_gap_representation
            elif self.aggregation_type == 'mask-multiplication':
                sequence_representation = temporal_output.event_representation * temporal_output.time_gap_representation
            elif self.aggregation_type == 'element-wise-sum':
                sequence_representation = temporal_output.event_representation + temporal_output.time_gap_representation
       
       # use both patient and time info 
        elif (self.use_patient_info) and (self.use_time_info): 
            bs, patient_dim = temporal_output.patient_information_representation.shape
            seq_len = temporal_output.event_representation.size(1)
            repeated_patient_info = temporal_output.patient_information_representation.unsqueeze(1).repeat(1, seq_len, 1)
            if self.aggregation_type == 'element-wise-sum-multiplication':
                sequence_representation = temporal_output.event_representation * (temporal_output.time_gap_representation+ repeated_patient_info)
            elif self.aggregation_type == 'concat': 
                sequence_representation = torch.cat([temporal_output.event_representation, 
                                                     temporal_output.time_gap_representation,
                                                      repeated_patient_info], -1)
            elif self.aggregation_type == 'mask-multiplication':
                sequence_representation = torch.cat([temporal_output.event_representation * temporal_output.time_gap_representation,  repeated_patient_info], dim=-1)                  
            elif self.aggregation_type == 'element-wise-multiplication':
                sequence_representation = temporal_output.event_representation * temporal_output.time_gap_representation * repeated_patient_info
            elif self.aggregation_type == 'element-wise-sum':
                sequence_representation = temporal_output.event_representation + temporal_output.time_gap_representation + repeated_patient_info
            
        # use only patient info 
        elif (self.use_patient_info) and (not self.use_time_info): 
            bs, patient_dim = temporal_output.patient_information_representation.shape
            seq_len = temporal_output.event_representation.size(1)
            repeated_patient_info = temporal_output.patient_information_representation.unsqueeze(1).repeat(1, seq_len, 1)
            if self.aggregation_type == 'element-wise-sum-multiplication':
                sequence_representation = temporal_output.event_representation * (repeated_patient_info)
            elif self.aggregation_type == 'concat': 
                sequence_representation = torch.cat([temporal_output.event_representation, repeated_patient_info], -1)
            elif self.aggregation_type == 'mask-multiplication':
                raise ValueError("mask_multiplication only works when use_time_info is set to True")
            elif self.aggregation_type == 'element-wise-multiplication':
                sequence_representation = temporal_output.event_representation * repeated_patient_info
            elif self.aggregation_type == 'element-wise-sum':
                sequence_representation = temporal_output.event_representation + repeated_patient_info
        
        #  only use event info  
        elif (not self.use_patient_info) and (not self.use_time_info): 
                sequence_representation = temporal_output.event_representation
        return (sequence_representation, temporal_output.time_gap_representation, temporal_output.patient_information_representation)

        
#####################################
#                                   #
#           PatientConfigs          #
#                                   #
#####################################

class TimeConfig(object): 
    def __init__(self, time_representation: str,
                hidden_dims: List[int],
                projection_size: int,
                tf_activation: str,
                output_dim: int, 
                embeddings_init_std: float=0.01):
        self.time_representation = time_representation
        self.hidden_dims = hidden_dims
        self.projection_size = projection_size
        self.output_dim = output_dim
        self.embeddings_init_std = embeddings_init_std
        self.tf_activation = ACT2FN[tf_activation]

class PatientConfig(object):
    def __init__(self, patient_representation: str,
                input_dim: int,
                hidden_dims: List[int],
                tf_activation: str,
                output_dim: int):
        self.patient_representation = patient_representation
        self.hidden_dims = hidden_dims
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.tf_activation = ACT2FN[tf_activation]


class EventConfig(object):
    def __init__(self,
                categoricals: List[str] = [],
                continuous: List[str] = [], 
                categorical_representation: str = None,
                continuous_representation: str = None,
                #categorical_input_dim: int = 0,
                #continuous_input_dim: int = 0,
                categorical_embeddings: List[int] = [],
                continuous_hidden_dims: List[int] = [],
                continuous_output_dim: int = 64,
                tf_activation: str = 'relu'):
        self.categoricals = categoricals
        self.continuous = continuous
        self.categorical_embeddings = categorical_embeddings
        self.categorical_representation = categorical_representation
        self.continuous_representation = continuous_representation
        self.continuous_hidden_dims = continuous_hidden_dims
        self.tf_activation = ACT2FN[tf_activation]
        self.continuous_output_dim = continuous_output_dim

#####################################
#                                   #
#     PatientEmbedding class        #
#                                   #
#####################################

class PatientEmbedding(nn.Module): 
    def __init__(self, aggregation_type, use_patient_info, use_time_info, event_config, time_config, patient_config):
        super(PatientEmbedding, self).__init__()
        self.aggregation_type = aggregation_type
        self.use_patient_info = use_patient_info
        self.use_time_info = use_time_info
        self.time_representation = time_config.time_representation
        self.patient_info_representation = patient_config.patient_representation
        self.feature_norm = False
        
        # time gap representation 
        if self.time_representation == 'mlp': 
            self.time_module = MLPProjection(1, time_config.hidden_dims, time_config.output_dim, time_config.tf_activation)
        elif self.time_representation == 'time_mask': 
            self.time_module = TimeMask(time_config.projection_size, time_config.output_dim)
        elif self.time_representation == 'soft-one-hot': 
            self.time_module = SoftOneHotEmbedding(time_config.projection_size, time_config.output_dim, time_config.embeddings_init_std)
        elif self.time_representation == 'time_encode': 
            self.time_module = TimeEncode(time_config.output_dim)
        elif self.time_representation == 'identity':
            self.time_module = nn.Identity()
        elif self.time_representation == 'positional_kernel': 
            self.time_module = Positional_time_encoding(time_config.output_dim, time_config.projection_size)

        # patient info representation 
        if self.patient_info_representation == 'mlp': 
            self.patient_module = MLPProjection(patient_config.input_dim, patient_config.hidden_dims, patient_config.output_dim, patient_config.tf_activation)
        elif self.patient_info_representation == 'identity':
            self.patient_module = nn.Identity()  

        # event embedding representation 
        self.event_module = nn.ModuleDict()

        if event_config.continuous_representation == 'mlp':
            for feature_name in event_config.continuous : 
                self.event_module[feature_name] = MLPProjection(1, event_config.continuous_hidden_dims, event_config.continuous_output_dim, event_config.tf_activation)
        if event_config.continuous_representation == 'identity':
            for feature_name in event_config.continous : 
                self.event_module[feature_name] = nn.Identity()
        self.event_names = event_config.continuous + event_config.categoricals
         #TODO: Add categorical embedding representation to event_module 

        # Patient embedding aggregation
        self.aggregator = Aggregation(self.aggregation_type, self.use_patient_info, self.use_time_info)

        # feature-wise normalization 
        if self.feature_norm: 
             self.layernorms = nn.ModuleList([nn.LayerNorm(normalized_shape =  patient_config.output_dim),
                                            nn.LayerNorm(normalized_shape =  time_config.output_dim),
                                            nn.LayerNorm(normalized_shape= event_config.continuous_output_dim)])


    def forward(self, temporal_inputs): 
        # Represent time-gap vector 
        x_time = self.time_module(temporal_inputs.time_gap_sequence)
        x_patient = self.patient_module(temporal_inputs.patient_information)
        # Represent event vector 
        x_event = torch.cat([self.event_module[name](temporal_inputs.event_sequences[name]) for name in self.event_names], dim=-1)
        if self.feature_norm: 
            x_patient, x_time, x_event = [layer(x) for x, layer in zip([x_patient, x_time, x_event ], self.layernorms)]

        temporal_output = TemporalOutput(x_event,  temporal_inputs.time_gap_sequence, x_time, x_patient, temporal_inputs.sequence_lengths)

        input_rep, input_time, input_patient = self.aggregator(temporal_output)

        return {'event_representation': input_rep, "raw_time_gap":temporal_inputs.time_gap_sequence, "time_representation":input_time,
                'patient_representation': input_patient, "sequence_lengths" :temporal_inputs.sequence_lengths}


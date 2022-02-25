import imp
import torch 
from torch import nn
from .time_model import TimeModel
from .patient_embedding import TimeConfig, PatientConfig, EventConfig, PatientEmbedding, TemporalInputs
from .medical_classifier import MedicalClassifier

class Hba1cModel(nn.Module): 
    def __init__(self, config): 
        super(Hba1cModel, self).__init__()
        self.config = set_up_dimensions(config)
        

        # init patient embedding config class 
        patient_config_dict = config['patient_config']
        patient_config = PatientConfig(patient_config_dict['representation_type'],
                                    patient_config_dict['num_inputs'],
                                    patient_config_dict['hidden_dims'],
                                    patient_config_dict['activation'],
                                    patient_config_dict['output_dim'],
                                    )
        # init time embedding config class 
        time_config_dict = config['time_config']
        time_config = TimeConfig(time_config_dict['representation_type'],
                                    time_config_dict['hidden_dims'],
                                    time_config_dict['projection_size'],
                                    time_config_dict['activation'],
                                    time_config_dict['output_dim'],
                                    time_config_dict['embeddings_init_std']
                                    )
        
        # init event embedding config class 
        event_config_dict = config['event_config']
        event_config = EventConfig(event_config_dict['categoricals'],
                                event_config_dict['continuous'], 
                                event_config_dict['categorical_representation'],
                                event_config_dict['continuous_representation'],
                                event_config_dict['categorical_embeddings'],
                                event_config_dict['continuous_hidden_dims'],
                                event_config_dict['continuous_output_dim'],
                                event_config_dict['tf_activation'])

        self.patient_embedding = PatientEmbedding(config['aggregation_mode'], 
                                                config['use_patient_info'],
                                                config['use_time_info'], 
                                                event_config,
                                                time_config,
                                                patient_config)

        self.model_type = config['model_type']
        self.time_model = TimeModel(config['model_type'], config['max_len'], config['temporal_model'])

        config_classifier = config['classifier']
        self.classifier = MedicalClassifier(input_dim=config_classifier['input_dim'],
                             hidden_dim=config_classifier['hidden_dim'],
                             output_dropout=config_classifier['output_dropout'],
                             num_classes=config_classifier['num_classes'],
                             use_patient_info=config_classifier['use_patient_info'],
                             patient_embedding_dim=config_classifier['patient_embedding_dim'])
        
    def forward(self, inputs):
        x_rep = self.patient_embedding(inputs)
        x, self_attention_weights = self.time_model(x_rep, mask=None)
        out, attention = self.classifier(x, x_rep['sequence_lengths'], x_rep['patient_representation'])
        if self.model_type == 'attention':
            return out, [attention, self_attention_weights]
        return out, [attention]



def set_up_dimensions(config): 
    event_config_dict = config['event_config']
    patient_config_dict = config['patient_config']
     # set up dimension of inputs of TimeModel 
    if config['aggregation_mode'] in ['element-wise-sum-multiplication', 'element-wise-sum', 'element-wise-multiplication']: 
        config['patient_config']['output_dim'] = event_config_dict['continuous_output_dim']
        config['time_config']['output_dim'] = event_config_dict['continuous_output_dim']
        config['temporal_model']['input_size'] =  event_config_dict['continuous_output_dim']
        # set up dimension of inputs of TimeModel  for masking mode 
    elif config['aggregation_mode'] ==  'mask-multiplication':
        config['time_config']['output_dim'] = event_config_dict['continuous_output_dim']
        if not config['use_patient_info']:
            config['temporal_model']['input_size'] =  event_config_dict['continuous_output_dim']
        else: 
            config['temporal_model']['input_size'] =  event_config_dict['continuous_output_dim']+ patient_config_dict['output_dim']
    # set up dimension of inputs of TimeModel  for concat mode 
    elif config['aggregation_mode'] == 'concat':
        if (config['use_time_info']) and (config['use_patient_info']):
            config['temporal_model']['input_size'] =  event_config_dict['continuous_output_dim'] + config['time_config']['output_dim'] + patient_config_dict['output_dim']
        elif config['use_time_info'] and not config['use_patient_info']:
            config['temporal_model']['input_size'] =  event_config_dict['continuous_output_dim'] + config['time_config']['output_dim']
        elif not config['use_time_info'] and config['use_patient_info']:
            config['temporal_model']['input_size'] =  event_config_dict['continuous_output_dim'] + patient_config_dict['output_dim']
        else:
            config['temporal_model']['input_size'] =  event_config_dict['continuous_output_dim'] 
    
    if config['model_type'] == 'attention':
        config['temporal_model']['hidden_size'] = config['temporal_model']['input_size']
    # infer attention heads of Transformer bloc
    config['temporal_model']['attn_heads'] =  config['temporal_model']['input_size'] // config['temporal_model']['hidden_size'] 

    
    # set up input dim of  classifier 
    if config['temporal_model']['bidirectional'] and config['model_type'] == 'lstm':
        config['classifier']['input_dim'] = config['temporal_model']['hidden_size'] * 2
    elif config['model_type'] == 'attention':
        config['classifier']['input_dim'] = config['temporal_model']['hidden_size'] * config['temporal_model']['attn_heads']
    else: 
        config['classifier']['input_dim'] = config['temporal_model']['hidden_size'] 
    
    config['classifier']['patient_embedding_dim'] = config['patient_config']['output_dim']
 
    return config

#import numpy as np 
#cfg = load_cfg("./config_files/test_1/test.yaml")
#model = Hba1cModel(cfg)
#patient = torch.tensor(np.random.uniform(0, 1, (16,  1)))
#time = torch.tensor(np.random.uniform(0, 1, (16, 10, 1)))
#event = {'seq_hba1c': torch.tensor(np.random.uniform(0, 1, (16, 10, 1)))} 
#sequence_lengths = torch.tensor(np.random.randint(1, 11, 16))
#temporal_inputs = TemporalInputs(event, time, patient, sequence_lengths)
#print(model(temporal_inputs)[0])

        


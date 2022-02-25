import torch 
from torch import nn 
from .utils.utils import mask_softmax, seq_mask
from .time_model import TimeModel 
from .patient_embedding import *

class MedicalClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dropout, num_classes, use_patient_info, patient_embedding_dim): 
        super(MedicalClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc_att = nn.Linear(input_dim, 1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(output_dropout)
        self.use_patient_info = use_patient_info
        self.num_classes = num_classes
        if self.use_patient_info:
            self.classifier_in_dim = hidden_dim + patient_embedding_dim
        else: 
            self.classifier_in_dim = hidden_dim
        self.classifier_layer = nn.Linear(self.classifier_in_dim, self.num_classes)
        self.sigmoid = nn.Sigmoid()
        self.output_attention = True

    def forward(self, inputs, sequence_len, x_patient_info):
        # Compute attention weights of the sequencee 
        max_seq_len = torch.max(sequence_len)
        mask = seq_mask(sequence_len, max_seq_len)  # [b,seq_len]
        att = self.fc_att(inputs.float()).squeeze(-1)  # [b,seq_len,hidden]->[b,seq_len]
        att = mask_softmax(att, mask)  # [b,seq_len]
        # sequence represented as a weithed sum of hidden representations 
        r_att = torch.sum(att.unsqueeze(-1) * inputs, dim=1)  # [b,hidden]
        # project hidden representation 
        r_att = self.drop(self.act(self.fc(r_att))) #[b,h]
        # concat the hidden representation of the sequence with patient side-info 
        if self.use_patient_info: 
            r_att = torch.cat([r_att, x_patient_info], dim = 1)

        logits = self.sigmoid(self.classifier_layer(r_att))  # [b,h]->[b, num_classes]

        if self.output_attention:
            return logits, att
        return logits, []

    


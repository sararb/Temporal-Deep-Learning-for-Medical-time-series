import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math 

from .patient_embedding import * 
from .utils.utils import SublayerConnection, PositionwiseFeedForward, MultiHeadedAttention

#from .attention.multi_head import MultiHeadedAttention
#from models.feedforward import PositionwiseFeedForward
#from .utils.sublayerimport SublayerConnection

####################################################################################################################
#                                                1- BI-LSTM model                                                  #
####################################################################################################################

class DynamicLSTM(nn.Module):
    """
    Dynamic LSTM module, which can handle variable length x_seq sequence.

    Parameters
    ----------
    input_size : x_seq size
    hidden_size : hidden size
    num_layers : number of hidden layers. Default: 1
    dropout : dropout rate. Default: 0.5
    bidirectional : If True, becomes a bidirectional RNN. Default: False.

    Forward - Inputs
    ------
        x: tensor, shaped [batch, max_step, input_size]
        seq_lens: tensor, shaped [batch], sequence lengths of batch

    Forward - Outputs
    -------
        y: tensor, shaped [batch, max_step, num_directions * hidden_size],
             tensor containing the output features (h_t) from the last layer
             of the LSTM, for each t.
    """

    def __init__(self, config):
        super(DynamicLSTM, self).__init__()

        self.lstm = nn.LSTM(config['input_size'], config['hidden_size'], config['num_layers'], bias=True,
            batch_first=True, dropout=config['dropout'], bidirectional=config['bidirectional'])

    def forward(self, inputs, mask):
        # sort x_seq by descending length
        x = inputs['event_representation']
        seq_lens = inputs['sequence_lengths']
        _, idx_sort = torch.sort(seq_lens, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)
        x_sort = torch.index_select(x, dim=0, index=idx_sort)
        seq_lens_sort = torch.index_select(seq_lens, dim=0, index=idx_sort)

        # pack x_seq
        x_packed = pack_padded_sequence(
            x_sort, seq_lens_sort.cpu(), batch_first=True)

        # pass through rnn
        y_packed, _ = self.lstm(x_packed)

        # unpack output
        y_sort, length = pad_packed_sequence(y_packed, batch_first=True)

        # unsort output to original order
        y = torch.index_select(y_sort, dim=0, index=idx_unsort)

        return y, []

####################################################################################################################
#                                                2- C-LSTM model                                                   #
####################################################################################################################
class C_LSTM(nn.Module):
    def __init__(self, config):
        super(C_LSTM, self).__init__()
        self.hidden_size = config['hidden_size']
        self.timedecay_size = config['timedecay_size']
        self.W = nn.Parameter(torch.zeros((1, self.hidden_size * 4), device='cuda', dtype=torch.float), requires_grad=True)
        self.U = nn.Parameter(torch.zeros((self.hidden_size, self.hidden_size * 4), device='cuda', dtype=torch.float), requires_grad=True)
        self.Q = nn.Parameter(torch.zeros((self.timedecay_size, self.hidden_size), device='cuda', dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros((self.hidden_size * 4), device='cuda', dtype=torch.float), requires_grad=True)
        self.model_time = config['model_time']
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
        self.W.data.uniform_(-stdv, stdv)
        self.U.data.uniform_(-stdv, stdv)
        self.Q.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, mask,
                init_states=None):
        """Assumes x_in is of shape (batch, sequence_len, feature_sz)"""
        events = inputs['event_representation']
        bs, seq_sz, _ = events.size()
        timedecay = inputs['raw_time_gap']
        # compute different order of time-gap vector 
        timedecay = torch.cat([torch.pow(timedecay, i) for i in range(1,
                                                                      self.timedecay_size+1)]).reshape(bs, seq_sz, -1)
        hidden_seq = []
        if init_states is None:
            h_t, c_t = (torch.zeros(bs, self.hidden_size).to(events.device),
                        torch.zeros(bs, self.hidden_size).to(events.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = events[:, t, 0].unsqueeze(-1)
            # batch the computations into a single matrix multiplication
            in_ = x_t @ self.W
            hidd_ = h_t @ self.U
            gates = in_ + hidd_ + self.bias
            gates = gates
            # get different par of lstm:
            if self.model_time == 'forget':
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # x_seq
                    torch.sigmoid(gates[:, HS:HS * 2] + timedecay[:, t, :] @ self.Q),  # forget + time-decay
                    torch.tanh(gates[:, HS * 2:HS * 3]),  # gating
                    torch.sigmoid(gates[:, HS * 3:]),  # output
                )

            elif self.model_time == 'output':
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # x_seq
                    torch.sigmoid(gates[:, HS:HS * 2]),  # forget + time-decay
                    torch.tanh(gates[:, HS * 2:HS * 3]),  # gating
                    torch.sigmoid(gates[:, HS * 3:] + timedecay[:, t, :] @ self.Q),  # output
                )

            elif self.model_time == 'forget_output':
                i_t, f_t, g_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # x_seq
                    torch.sigmoid(gates[:, HS:HS * 2] + timedecay[:, t, :] @ self.Q),  # forget + time-decay
                    torch.tanh(gates[:, HS * 2:HS * 3]),  # gating
                    torch.sigmoid(gates[:, HS * 3:] + timedecay[:, t, :] @ self.Q),  # output
                )

            else:
                raise ValueError(
                    "model_time should be one of the following options: ['output', 'forget_output', 'forget']")

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim = 0)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

####################################################################################################################
#                                             3- Self-Attention model                                              #
####################################################################################################################
class PositionEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim):
        super(PositionEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)

    def forward(self, x):
        _, seq_len = x.size()[:2]
        embeddings = self.weight[:seq_len, :].view(1, seq_len, self.embedding_dim)
        return x + embeddings

    def extra_repr(self):
        return 'num_embeddings={}, embedding_dim={} mode="Add"'.format(
            self.num_embeddings, self.embedding_dim
        )

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    Parameters:
        :param hidden:  hidden size of transformer
        :param attn_heads: number of multi-head attention heads
        :param attention_dropout_prob: attention_dropout_prob probability of attention layer
        :param feed_forward_hidden: hidden size of point-wise feed forward network, usually set at 4*hidden_size
        :param hidden_dropout_prob: attention_dropout_prob probability of feed forward layer
        :param hidden_act: The activation function to apply to "intermediate" layer
    """

    def __init__(self, input_dim, hidden, attn_heads, attention_dropout_prob,
                 feed_forward_hidden, hidden_dropout_prob, hidden_act, max_len=151, use_position_embedding=False):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(num_attention_heads = attn_heads,
                                              input_dim = input_dim,
                                              d_model = hidden,
                                              attention_dropout_prob = attention_dropout_prob)
        self.feed_forward = PositionwiseFeedForward(d_model = hidden,
                                                    d_ff = feed_forward_hidden,
                                                    hidden_dropout_prob = hidden_dropout_prob,
                                                    hidden_act = hidden_act)

        self.input_sublayer = SublayerConnection(hidden_size = hidden,
                                                 hidden_dropout_prob = hidden_dropout_prob,
                                                 attention_layer = True)

        self.output_sublayer = SublayerConnection(hidden_size = hidden,
                                                  hidden_dropout_prob = hidden_dropout_prob,
                                                  attention_layer = False
                                                  )
        self.dropout = nn.Dropout(p = hidden_dropout_prob)

        self.use_position_embedding = use_position_embedding
        self.max_len = max_len
        if self.use_position_embedding: 
            self.positional_embedding = PositionEmbedding(self.max_len, hidden)

    def forward(self, inputs, mask, output_attention=True):
        """
        :param inputs: sequence of event embedding 
        :param mask:  Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        :param output_attention: output attention weights or not
        :return:
            x: encoded vector
        """
        if self.use_position_embedding: 
            inputs = self.positional_embedding(inputs)
        inputs, weights = self.input_sublayer(inputs, lambda _x: self.attention.forward(_x, _x, _x,
                                                                                    mask = mask,
                                                                                    output_attention=output_attention))
        inputs = self.output_sublayer(inputs, self.feed_forward)
        inputs = self.dropout(inputs)

        if output_attention:
            return inputs, weights
        return inputs, []

class AttentionModel(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    Parameters:
        :param config: the index of PAD token
    """

    def __init__(self, config, max_len):
        super(AttentionModel, self).__init__()
        self.pad_value = config['pad_value']
        self.hidden = config['hidden_size'] * config['attn_heads']
        self.n_layers = config['num_layers']
        self.attn_heads = config['attn_heads']
        self.time_dim = self.hidden - 1
        self.expand_dim = self.hidden - 1
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = config['feed_forward_hidden']
        self.output_attention = config['output_self_attention']


        # multi-layers transformer blocks, deep network : Stack of BERT encoders
        self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden = self.hidden,
                                                                  input_dim =self.time_dim + config['num_input'],
                                                                  attn_heads = self.attn_heads,
                                                                  feed_forward_hidden = self.feed_forward_hidden,
                                                                  attention_dropout_prob = config['attn_dropout_prob'],
                                                                  hidden_dropout_prob = self.hidden_dropout_prob,
                                                                  hidden_act = config['hidden_act'], 
                                                                  use_position_embedding=config['use_position_embedding'],
                                                                  max_len= max_len
                                                                  )
                                                 for _ in range(self.n_layers)])

    def forward(self,
                inputs,
                attention_mask=None):
        """
        Parameters:
            :param inputs:  
            :param attention_mask:
            :param output_attention:
        :return:
            x_in: BERT hidden states vector
        """
        # attention masking for padded token
        x_input = inputs['event_representation']
        # running over multiple transformer blocks
        output_attn = ()
        for transformer in self.transformer_blocks:
            x_input, weights = transformer.forward(x_input, attention_mask, self.output_attention)
            output_attn = output_attn + (weights,)

        if self.output_attention:
            return x_input, output_attn
        return x_input, []


##################################################################################################################
#                                                                                                                #
#                                            TimeModel Meta-Class module                                         #
#                                                                                                                #
##################################################################################################################

class TimeModel(nn.Module):
    def __init__(self, model_type, max_len, config):
        super(TimeModel, self).__init__()
        self.model_type = model_type
        self.config = config
        if self.model_type == 'clstm': 
            self.temporal_module = C_LSTM(config)
        elif self.model_type == 'attention':
            self.temporal_module = AttentionModel(config, max_len)
        elif self.model_type == 'lstm':
            self.temporal_module = DynamicLSTM(config)
    
    def forward(self, inputs, mask): 
        return self.temporal_module(inputs, mask)

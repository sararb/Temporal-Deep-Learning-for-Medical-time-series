import torch
import torch.nn.functional as F
import torch.nn as nn

import math

NEG_INF = -10000


def attention(lstm_output, final_state, bidirection=False, output_attention=False):
    """
    Now we will incorporate Attention mechanism in our LSTM model.
    In this new model, we will use attention to compute soft alignment score corresponding
    between each of the hidden_state and the last hidden_state of the LSTM.
    ==> We will be using torch.bmm for the batch matrix multiplication.

    Arguments
    ---------
    :param: lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
    :param: final_state : Final time-step hidden state (h_n) of the LSTM

    ---------

    It performs attention mechanism by first computing weights for each of the sequence present in
    lstm_output and and then finally computing the new hidden state.

    :returns:
    new_hidden_state: a weighted final hidden state vector to use for prediction

    Tensor Size :
                hidden.size() = (batch_size, num_direction*hidden_size)
                attn_weights.size() = (batch_size, num_seq)
                soft_attn_weights.size() = (batch_size, num_seq)
                new_hidden_state.size() = (batch_size, num_direction*hidden_size)

    """

    bs, h_dim = final_state.shape
    if bidirection:
        hidden = torch.cat([s for s in final_state], 1)
    else:
        hidden = final_state
    hidden = hidden.view(bs, h_dim, 1)
    attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
    soft_attn_weights = F.softmax(attn_weights, 1)
    new_hidden_state = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
    if output_attention:
        return new_hidden_state, attn_weights
    return new_hidden_state, []


def mask_softmax(matrix, mask=None):
    """Perform softmax on length dimension with masking.

    Parameters
    ----------
    matrix: torch.float, shape [batch_size, .., max_len]
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.

    Returns
    -------
    output: torch.float, shape [batch_size, .., max_len]
        Normalized output in length dimension.
    """

    if mask is None:
        result = F.softmax(matrix, dim = -1)
    else:
        mask_norm = ((1 - mask) * NEG_INF).to(matrix)
        for i in range(matrix.dim() - mask_norm.dim()):
            mask_norm = mask_norm.unsqueeze(1)
        result = F.softmax(matrix + mask_norm, dim = -1)

    return result


def seq_mask(seq_len, max_len):
    """Create sequence mask.

    Parameters
    ----------
    seq_len: torch.long, shape [batch_size],
        Lengths of sequences in a batch.
    max_len: int
        The maximum sequence length in a batch.

    Returns
    -------
    mask: torch.long, shape [batch_size, max_len]
        Mask tensor for sequence.
    """

    idx = torch.arange(max_len).to(seq_len).repeat(seq_len.size(0), 1)
    mask = torch.gt(seq_len.unsqueeze(1), idx).to(seq_len)

    return mask

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):
        """
        Construct a layer norm module in the TF style (epsilon inside the square root).
        params:
            hidden_size: the size of x_seq hidden vector
            variance_epsilon: to avoid zero division
        """
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation for Feed Forward module of the Encoder.
    :param d_model: Hidden size of Transformer model
    :param d_ff: dimension of "intermediate" feed forward layer
    :param hidden_dropout_prob: drop out probability
    :param hidden_act: activation function applied to intermediate layer
    """

    def __init__(self, d_model, d_ff, hidden_dropout_prob=0.1, hidden_act='gelu'):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.activation = ACT2FN[hidden_act] if isinstance(hidden_act, str) else hidden_act

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    params:
        hidden_size: size of x_seq hidden vector
        hidden_dropout_prob: hidden_dropout_prob probability
    """

    def __init__(self, hidden_size, hidden_dropout_prob, attention_layer):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.attention_layer = attention_layer

    def forward(self, x, sublayer):
        """Apply residual connection to any sublayer with the same hidden_size. """
        if self.attention_layer:
            x_attn, weights = sublayer(x)
            return self.norm(x + self.dropout(x_attn)), weights
        return self.norm(x + self.dropout(sublayer(x)))

"""
Implement the activations functions to use as hidden activation of point-wise fully connected feed forward layer
"""
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x_in * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x_in + 0.044715 * torch.pow(x_in, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "tanh":torch.tanh,
          "swish": swish, "gelu_new": gelu_new, "mish": mish,
          "gelu_fast": gelu_fast}

import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product (multiplicative) Attention'
     To attend to information from different representation subspaces at different positions
    """

    def forward(self, query, key, value, mask=None, attention_dropout=None):
        """
        :param query: representation vector of queries
        :param key: representation vector of keys to combine with query in order to compute the weight of each value
        :param value: representation vector of values
        :param mask: Mask padded values to not attend to ( Pre-computed in forward method of the bert-base models)
        :param attention_dropout: attention_dropout_prob probability of attention layer
        :return: weighted sum of the values vector & attention weights
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            # Apply the attention mask pre-computed
            scores = scores.masked_fill(mask, -1e10)
        p_attn = F.softmax(scores, dim=-1)
        if attention_dropout is not None:
            p_attn = attention_dropout(p_attn)
        context_layer = torch.matmul(p_attn, value)
        return context_layer, p_attn


class MultiHeadedAttention(nn.Module):
    """
    Compute Attention context vectors from query, key and value vectors
    Concat different information of Context-Vector in one general Attention-Vector
    Parameters:
        :param num_attention_heads: The number of attention heads
        :param d_model: the hidden size of the Transformer
        :param attention_dropout_prob: the dropout probability of attention layer

    """

    def __init__(self, num_attention_heads, input_dim,  d_model, attention_dropout_prob=0.1):
        """
        """
        super(MultiHeadedAttention, self).__init__()
        # Check that the hidden size of the transformer is a multiple of number of attention heads
        if d_model % num_attention_heads != 0:
            raise ValueError(
                "The hidden hidden_size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (d_model, num_attention_heads))
        # We assume d_v always equals d_k : so that d_k represents the attention_head_size
        self.d_k = d_model // num_attention_heads
        self.num_attention_heads = num_attention_heads
        # Linear module fo the triplet : query, key, value
        self.linear_layers = nn.ModuleList([nn.Linear(input_dim, d_model) for _ in range(3)])
        # Linear module for attention vector output
        self.output_linear = nn.Linear(d_model, d_model)
        # Attention layer
        self.attention = Attention()
        self.dropout = nn.Dropout(p=attention_dropout_prob)

    def forward(self, query, key, value, mask=None, output_attention=False):
        batch_size = query.size(0)
        # 1) Do all the linear projections in batch from d_model => num_attention_heads x_in d_k
        query, key, value = [linear(x).view(batch_size, -1, self.num_attention_heads, self.d_k).transpose(1, 2)
                             for linear, x in zip(self.linear_layers, (query, key, value))]
        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, attention_dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_attention_heads * self.d_k)
        # 4) Return linear transform of attention vector
        if output_attention:
            return self.output_linear(x), attn
        return self.output_linear(x), []


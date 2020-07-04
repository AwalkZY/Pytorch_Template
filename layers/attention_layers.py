import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.asserter import assert_param
from utils.helper import sequence_mask


class TanhAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        self.ws1 = nn.Linear(d_model, d_model, bias=True)
        self.ws2 = nn.Linear(d_model, d_model, bias=False)
        self.wst = nn.Linear(d_model, 1, bias=False)

    def reset_parameters(self):
        self.ws1.reset_parameters()
        self.ws2.reset_parameters()
        self.wst.reset_parameters()

    def forward(self, x, memory, memory_mask=None, fast_weights=None, **kwargs):
        if fast_weights is None:
            item1 = self.ws1(x)  # [nb, len1, d]
            item2 = self.ws2(memory)  # [nb, len2, d]
            # print(item1.shape, item2.shape)
            item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
            S_logit = self.wst(torch.tanh(item)).squeeze(-1)  # [nb, len1, len2]
        else:
            item1 = F.linear(x, fast_weights['ws1.weight'], fast_weights['ws1.bias'])  # [nb, len1, d]
            item2 = F.linear(memory, fast_weights['ws2.weight'])  # [nb, len2, d]
            # print(item1.shape, item2.shape)
            item = item1.unsqueeze(2) + item2.unsqueeze(1)  # [nb, len1, len2, d]
            S_logit = F.linear(torch.tanh(item), fast_weights['wst.weight']).squeeze(-1)  # [nb, len1, len2]
        if memory_mask is not None:
            memory_mask = memory_mask.unsqueeze(1)  # [nb, 1, len2]
            S_logit = S_logit.masked_fill(memory_mask == 0, float('-inf'))
        S = F.softmax(S_logit, -1)
        return torch.matmul(S, memory), S_logit  # [nb, len1, d], [nb, len1, len2]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_logit = self.attention_weight = None

    """
        First Config (For Multi-head Attention):
            Query: [batch_size, head_num, query_len, key_dim]
            Key: [batch_size, head_num, key_value_len, key_dim]
            Value: [batch_size, head_num, key_value_len, value_dim]
            Key_mask: [batch_size, head_num, query_len, key_value_len] / [batch_size, head_num, key_value_len]
        Second Config (For normal method):
            Query: [batch_size, query_len, key_dim]
            Key: [batch_size, key_value_len, key_dim]
            Value: [batch_size, key_value_len, value_dim]
            Key_mask: [batch_size, query_len, key_value_len] / [batch_size, key_value_len]
    """

    def forward(self, query, key, value, key_mask=None, dropout=None):
        if (key_mask is not None) and (key_mask.dim() != key.dim()):
            key_mask = key_mask.unsqueeze(1)
        out = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        # Out: [batch_size, head_num, query_len, key_len] / [bs, query_len, key_len]
        if key_mask is not None:
            out = out.masked_fill(key_mask == 0, -1e30)
        self.attention_logit = out
        attn = F.softmax(out, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        self.attention_weight = attn
        return torch.matmul(attn, value)


class SelfScaledDotAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = ScaledDotProductAttention()
        self.attention_logit = self.attention_weight = None

    def forward(self, inputs, inputs_mask=None, dropout=None):
        ans = self.attention(inputs, inputs, inputs, inputs_mask, dropout)
        self.attention_logit = self.attention.attention_logit
        self.attention_weight = self.attention.attention_weight
        return ans


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field='head_num', field_type=int)
        assert_param(param=params, field='input_dim', field_type=int)
        assert params['input_dim'] % params['head_num'] == 0
        if 'dropout' not in params:
            params['dropout'] = 0

        self.params = params
        self.params['hidden_dim'] = params['input_dim'] // params['head_num']
        self.query_head = nn.Linear(params['input_dim'], params['input_dim'])
        self.key_head = nn.Linear(params['input_dim'], params['input_dim'])
        self.value_head = nn.Linear(params['input_dim'], params['input_dim'])
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(p=params['dropout'])
        self.out = nn.Linear(params['input_dim'], params['input_dim'])
        self.attention_weight = None

    # Query: [batch_size, query_len, input_dim]
    # Key: [batch_size, key_value_len, input_dim]
    # Value: [batch_size, key_value_len, input_dim]
    # Key_mask: [batch_size, query_len, key_value_len] / [batch_size, key_value_len]
    def forward(self, query_input, key_input, value_input, key_mask=None):

        if (key_mask is not None) and (key_mask.dim() != key_input.dim()):
            key_mask = key_mask.unsqueeze(1)
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(1)
        batch_size = query_input.size(0)
        multi_head_query = self.query_head(query_input).contiguous().view(batch_size, -1, self.params['head_num'],
                                                                          self.params['hidden_dim']).transpose(1, 2)
        multi_head_key = self.key_head(key_input).contiguous().view(batch_size, -1, self.params['head_num'],
                                                                    self.params['hidden_dim']).transpose(1, 2)
        multi_head_value = self.value_head(value_input).contiguous().view(batch_size, -1, self.params['head_num'],
                                                                          self.params['hidden_dim']).transpose(1, 2)
        ans = self.attention(multi_head_query, multi_head_key, multi_head_value, key_mask, dropout=self.dropout)
        self.attention_weight = self.attention.attention_weight
        ans = ans.transpose(1, 2).contiguous().view(batch_size, -1, self.params['hidden_dim'] * self.params['head_num'])
        # Return: [batch_size, query_len, value_dim]
        return self.out(ans)


class SelfAttentionWithMultiHead(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field='head_num', field_type=int)
        assert_param(param=params, field='input_dim', field_type=int)

        self.params = params
        self.attention = MultiHeadAttention({
            'head_num': params['head_num'],
            'input_dim': params['input_dim']
        })
        self.attention_weight = None

    def forward(self, inputs, input_mask=None):
        ans = self.attention(inputs, inputs, inputs, input_mask)
        self.attention_weight = self.attention.attention_weight
        return ans

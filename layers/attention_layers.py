import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.asserter import assert_param
from utils.helper import sequence_mask


class TanhAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='memory_dim', field_type=int)
        assert_param(param=params, field='target_dim', field_type=int)

        self.input_part = nn.Linear(params['input_dim'], params['target_dim'], bias=True)
        self.memory_part = nn.Linear(params['memory_dim'], params['target_dim'], bias=False)
        self.final = nn.Linear(params['target_dim'], 1, bias=False)
        self.attention_weight = None

    def forward(self, inputs, memory, input_lengths=None, memory_lengths=None):
        inputs_compact = inputs.view(-1, inputs.size(-2), inputs.size(-1))
        memory_compact = memory.view(-1, memory.size(-2), memory.size(-1))
        input_item = self.input_part(inputs_compact)  # [batch_size, input_length, target_dimension]
        memory_item = self.memory_part(memory_compact)  # [batch_size, memory_length, target_dimension]
        items = input_item.unsqueeze(2) + memory_item.unsqueeze(1)
        result_shape = tuple(list(inputs.size()[:-1]) + [memory.size(-1)])
        score_shape = tuple(list(inputs.size()[:-1]) + [memory.size(-2)])
        # [batch_size, input_length, memory_length, target_dimension]
        attention_score = self.final(torch.tanh(items)).squeeze(-1)  # [batch_size, input_length, memory_length]
        if input_lengths is not None:
            input_lengths = input_lengths.contiguous().view(-1)
            score_mask = sequence_mask(input_lengths,
                                       attention_score.size(1)).unsqueeze(-1).repeat(1, 1, attention_score.size(-1))
            attention_score = attention_score.masked_fill(score_mask == 0, -1e30)
        if memory_lengths is not None:
            memory_lengths = memory_lengths.contiguous().view(-1)
            score_mask = sequence_mask(memory_lengths,
                                       attention_score.size(2)).unsqueeze(1).repeat(1, attention_score.size(1), 1)
            attention_score = attention_score.masked_fill(score_mask == 0, -1e30)
        attention_weight = F.softmax(attention_score, -1)
        attention_result = torch.matmul(attention_weight,
                                        memory_compact).view(result_shape)
        self.attention_weight = attention_weight.view(score_shape)
        return attention_result  # [batch_size, input_length, target_dimension]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_weight = None
    """
        First Config (For Multi-head Attention):
            Query: [batch_size, head_num, time_step, key_dim]
            Key: [batch_size, head_num, time_step, key_dim]
            Value: [batch_size, head_num, time_step, value_dim]
            Key_mask: [batch_size, time_step, time_step]
        Second Config (For normal method):
            Query: [batch_size, time_step, key_dim]
            Key: [batch_size, time_step, key_dim]
            Value: [batch_size, time_step, value_dim]
            Key_mask: [batch_size, time_step]
    """

    def forward(self, query, key, value, key_mask=None, dropout=None):
        out = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        # Out: [batch_size, head_num, time_step, time_step]
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(1)
            out = out.masked_fill(key_mask == 0, -1e30)
        attn = F.softmax(out, dim=-1)
        if dropout is not None:
            attn = dropout(attn)
        self.attention_weight = attn
        return torch.matmul(attn, value)


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

    # Query: [batch_size, time_step, input_dim]
    # Key: [batch_size, time_step, input_dim]
    # Value: [batch_size, time_step, input_dim]
    # Key_mask: [batch_size, time_step, time_step]
    def forward(self, query_input, key_input, value_input, key_mask=None):
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
        # Return: [batch_size, time_step, value_dim]
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

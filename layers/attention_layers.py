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

    def forward(self, inputs, memory, input_lengths=None, memory_lengths=None):
        input_item = self.input_part(inputs)  # [batch_size, input_length, target_dimension]
        memory_item = self.memory_part(memory)  # [batch_size, memory_length, target_dimension]
        items = input_item.unsqueeze(2) + memory_item.unsqueeze(1)
        # [batch_size, input_length, memory_length, target_dimension]
        attention_score = self.final(torch.tanh(items)).squeeze(-1)  # [batch_size, input_length, memory_length]
        if input_lengths is not None:
            input_lengths = input_lengths.contiguous().view(-1)
            score_mask = sequence_mask(input_lengths, attention_score.size(1)).unsqueeze(-1).repeat(1, 1, attention_score.size(-1))
            attention_score = attention_score.masked_fill(score_mask == 0, -1e30)
        if memory_lengths is not None:
            memory_lengths = memory_lengths.contiguous().view(-1)
            score_mask = sequence_mask(memory_lengths, attention_score.size(2)).unsqueeze(1).repeat(1, attention_score.size(1), 1)
            attention_score = attention_score.masked_fill(score_mask == 0, -1e30)
        attention_weight = F.softmax(attention_score, -1)
        return torch.matmul(attention_weight, memory)  # [batch_size, input_length, target_dimension]


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    # Query: [batch_size, time_step, query_len, key_dim]
    # Key: [batch_size, time_step, key_len, key_dim]
    # Value: [batch_size, time_step, key_len, value_dim]
    # Key_mask: [batch_size, key_len]
    def forward(self, query, key, value, key_mask=None):
        out = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        # Out: [batch_size, time_step, query_len, key_len]
        if key_mask is not None:
            key_mask = key_mask.unsqueeze(1).unsqueeze(1)   # Key_mask: [batch_size, 1, 1, key_len]
            out = out.masked_fill(key_mask == 0, -1e30)
        attn = F.softmax(out, dim=-1)
        return torch.matmul(attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field='head_num', field_type=int)
        assert_param(param=params, field='key_dim', field_type=int)
        assert_param(param=params, field='value_dim', field_type=int)

        self.params = params
        self.query_head = nn.Linear(params['key_dim'], params['key_dim'] * params['head_num'])
        self.key_head = nn.Linear(params['key_dim'], params['key_dim'] * params['head_num'])
        self.value_head = nn.Linear(params['value_dim'], params['value_dim'] * params['head_num'])
        self.attention = ScaledDotProductAttention()
        self.out = nn.Linear(params['value_dim'] * params['head_num'], params['value_dim'])

    # Query: [batch_size, time_step, query_len, key_dim]
    # Key: [batch_size, time_step, key_len, key_dim]
    # Value: [batch_size, time_step, key_len, value_dim]
    # Key_mask: [batch_size, key_len]
    def forward(self, query_input, key_input, value_input, key_mask=None):
        batch_size = query_input.size(0)
        multi_head_query = self.query_head(query_input).contiguous().view(batch_size, -1, self.params['head_num'],
                                                                          self.params['key_dim']).transpose(1, 2)
        multi_head_key = self.key_head(key_input).contiguous().view(batch_size, -1, self.params['head_num'],
                                                                    self.params['key_dim']).transpose(1, 2)
        multi_head_value = self.value_head(value_input).contiguous().view(batch_size, -1, self.params['head_num'],
                                                                          self.params['value_dim']).transpose(1, 2)
        ans = self.attention(multi_head_query, multi_head_key, multi_head_value, key_mask)
        ans = ans.transpose(1, 2).contiguous().view(batch_size, -1, self.params['value_dim'] * self.params['head_num'])
        # Return: [batch_size, time_step, query_len, value_dim]
        return self.out(ans)


class SelfAttentionWithMultiHead(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field='head_num', field_type=int)
        assert_param(param=params, field='input_dim', field_type=int)

        self.params = params
        self.attention = MultiHeadAttention({
            'head_num': params['head_num'],
            'key_dim': params['input_dim'],
            'value_dim': params['input_dim']
        })

    def forward(self, inputs, input_mask=None):
        return self.attention(inputs, inputs, inputs, input_mask)

import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F

from layers.attention_layers import TanhAttention
from utils.asserter import assert_param
from utils.organizer import configOrganizer

default_params = {
    "layer_num": 1,
    "rnn_drop_prob": 0,
    "drop_prob": 0
}


class DynamicDecoder(nn.Module):
    def __init__(self, param_name):
        params = configOrganizer.fetch_config(param_name)

        assert_param(param=params, field='hidden_dim', field_type=int)
        assert_param(param=params, field='use_attention', field_type=bool)
        if params['use_attention']:
            assert_param(param=params, field='memory_dim', field_type=int)
        assert_param(param=params, field='batch_size', field_type=int)
        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='is_bidirectional', field_type=bool)

        super().__init__()
        params.update(default_params)
        self.params = params
        self.rnn = nn.GRU(input_size=self.params['input_dim'],
                          hidden_size=self.params['hidden_dim'],
                          num_layers=self.params['num_layers'],
                          dropout=self.params['rnn_drop_prob'],
                          batch_first=True,
                          bidirectional=self.params['is_bidirectional'])
        self.attention = TanhAttention({
            'input_dim': self.params['input_dim'],
            'memory_dim': self.params['memory_dim'],
            'target_dim': self.params['hidden_dim']
        }) if self.params['use_attention'] else None
        self.mixture = nn.Linear(self.params['hidden_dim'], self.params['hidden_dim']) \
            if not self.params['use_attention'] else None

        self.rnn.flatten_parameters()

    def forward(self, input_sequences, input_lengths, state, encoder_outputs=None):
        """
        :param input_sequences: Sequences in shape (batch_size, time_step, embedding_dim)
        :param input_lengths: Lengths of sequences (unsorted) in shape (batch_size)
        :param state: last hidden state in shape (direction_num, batch_size, hidden_dim)
        :param encoder_outputs: encoder outputs from Encoder, in shape (batch_size, time_step, hidden_dim)
        :return:
            output: output of current unit in shape (batch_size, time_step, hidden_dim)
            state: current hidden state
        """
        intermediate_result = self.attention(input_sequences, encoder_outputs, input_lengths) \
            if self.params['use_attention'] else self.mixture(input_sequences)
        max_num_steps = intermediate_result.size(1)
        input_lengths, sorted_indices = torch.sort(input_lengths, descending=True)
        _, original_idx = torch.sort(sorted_indices, dim=0, descending=False)
        sorted_indices = sorted_indices.to(intermediate_result.device)
        batch_dim = 0  # batch_first
        intermediate_result = intermediate_result.index_select(batch_dim, sorted_indices)
        packed_input = utils.rnn.pack_padded_sequence(input=intermediate_result, lengths=input_lengths,
                                                      batch_first=True)
        output, state = self.rnn(packed_input, state)
        output, output_length = utils.rnn.pad_packed_sequence(sequence=output, batch_first=True)
        output = output.index_select(batch_dim, original_idx.cuda(device=intermediate_result.device))
        if output.shape[1] < max_num_steps:
            output = F.pad(output, [0, 0, 0, max_num_steps - output.shape[1]])
        return output, state

    def sample(self, inputs, state, encoder_outputs=None):
        """
            :param inputs: Current Input in shape (batch_size, 1, embedding_dim)
            :param state: last hidden state in shape (direction_num, batch_size, hidden_dim)
            :param encoder_outputs: encoder outputs from Encoder, in shape (batch_size, time_step, hidden_dim)
            :return:
                output: output of current unit in shape (batch_size, 1, hidden_dim)
                state: current hidden state
        """
        intermediate_result = self.attention(inputs, encoder_outputs) \
            if self.params['use_attention'] else self.mixture(inputs)
        output, state = self.rnn(intermediate_result, state)
        return output, state

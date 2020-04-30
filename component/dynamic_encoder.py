import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F

from utils.asserter import assert_param
from utils.container import configContainer

default_params = {
    "layer_num": 1,
    "rnn_drop_prob": 0,
    "drop_prob": 0
}


class DynamicEncoder(nn.Module):
    def __init__(self, params):
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configContainer.fetch_config(params)

        assert_param(param=params, field='hidden_dim', field_type=int)
        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='batch_size', field_type=int)
        assert_param(param=params, field='is_bidirectional', field_type=bool)

        super().__init__()
        params.update(default_params)
        self.params = params
        self.rnn = nn.GRU(input_size=self.params['input_dim'],
                          hidden_size=self.params['hidden_dim'],
                          num_layers=self.params['layer_num'],
                          dropout=self.params['rnn_drop_prob'],
                          batch_first=True,
                          bidirectional=self.params['is_bidirectional'])
        self.rnn.flatten_parameters()

    def forward(self, input_sequence, input_lengths, state):
        """
        :param input_sequence: Sequences in shape (batch_size, time_step, embedding_dim)
        :param input_lengths: Lengths of sequences (unsorted) in shape (batch_size)
        :param state: previous state of the encoder in shape (direction_num, batch_size, hidden_dim)
        :return:
            output_vector: GRU output in shape (batch_size, time_step, hidden_dim * direction_num)
            state: current state of the encoder in shape (direction_num, batch_size, hidden_dim)
        """
        max_time_step = input_sequence.size(1)
        input_lengths, sorted_indices = torch.sort(input_lengths, descending=True)
        _, original_idx = torch.sort(sorted_indices, dim=0, descending=False)
        sorted_indices = sorted_indices.to(input_sequence.device)
        batch_dim = 0  # batch_first
        input_sequence = input_sequence.index_select(batch_dim, sorted_indices)

        packed = utils.rnn.pack_padded_sequence(input=input_sequence, lengths=input_lengths,
                                                batch_first=True)
        output, state = self.rnn(packed, state)
        output, output_length = utils.rnn.pad_packed_sequence(sequence=output, batch_first=True)
        output = output.index_select(batch_dim, original_idx.cuda(device=input_sequence.device))
        if output.shape[1] < max_time_step:
            output = F.pad(output, [0, 0, 0, max_time_step - output.shape[1]])
        return output, state

    def init_state(self, in_batch=True):
        """
        :return:  initial state of the encoder in shape (direction_num, batch_size, hidden_dim)
        """
        if self.params['is_bidirectional']:
            return torch.zeros(2, self.params['batch_size'] if in_batch else 1, self.params['hidden_dim'])
        return torch.zeros(1, self.params['batch_size'] if in_batch else 1, self.params['hidden_dim'])

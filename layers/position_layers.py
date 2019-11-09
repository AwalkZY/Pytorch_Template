from torch import nn
import torch
import torch.nn.functional as F
import math

from utils.asserter import assert_param


class PositionEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field_type=int, field="input_dim")
        assert_param(param=params, field_type=int, field="length")
        input_dim, length = params["input_dim"], params["length"]
        frequency = torch.tensor(
            [10000 ** (-i / input_dim) if i % 2 == 0 else -10000 ** ((1 - i) / input_dim) for i in range(input_dim)]) \
            .unsqueeze(dim=1)
        phases = torch.tensor([0 if i % 2 == 0 else math.pi / 2 for i in range(input_dim)]).unsqueeze(dim=1)
        pos = torch.arange(length).repeat(input_dim, 1).float()
        self.pos_encoding = torch.sin(torch.add(torch.mul(pos, frequency), phases))
        self.pos_encoding = self.pos_encoding.transpose(1, 0)
        self.pos_encoding = nn.Parameter(self.pos_encoding, requires_grad=False)

    def forward(self, x):
        return x + self.pos_encoding


class PositionWiseFeedForward(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field_type=int, field="input_dim")
        assert_param(param=params, field_type=int, field="hidden_dim")
        if "dropout" not in params:
            params["dropout"] = 0.0
        input_dim, hidden_dim, dropout = params["input_dim"], params["hidden_dim"], params["dropout"]
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x

import torch
import torch.nn as nn

from utils.asserter import assert_param


class CrossGate(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field="input_dim", field_type=int)
        self.linear_transformation1 = nn.Linear(params["input_dim"], params["input_dim"], bias=True)
        self.linear_transformation2 = nn.Linear(params["input_dim"], params["input_dim"], bias=True)

    def forward(self, x1, x2):
        g1 = torch.sigmoid(self.linear_transformation1(x1))
        h2 = g1 * x2
        g2 = torch.sigmoid(self.linear_transformation2(x2))
        h1 = g2 * x1
        return torch.cat([h1, h2], dim=-1)
        # return h1, h2

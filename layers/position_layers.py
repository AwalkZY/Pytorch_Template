from torch import nn
import torch
import torch.nn.functional as F
import math

from utils.asserter import assert_param


class PositionEncoder(nn.Module):
    """Implement the PE function."""
    def __init__(self, params):
        assert_param(param=params, field_type=int, field="input_dim")
        if "dropout" not in params:
            params["dropout"] = 0.0
        if "max_len" not in params:
            params["max_len"] = 5000
        super().__init__()
        self.dropout = nn.Dropout(p=params['dropout'])

        # Compute the positional encodings once in log space.
        pos_encoding = torch.zeros(params['max_len'], params['input_dim'])
        position = torch.arange(0, params['max_len']).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, params['input_dim'], 2) *
                             -(math.log(10000.0) / params['input_dim']))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0).clone().detach().requires_grad_(False)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1)]
        return self.dropout(x)


class PositionWiseFeedForward(nn.Module):
    """Implements FFN equation."""

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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(15, 5))
    pe = PositionEncoder({
        "input_dim": 20,
        "max_len": 5000
    })
    y = pe.forward(torch.zeros(1, 100, 20))
    plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    plt.legend(["dim %d" % p for p in [4, 5, 6, 7]])
    plt.show()

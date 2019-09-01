import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        raise NotImplementedError

    def sample(self, *inputs):
        raise NotImplementedError

    def calculate_loss(self, *inputs):
        raise NotImplementedError

import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        raise NotImplementedError

    def inference(self, *inputs):
        raise NotImplementedError

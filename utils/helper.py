import torch
from torch import nn
import numpy as np


def sequence_mask(lengths, max_length, dtype=torch.bool):
    if max_length is None:
        max_length = lengths.max()
    inter = torch.ones((len(lengths), max_length)).to(device=lengths.device).cumsum(dim=1).t() > lengths.type(
        torch.float32)
    mask = (1 - inter).t().type(dtype)
    return mask


def sample(data, weight, num_sample, replace=True):
    weight = weight.cpu().numpy()
    weight = weight / np.sum(weight)
    return torch.tensor(np.random.choice(data, num_sample, replace=replace, p=weight))

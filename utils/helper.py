import torch
from torch import nn
import numpy as np
import copy
from utils.accessor import load_json


def clones(module, number):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(number)])


def sequence_mask(lengths, max_length, dtype=torch.bool):
    """Generate sequence masks from variant lengths"""
    if max_length is None:
        max_length = lengths.max()
    inter = torch.ones((len(lengths), max_length)).to(device=lengths.device).cumsum(dim=1).t() > lengths.type(
        torch.float32)
    mask = (1 - inter).t().type(dtype)
    return mask


def no_peak_mask(length):
    """Generate mask to avoid Attention Modules attend to unpredicted positions"""
    np_mask = np.triu(np.ones((1, length, length)), k=1).astype('uint8')
    mask = torch.from_numpy(np_mask) == 0
    return mask


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_masks = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_masks) == 0


def sample(data, weight, num_sample, replace=True):
    weight = weight.cpu().numpy()
    weight = weight / np.sum(weight)
    return torch.tensor(np.random.choice(data, num_sample, replace=replace, p=weight))

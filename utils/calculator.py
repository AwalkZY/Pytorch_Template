import torch
import numpy as np


def calculate_dist(x, y, dist='Euclid'):
    assert x.size(-1) == y.size(-1), "Incompatible dimension of two matrix! {} and {} are given. ".format(x.size(),
                                                                                                          y.size())
    assert dist in ['Euclid', 'Cosine'], "Invalid distance criterion!"
    d = x.size(-1)
    x = x.contiguous().view(-1, d)
    y = y.contiguous().view(-1, d)
    m, n = x.size(0), y.size(0)
    # x : m*d, y : n*d
    x = torch.unsqueeze(x, dim=1).expand(m, n, d)
    y = torch.unsqueeze(y, dim=0).expand(m, n, d)
    if dist == 'Euclid':
        return torch.dist(x, y, p=2)
    elif dist == 'Cosine':
        return 1 - torch.cosine_similarity(x, y, dim=2)  # shape: m*n


def sample(data, weight, num_sample, replace=True):
    weight = weight.cpu().numpy()
    weight = weight / np.sum(weight)
    return torch.tensor(np.random.choice(data, num_sample, replace=replace, p=weight))

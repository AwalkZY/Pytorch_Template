import torch


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


def calculate_iou(box0, box1):
    box0 = box0.contiguous().view(-1, 4)
    box1 = box1.contiguous().view(-1, 4)
    l0, t0, r0, b0 = box0[:, 0], box0[:, 1], box0[:, 0] + box0[:, 2], box0[:, 1] + box0[:, 3]
    l1, t1, r1, b1 = box1[:, 0], box1[:, 1], box1[:, 0] + box1[:, 2], box1[:, 1] + box1[:, 3]
    width = torch.min(r0, r1) - torch.max(l0, l1)
    width[width < 0] = 0
    height = torch.min(b0, b1) - torch.max(t0, t1)
    height[height < 0] = 0
    intersection = width * height
    union = (b0 - t0) * (r0 - l0) + (b1 - t1) * (r1 - l1) - width * height + 1e-15
    return torch.mean(intersection / union)

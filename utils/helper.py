import torch


def sequence_mask(lengths, max_length, dtype=torch.bool):
    if max_length is None:
        max_length = lengths.max()
    inter = torch.ones((len(lengths), max_length)).to(device=lengths.device).cumsum(dim=1).t() > lengths.type(
        torch.float32)
    mask = (1 - inter).t().type(dtype)
    return mask

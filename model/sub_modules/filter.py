import torch
from torch import nn

from component.net_vlad import NetVLAD
from utils.calculator import max_min_norm


class Filter(nn.Module):
    def __init__(self, hidden_size, cluster_size):
        super().__init__()
        self.word_vlad = NetVLAD(cluster_size=cluster_size, feature_size=hidden_size)
        self.word_linear = nn.Linear(cluster_size * hidden_size, hidden_size)
        self.gate1 = nn.Linear(hidden_size, hidden_size)
        self.gate2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, frames_feat, words_feat, words_mask):
        frames_feat, words_feat = frames_feat.detach(), words_feat.detach()
        words_global = self.word_vlad(words_feat, words_mask, flatten=True)
        words_global = self.word_linear(words_global)
        words_global = words_global.unsqueeze(1)
        # [B, L, D] -> [B, 1, D]
        fuse_feat = frames_feat * words_global
        # weight = sigmoid(W_T*tanh(W_T*(V·S)))
        resp = torch.tanh(self.gate1(fuse_feat))
        resp = torch.sigmoid(self.gate2(resp))  # [B, L, 1]
        resp = resp.squeeze()
        resp = max_min_norm(resp, dim=-1)
        return resp.unsqueeze(-1)


def build_filter(cfg) -> Filter:
    return Filter(cfg.HIDDEN_SIZE, cfg.CLUSTER_SIZE)

import math

import torch
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np
from layers.attention_layers import TanhAttention, SelfAttentionWithMultiHead, SelfScaledDotAttention
from layers.gates import CrossGate
from utils.calculator import max_min_norm


def sequence_mask(lengths, max_length):
    """Generate sequence masks from variant lengths"""
    if max_length is None:
        max_length = lengths.max()
    if isinstance(lengths, torch.Tensor):
        inter = torch.ones((len(lengths), max_length)).to(device=lengths.device).cumsum(dim=1).t() > lengths.type(
            torch.float32)
        mask = (~inter).t().type(torch.bool)
    elif isinstance(lengths, np.ndarray):
        inter = np.ones((len(lengths), max_length)).cumsum(axis=1).T > lengths.astype(np.float32)
        mask = (~inter).T.astype(np.bool)
    else:
        raise NotImplementedError
    return mask


class BottleNeck(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                      kernel_size=1, padding=0, stride=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                      kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm1d(feature_dim)
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                      kernel_size=1, padding=0, stride=2),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        return torch.relu(self.bottleneck(x) + self.shortcut(x))


class LateralLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.core = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                              kernel_size=1, padding=0, stride=1)

    def forward(self, last_feature, fuse_feature):
        length = fuse_feature.size(-1)
        return F.interpolate(last_feature, size=length, mode='linear') + self.core(fuse_feature)


class VisualPyramid(nn.Module):
    def __init__(self, feature_dim, frame_num, salient_segment):
        super().__init__()
        self.salient_segment = salient_segment
        self.conv1 = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                               kernel_size=3, padding=1, stride=1)
        self.block_num = int(round(math.log2(frame_num)))
        self.bottleneck_blocks = nn.ModuleList([
            BottleNeck(feature_dim) for _ in range(self.block_num)
        ])
        self.top_layer = nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                                   kernel_size=1, padding=0, stride=1)
        self.lateral_layer = nn.ModuleList([
            LateralLayer(feature_dim) for _ in range(self.block_num - 1)
        ])
        self.smooth_layer = nn.ModuleList([
            nn.Conv1d(in_channels=feature_dim, out_channels=feature_dim,
                      kernel_size=3, padding=1, stride=1)
            for _ in range(self.block_num - 1)
        ])
        self.attention = SelfScaledDotAttention()
        self.cross_gate = CrossGate({"input_dim": feature_dim})
        self.score_linear = nn.Linear(2 * feature_dim, 1)

    def forward(self, visual_feat):
        """
        :param visual_feat: input features in shape (batch_size, feature_dim, visual_len)
        :return: List of fused features
        """
        bottom_up = []
        top_down = []
        visual_feat = torch.relu(self.conv1(visual_feat))
        for block_idx, block in enumerate(self.bottleneck_blocks):
            visual_feat = block(visual_feat)
            bottom_up.append(visual_feat.clone())
        top_down.append(self.top_layer(visual_feat))
        for layer_idx, layer in enumerate(self.lateral_layer):
            invert_idx = self.block_num - layer_idx - 2
            top_down.append(layer(top_down[-1], bottom_up[invert_idx]))
        for layer_idx, layer in enumerate(self.smooth_layer):
            top_down[layer_idx] = layer(top_down[layer_idx])
        features, feature_masks = fpn_process(top_down)
        features = features.to(visual_feat.device)
        feature_masks = feature_masks.to(visual_feat.device)
        if self.salient_segment:
            for layer_idx in range(self.block_num):
                # 首先计算聚合特征，然后用cross gate融合当前特征和对应的聚合特征，以此计算当前帧的得分
                agg_feat = self.attention(features[:, layer_idx], feature_masks[:, layer_idx])
                cross_feat = self.cross_gate(features[:, layer_idx], agg_feat)
                scores = torch.sigmoid(self.score_linear(cross_feat))
                features[:, layer_idx] = features[:, layer_idx] * scores
        return features, feature_masks


def fpn_process(feature_list):
    batch_size, feature_dim, max_len = feature_list[-1].size()
    lengths = torch.zeros(len(feature_list))
    features = torch.zeros(batch_size, len(feature_list), feature_dim, max_len)
    for feature_idx, feature in enumerate(feature_list):
        lengths[feature_idx] = feature.size(-1)
        features[:, feature_idx, :, :feature.size(-1)] = feature
    feature_mask = sequence_mask(lengths, max_len)
    feature_mask = feature_mask.unsqueeze(dim=0).repeat(batch_size, 1, 1)
    features = features.transpose(-1, -2)
    return features, feature_mask
    # feature_mask in shape (batch_size, len(feature_list), max_len)
    # feature in shape (batch_size, len(feature_list), max_len, feature_dim)


def build_visual_pyramid(cfg) -> VisualPyramid:
    return VisualPyramid(cfg.FRAME_DIM, cfg.MAX_FRAME_NUM, cfg.SALIENT_SEGMENT)

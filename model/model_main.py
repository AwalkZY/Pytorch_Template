import torch
import torch.nn.functional as F

from torch import nn
from model.model_base import BaseModel
from model.sub_modules.filter import build_filter
from model.sub_modules.fusion import build_fusion
from model.sub_modules.query_encoder import build_query_encoder
from model.sub_modules.feature_pyramid import build_visual_pyramid
from model.sub_modules.scorer import build_scorer
from model.sub_modules.sparse_prop import build_sparse_prop
from utils.slimer import expand_and_repeat, union_dim, split_dim


def expand_query(words_encoded, words_mask, scale_num):
    exp_words_encoded = expand_and_repeat(words_encoded, 1, scale_num)
    exp_words_mask = expand_and_repeat(words_mask, 1, scale_num)
    # in shape (nb, ns, nw, nd), (nb, ns, nw)
    exp_words_encoded = union_dim(exp_words_encoded, 0, 1)
    exp_words_mask = union_dim(exp_words_mask, 0, 1)
    return exp_words_encoded, exp_words_mask
    # in shape (nb * ns, nw, nd), (nb * ns, nw)


def flatten_video(frames_encoded, frames_mask):
    flat_frames_encoded = union_dim(frames_encoded, 0, 1)
    flat_frames_mask = union_dim(frames_mask, 0, 1)
    return flat_frames_encoded, flat_frames_mask
    # in shape (nb * ns, nc, nd), (nb * ns, nc)


class MainModel(BaseModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.filter_branch = config.USE_FILTER
        self.dropout_layer = nn.Dropout(config.DROPOUT)
        self.video_encoder = build_visual_pyramid(config.VIDEO_ENCODER)
        self.query_encoder = build_query_encoder(config.QUERY_ENCODER)
        self.back = nn.Parameter(torch.zeros(1, 1, config.MODEL_SIZE), requires_grad=False)
        self.fusion = build_fusion(config.FUSION)
        self.filter = build_filter(config.FILTER)
        self.prop = build_sparse_prop(config.PROP)
        self.scorer = build_scorer(config.SCORER)

    def forward(self, frames_feat, words_feat, words_mask, props):
        batch_size = frames_feat.size(0)
        frames_encoded, frames_mask = self.video_encoder(frames_feat)
        scale_num = frames_encoded.size(1)
        """ Step 1:
            * 用金字塔提取原始特征，得到 (nb, ns, nc, nd）的特征和 (nb, ns, nc) 的掩码
            * 在参数中打开Salient Segment开关可以根据帧的表达计算显著性得分
            * 金字塔的输出大小是上小下大的
        """
        words_encoded = self.query_encoder(words_feat, words_mask)
        """ Step 2:
            * 用 Transformer 或 GRU 计算每个位置语句的融合特征，具体组件可在开关中选择
        """
        flat_words_encoded, flat_words_mask = expand_query(words_encoded, words_mask, frames_encoded.size(1))
        flat_frames_encoded, flat_frames_mask = flatten_video(frames_encoded, frames_mask)
        if self.filter_branch:
            weight = self.filter(flat_frames_encoded, flat_words_encoded, flat_words_mask)
            fuse_feat = self.fusion(flat_frames_encoded * weight + self.back * (1 - weight),
                                    flat_words_encoded, flat_words_mask)
        else:
            fuse_feat = self.fusion(flat_frames_encoded, flat_words_encoded, flat_words_mask)
        fuse_feat = self.dropout_layer(fuse_feat)  # [B * S, L, D]
        fuse_feat = split_dim(fuse_feat.transpose(1, 2), 0, batch_size, scale_num)
        # [B * S, L, D] -> [B * S, D, L] -> [B, S, D, L]
        map_feat, map_mask = self.prop(fuse_feat)
        """ Step 3: 计算融合特征并生成特征图 """
        score = self.scorer(map_feat, map_mask, props)
        """ Step 4: 计算评分 """
        return score

    def inference(self, *inputs):
        pass

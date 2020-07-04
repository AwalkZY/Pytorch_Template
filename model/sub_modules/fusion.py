import torch
from torch import nn

from component.transformer import TransformerEncoder
from layers.attention_layers import TanhAttention
from layers.dynamic_rnn import DynamicGRU
from layers.gates import CrossGate


class Fusion(nn.Module):
    def __init__(self, hidden_size, encoder_type):
        super(Fusion, self).__init__()
        self.fuse_attn = TanhAttention(hidden_size)
        self.fuse_gate = CrossGate(hidden_size)
        self.encoder_type = encoder_type
        if encoder_type == "Transformer":
            self.fuse_agg = TransformerEncoder({
                "input_dim": hidden_size * 2,
                "encoder_layer_num": 1
            })
        elif encoder_type == "GRU":
            self.fuse_agg = DynamicGRU(input_size=hidden_size * 2,
                                       hidden_size=hidden_size // 2,
                                       num_layers=1,
                                       bidirectional=True,
                                       batch_first=True)

    def forward(self, visual_input, textual_input, textual_mask):
        agg_txt_h, _ = self.fuse_attn(visual_input, textual_input, textual_mask)
        visual_h, agg_txt_h = self.fuse_gate(visual_input, agg_txt_h)
        x = torch.cat([visual_h, agg_txt_h], -1)
        enc = self.fuse_agg(x, None)
        return enc


def build_fusion(cfg) -> Fusion:
    return Fusion(cfg.HIDDEN_SIZE, cfg.ENCODER_TYPE)

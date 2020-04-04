from torch import nn
import math
import torch
import torch.nn.functional as F

from layers.attention_layers import SelfAttentionWithMultiHead
from layers.position_layers import PositionWiseFeedForward
from utils.asserter import assert_param
from utils.organizer import configOrganizer
from utils.processor import clones
from layers.blocks import LayerNormResidualBlock


class EncoderFromLayer(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, number):
        super().__init__()
        self.layers = clones(layer, number)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderFromLayer(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, number):
        super().__init__()
        self.layers = clones(layer, number)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward=None, dropout=0.0):
        super().__init__()
        self.size = size
        self.self_attn = LayerNormResidualBlock(layer=self_attn,
                                                size=self.size,
                                                drop_ratio=dropout)
        self.feed_forward = LayerNormResidualBlock(layer=feed_forward,
                                                   size=self.size,
                                                   drop_ratio=dropout)

    def forward(self, x, mask):
        x = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, source_attn, feed_forward=None, dropout=0.0):
        super().__init__()
        self.size = size
        self.self_attn = LayerNormResidualBlock(layer=self_attn,
                                                size=self.size,
                                                drop_ratio=dropout)
        self.feed_forward = LayerNormResidualBlock(layer=feed_forward,
                                                   size=self.size,
                                                   drop_ratio=dropout)
        self.source_attn = LayerNormResidualBlock(layer=source_attn,
                                                  size=self.size,
                                                  drop_ratio=dropout)

    def forward(self, x, memory, source_mask, target_mask):
        x = self.self_attn(x, target_mask)
        x = self.source_attn(x, memory, memory, source_mask)
        x = self.feed_forward(x)
        return x


default_params = {
    "head_num": 4,
    "encoder_layers_num": 6,
    "decoder_layers_num": 6
}


class Embeddings(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert_param(param=params, field_type=int, field="embedding_dim")
        assert_param(param=params, field_type=int, field="vocab")

        self.params = params
        self.embedding_layer = nn.Embedding(params['vocab'], params['embedding_dim'])

    def forward(self, x):
        return self.embedding_layer(x) * math.sqrt(self.params['embedding_dim'])


# Source and Target should be embedded into the same dimension
class Transformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configOrganizer.fetch_config(params)

        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='hidden_dim', field_type=int)
        if "dropout" not in params:
            params["dropout"] = 0.0

        self.params = default_params
        self.params.update(params)
        self.encoder = EncoderFromLayer(EncoderLayer(self.params["input_dim"],
                                                     SelfAttentionWithMultiHead({
                                                         "head_num": self.params["head_num"],
                                                         "input_dim": self.params["input_dim"],
                                                         "dropout": self.params["dropout"]
                                                     }),
                                                     PositionWiseFeedForward({
                                                         "input_dim": self.params["input_dim"],
                                                         "hidden_dim": self.params["hidden_dim"],
                                                         "dropout": self.params["dropout"]
                                                     })),
                                        self.params["encoder_layers_num"])
        self.decoder = DecoderFromLayer(DecoderLayer(self.params["input_dim"],
                                                     SelfAttentionWithMultiHead({
                                                         "head_num": self.params["head_num"],
                                                         "input_dim": self.params["input_dim"],
                                                         "dropout": self.params["dropout"]
                                                     }),
                                                     SelfAttentionWithMultiHead({
                                                         "head_num": self.params["head_num"],
                                                         "input_dim": self.params["input_dim"],
                                                         "dropout": self.params["dropout"]
                                                     }),
                                                     PositionWiseFeedForward({
                                                         "input_dim": self.params["input_dim"],
                                                         "hidden_dim": self.params["hidden_dim"],
                                                         "dropout": self.params["dropout"]
                                                     })),
                                        self.params["decoder_layers_num"])

    def forward(self, source, source_mask, target, target_mask):
        x = self.encoder(source, source_mask)  # [nb, len1, hid]
        x = self.decoder(target, x, source_mask, target_mask)  # [nb, len2, hid]
        return x

    def encode(self, source, source_mask):
        return self.encoder(source, source_mask)

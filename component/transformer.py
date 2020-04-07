from torch import nn
import math
import torch
import torch.nn.functional as F

from layers.attention_layers import SelfAttentionWithMultiHead
from layers.position_layers import PositionWiseFeedForward, PositionEncoder
from utils.asserter import assert_param
from utils.organizer import configOrganizer
from utils.helper import clones
from layers.blocks import LayerNormResidualBlock


class EncoderFromLayer(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, number):
        super().__init__()
        self.layers = clones(layer, number)
        self.norm = nn.LayerNorm(layer.input_dim)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderFromLayer(nn.Module):
    """Generic N layer decoder with masking."""

    def __init__(self, layer, number):
        super().__init__()
        self.layers = clones(layer, number)
        self.norm = nn.LayerNorm(layer.input_dim)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, input_dim, self_attn, feed_forward=None, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.position_encoder = PositionEncoder({
            "input_dim": input_dim
        })
        self.self_attn = LayerNormResidualBlock(layer=self_attn,
                                                input_dim=self.input_dim,
                                                drop_ratio=dropout)
        self.feed_forward = LayerNormResidualBlock(layer=feed_forward,
                                                   input_dim=self.input_dim,
                                                   drop_ratio=dropout)

    def forward(self, x, mask, positional_encoding=True):
        if positional_encoding:
            x = self.position_encoder(x)
        x = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, self_attn, source_attn, feed_forward=None, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.position_encoder = PositionEncoder({
            "input_dim": input_dim
        })
        self.self_attn = LayerNormResidualBlock(layer=self_attn,
                                                input_dim=self.input_dim,
                                                drop_ratio=dropout)
        self.feed_forward = LayerNormResidualBlock(layer=feed_forward,
                                                   input_dim=self.input_dim,
                                                   drop_ratio=dropout)
        self.source_attn = LayerNormResidualBlock(layer=source_attn,
                                                  input_dim=self.input_dim,
                                                  drop_ratio=dropout)

    def forward(self, x, memory, source_mask, target_mask, positional_encoding=True):
        if positional_encoding:
            x = self.position_encoder(x)
        x = self.self_attn(x, target_mask)
        x = self.source_attn(x, memory, memory, source_mask)
        x = self.feed_forward(x)
        return x


default_params = {
    "head_num": 4,
    "encoder_layers_num": 6,
    "decoder_layers_num": 6
}


# Source and Target should be embedded into the same dimension
class TransformerCore(nn.Module):
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

    def forward(self, source, source_mask, target, target_mask, positional_encoding=True):
        encoding = self.encoder(source, source_mask, positional_encoding)  # [nb, len1, hid]
        x = self.decoder(target, encoding, source_mask, target_mask, positional_encoding)  # [nb, len2, hid]
        return x, encoding

    def encode(self, source, source_mask, positional_encoding=True):
        return self.encoder(source, source_mask, positional_encoding)


class Any2SentenceTransformer(TransformerCore):
    def __init__(self, params):
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configOrganizer.fetch_config(params)
        super().__init__(params)

        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='target_vocab', field_type=int)
        if "share_embedding" not in params:
            params["share_embedding"] = True

        self.out = nn.Linear(params['input_dim'], params['target_vocab'])
        if self.params["share_embedding"]:
            self.embedding_weight = self.out.weight * math.sqrt(params['input_dim'])
        else:
            self.embedding_layer = nn.Embedding(params['target_vocab'], params['input_dim'])
            self.embedding_weight = self.embedding_layer.weight

    def forward(self, source, source_mask, target, target_mask, positional_encoding=True):
        target = F.embedding(target, self.embedding_weight)
        result, encoding = super().forward(source, source_mask, target, target_mask, positional_encoding)
        scores = F.softmax(self.out(result), dim=-1)
        return scores, encoding

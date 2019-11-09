from torch import nn
from torch.nn import LayerNorm
import math
import torch
import torch.nn.functional as F
from layers.position_layers import PositionWiseFeedForward
from utils.asserter import assert_param
from utils.organizer import configOrganizer
from utils.processor import clones


class EncoderFromLayer(nn.Module):
    def __init__(self, layer, number):
        super().__init__()
        self.layers = clones(layer, number)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderFromLayer(nn.Module):
    def __init__(self, layer, number):
        super().__init__()
        self.layers = clones(layer, number)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, source_mask, target_mask):
        for layer in self.layers:
            x = layer(x, memory, source_mask, target_mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward=None, dropout=0.0):
        super().__init__()
        self.size = size
        self.norm1 = LayerNorm(size)
        if feed_forward is None:
            self.norm2 = None
        else:
            self.norm2 = LayerNorm(size)
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        res = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = (res + x) / math.sqrt(2)

        if self.feed_forward is not None:
            res = x
            x = self.feed_forward(self.norm2(x))
            x = self.dropout(x)
            x = (res + x) / math.sqrt(2)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, source_attn, feed_forward=None, dropout=0.0):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.source_attn = source_attn
        if feed_forward is None:
            self.norms = clones(LayerNorm(size), 2)
        else:
            self.norms = clones(LayerNorm(size), 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, source_mask, target_mask):
        res = x
        x = self.norms[0](x)
        x = self.self_attn(x, x, x, target_mask)
        x = self.dropout(x)
        x = (res + x) / math.sqrt(2)

        res = x
        x = self.norms[1](x)
        x = self.source_attn(x, memory, memory, source_mask)
        x = self.dropout(x)
        x = (res + x) / math.sqrt(2)

        if self.feed_forward is not None:
            res = x
            x = self.feed_forward(self.norms[2](x))
            x = self.dropout(x)
            x = (res + x) / math.sqrt(2)
        return x


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=None, element_width=None, head_width=None):
        super().__init__()
        self.dropout = dropout
        self.element_width = element_width
        self.head_width = head_width

    # Q: [nb, nh, len1, hid1], K: [nb, nh, len2, hid2], V: [nb, nh, len2, hid2], mask: [nb, len2]
    def forward(self, Q, K, V, mask=None):
        if self.head_width is None:
            if self.element_width is not None:
                K = F.pad(K, [0, 0, self.element_width, self.element_width])
                V = F.pad(V, [0, 0, self.element_width, self.element_width])
                mask = F.pad(mask, [self.element_width, self.element_width])
            out = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))  # [nb, nh, len1, len2]
            if mask is not None:
                mask = mask.unsqueeze(1)  # mask: [nb, 1, len2]
                mask = mask.unsqueeze(1)  # mask: [nb, 1, 1, len2]
                out = out.masked_fill(mask == 0, -1e30)  # [nb, nh, len1, len2]
            if self.element_width is not None:
                mask = torch.zeros(out.size(-2), out.size(-1))
                for i in range(self.element_width, out.size(-2) - self.element_width):
                    mask[i, i-self.element_width:i+self.element_width+1] = 1
                mask = mask.unsqueeze(0)
                mask = mask.unsqueeze(0)
                mask = mask.cuda()
                out = out.masked_fill(mask == 0, -1e30)
            attn = F.softmax(out, dim=-1)
            if self.dropout is not None:
                attn = self.dropout(attn)
            out = torch.matmul(attn, V)  # [nb, nh, len1, hid2]
        else:
            if self.element_width is not None:
                K = F.pad(K, [0, 0, self.element_width, self.element_width])
                V = F.pad(V, [0, 0, self.element_width, self.element_width])
                mask = F.pad(mask, [self.element_width, self.element_width])
            if self.head_width is not None:
                K = F.pad(K, [0, 0, 0, 0, self.head_width, self.head_width])
                V = F.pad(V, [0, 0, 0, 0, self.head_width, self.head_width])
                Q = F.pad(Q, [0, 0, 0, 0, self.head_width, self.head_width])
            element_mask = None
            if mask is not None:
                mask = mask.unsqueeze(1)  # mask: [nb, 1, len2]
                mask = mask.unsqueeze(1)  # mask: [nb, 1, 1, len2]
            if self.element_width is not None:
                element_mask = torch.zeros(Q.size(-2), K.size(-2))
                for i in range(self.element_width, Q.size(-2) - self.element_width):
                    element_mask[i, i-self.element_width:i+self.element_width+1] = 1
                element_mask = element_mask.unsqueeze(0)  # [1, len1, len2]
                element_mask = element_mask.unsqueeze(0)  # [1, 1, len1, len2]
                element_mask = element_mask.cuda()

            num_heads = Q.size(1)
            attn_matrices = []
            K_transpose = K.transpose(-2, -1)
            for h in range(self.head_width, num_heads - self.head_width):
                attn_matrix = torch.matmul(Q[:, h:h+1], K_transpose[:, h-self.head_width:h+self.head_width]) / math.sqrt(Q.size(-1))
                # h->h-n...h+n: [nb, nrh, len1, len2]
                if mask is not None:
                    attn_matrix = attn_matrix.masked_fill(mask == 0, -1e30)
                if element_mask is not None:
                    attn_matrix = attn_matrix.masked_fill(element_mask == 0, -1e30)
                attn_matrices.append(attn_matrix)

            # softmax
            for i, h in enumerate(range(self.head_width, num_heads - self.head_width)):
                attn_matrix = attn_matrices[i]
                nb, nrh, len1, len2 = attn_matrix.shape
                attn_score = F.softmax(attn_matrix.transpose(1, 2).contiguous().view(nb, len1, nrh * len2), -1)
                attn_score = attn_score.transpose(1, 2).contiguous().view(nb, nrh, len1, len2)
                attn_matrices[i] = attn_score

            outs = []
            for i, h in enumerate(range(self.head_width, num_heads - self.head_width)):
                out = torch.matmul(attn_matrices[i],
                                   V[:, h-self.head_width:h+self.head_width])  # [nb, nrh, len1, hid2]
                out = torch.sum(out, 1)  # [nb, len1, hid2]
                outs.append(out)
                # print(out.shape)
            out = torch.stack(outs, 0).transpose(0, 1)
            # outs: [nb, nh, len1, hid2]
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.0, element_width=None, head_width=None):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.linear_layers = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.element_width = element_width
        self.head_width = head_width
        self.attn = ScaledDotProductAttention(self.dropout, element_width=element_width, head_width=head_width)

    def forward(self, Q, K, V, mask=None):
        # Q: [nb, len1, d_model], K: [nb, len2, d_model], V: [nb, len2, d_model], mask: [nb, len2]
        num_batches = Q.size(0)
        Q, K, V = [
            l(x).view(num_batches, -1, self.num_heads, self.d_k).transpose(1, 2)
            for l, x in zip(self.linear_layers, (Q, K, V))
        ]
        # Q: [nb, nh, len1, d_k], K: [nb, nh, len2, d_k], V: [nb, nh, len2, d_k]
        x = self.attn(Q, K, V, mask)
        x = x.transpose(1, 2).contiguous().view(num_batches, -1, self.num_heads * self.d_k)
        # [nb, len1, d_model]
        return self.linear_layers[-1](x)


default_params = {
    "num_heads": 4,
    "encoder_layers_num": 6,
    "decoder_layers_num": 6
}


# Source and Target should be embedded into the same dimension
class Transformer(nn.Module):
    def __init__(self, param_name):
        super().__init__()
        params = configOrganizer.fetch_config(param_name)

        assert_param(param=params, field='input_dim', field_type=int)

        self.params = default_params
        self.params.update(params)
        self.encoder = EncoderFromLayer(EncoderLayer(self.params["input_dim"],
                                                     MultiHeadAttention(self.params["num_heads"],
                                                                        self.params["input_dim"]),
                                                     PositionWiseFeedForward(self.params["input_dim"],
                                                                             self.params["hidden_dim"])),
                                        self.params["encoder_layers_num"])
        self.decoder = DecoderFromLayer(DecoderLayer(self.params["input_dim"],
                                                     MultiHeadAttention(self.params["num_heads"],
                                                                        self.params["input_dim"]),
                                                     MultiHeadAttention(self.params["num_heads"],
                                                                        self.params["input_dim"]),
                                                     PositionWiseFeedForward(self.params["input_dim"],
                                                                             self.params["hidden_dim"])),
                                        self.params["decoder_layers_num"])

    def forward(self, source, source_mask, target, target_mask):
        x = self.encoder(source, source_mask)  # [nb, len1, hid]
        x = self.decoder(target, x, source_mask, target_mask)  # [nb, len2, hid]
        return x

    def encode(self, source, source_mask):
        return self.encoder(source, source_mask)

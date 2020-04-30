import copy

from torch import nn
import math
import torch
import torch.nn.functional as F

from layers.attention_layers import SelfAttentionWithMultiHead
from layers.position_layers import PositionWiseFeedForward, PositionEncoder
from utils.asserter import assert_param
from utils.container import configContainer
from utils.helper import clones, make_unidirectional_mask, random_mask
from layers.blocks import LayerNormResidualBlock
from utils.text_processor import Vocabulary


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
        self.self_attn = LayerNormResidualBlock(layer=self_attn,
                                                input_dim=self.input_dim,
                                                drop_ratio=dropout)
        self.feed_forward = LayerNormResidualBlock(layer=feed_forward,
                                                   input_dim=self.input_dim,
                                                   drop_ratio=dropout)

    def forward(self, x, mask):
        x = self.self_attn(x, mask)
        x = self.feed_forward(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, self_attn, source_attn, feed_forward=None, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.self_attn = LayerNormResidualBlock(layer=self_attn,
                                                input_dim=self.input_dim,
                                                drop_ratio=dropout)
        self.feed_forward = LayerNormResidualBlock(layer=feed_forward,
                                                   input_dim=self.input_dim,
                                                   drop_ratio=dropout)
        self.source_attn = LayerNormResidualBlock(layer=source_attn,
                                                  input_dim=self.input_dim,
                                                  drop_ratio=dropout)

    def forward(self, target, target_mask, source=None, source_mask=None):
        target = self.self_attn(target, target_mask)
        if source is not None:
            target = self.source_attn(target, source, source, source_mask)
        target = self.feed_forward(target)
        return target


class TransformerEncoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.position_encoder = PositionEncoder({
            "input_dim": params["input_dim"]
        })
        self.core = EncoderFromLayer(
            EncoderLayer(
                self.params["input_dim"],
                SelfAttentionWithMultiHead(params),
                PositionWiseFeedForward(params)
            ),
            self.params["encoder_layer_num"])
        if params["max_target_len"] is not None:
            self.length_embedding = nn.Embedding(params["max_target_len"], params["input_dim"])

    def forward(self, x, mask, positional_encoding=True):
        """
        :param positional_encoding: True if use_positional_encoding
        :param x: in shape (batch_size, seq_len, input_dim)
        :param mask: in shape (batch_size, seq_len)
        :return: encoding in shape (batch_size, seq_len, input_dim)
        """
        if positional_encoding:
            x = self.position_encoder(x)
        return self.core(x, mask)

    def encoding_with_length(self, x, mask, positional_encoding=True):
        assert self.length_embedding is not None, "Invalid model configuration!"
        if positional_encoding:
            x = self.position_encoder(x)
        len_tokens = torch.zeros(x.size(0), 1)
        len_embeddings = self.length_embedding(len_tokens).view(x.size(0), 1, -1)
        x = torch.cat([len_embeddings, x], dim=1)
        mask = torch.cat([torch.zeros(x.size(0), 1), mask], dim=1)
        encoding = self.core(x, mask)
        len_embed_weight = self.length_embedding.weight.transpose(0, 1)
        predicted_length_logits = torch.matmul(encoding[:, 0, :], len_embed_weight).float()
        predicted_length_logits[:, 0] += float('-inf')
        predicted_lengths = F.log_softmax(predicted_length_logits, dim=-1)
        return encoding[:, 1:, :], predicted_lengths


class TransformerDecoder(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.position_encoder = PositionEncoder({
            "input_dim": params["input_dim"]
        })
        self.core = DecoderFromLayer(
            DecoderLayer(
                params["input_dim"],
                SelfAttentionWithMultiHead(params),
                SelfAttentionWithMultiHead(params),
                PositionWiseFeedForward(params)
            ),
            params["decoder_layer_num"])

    def forward(self, target, target_mask, source=None, source_mask=None, positional_encoding=True):
        if positional_encoding:
            target = self.position_encoder(target)
        return self.core(target, target_mask, source, source_mask)


default_params = {
    "head_num": 4,
    "encoder_layer_num": 6,
    "decoder_layer_num": 6,
    "bidirectional_decoding": False,
    "share_parameter": False,
    "dropout": 0.0
}


# Source and Target should be embedded into the same dimension
class TransformerCore(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configContainer.fetch_config(params)

        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='hidden_dim', field_type=int)

        self.params = default_params
        self.params.update(params)
        if not params["share_parameter"]:
            self.encoder = TransformerEncoder(params)
            self.decoder = TransformerDecoder(params)
        else:
            self.encoder = self.decoder = TransformerDecoder(params)

    def forward(self, source, source_mask, target, target_mask, positional_encoding=True):
        if self.params["share_parameter"]:
            encoding = self.encoder(source, source_mask, None, None, positional_encoding)
        else:
            encoding = self.encoder(source, source_mask, positional_encoding)  # [nb, len1, hid]
        if not self.params["bidirectional_decoding"]:
            target_mask = make_unidirectional_mask(target_mask)
        x = self.decoder(target, target_mask, encoding, source_mask, positional_encoding)  # [nb, len2, hid]
        return x, encoding

    def encode(self, source, source_mask, positional_encoding=True):
        if self.params["share_parameter"]:
            return self.encoder(source, source_mask, None, None, positional_encoding)
        else:
            return self.encoder(source, source_mask, positional_encoding)


class Any2SentenceTransformer(TransformerCore):
    def __init__(self, params):
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configContainer.fetch_config(params)
        super().__init__(params)

        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='target_vocab', field_type=Vocabulary)
        if "share_embedding" not in params:
            params["share_embedding"] = True

        self.out = nn.Linear(params['input_dim'], params['target_vocab'].word_num)
        if self.params["share_embedding"]:
            self.embedding_weight = self.out.weight * math.sqrt(params['input_dim'])
        else:
            self.embedding_layer = nn.Embedding(params['target_vocab'].word_num, params['input_dim'])
            self.embedding_weight = self.embedding_layer.weight

    def forward(self, source, source_mask, target, target_mask, positional_encoding=True):
        target = F.embedding(target, self.embedding_weight)
        result, encoding = super().forward(source, source_mask, target, target_mask, positional_encoding)
        result = self.out(result)
        scores = F.softmax(result, dim=-1)
        return result, scores, encoding


class DualTransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configContainer.fetch_config(params)

        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='hidden_dim', field_type=int)
        assert_param(param=params, field='layer_number_1', field_type=int)
        assert_param(param=params, field='layer_number_2', field_type=int)
        if "dropout" not in params:
            params["dropout"] = 0.0
        if "share_parameter" not in params:
            params["share_parameter"] = True

        self.params = default_params
        self.params.update(params)

        self.transformer1 = TransformerCore({
            "head_num": self.params["head_num"],
            "input_dim": self.params["input_dim"],
            "decoder_layer_num": self.params["layer_number_1"],
            "hidden_dim": self.params["hidden_dim"],
            "share_parameter": params["share_parameter"]
        })

        self.transformer2 = TransformerCore({
            "head_num": self.params["head_num"],
            "input_dim": self.params["input_dim"],
            "decoder_layer_num": self.params["layer_number_2"],
            "hidden_dim": self.params["hidden_dim"],
            "share_parameter": params["share_parameter"]
        })

    def forward(self, source1, source_mask1, source2, source_mask2, decoder_id, encoding=None,
                positional_encoding=True):
        if decoder_id == 1:
            if encoding is None:
                encoding = self.transformer2.encode(source2, source_mask2, positional_encoding)
            result = self.transformer1(source1, source_mask1, encoding, source_mask2)
        elif decoder_id == 2:
            if encoding is None:
                encoding = self.transformer1(source1, source_mask1, positional_encoding)
            result = self.transformer2(source2, source_mask2, encoding, source_mask1)
        else:
            raise NotImplementedError
        return encoding, result


def predict_length_beam(gold_length, predicted_lengths, length_beam_size):
    if gold_length is not None:
        # Search lengths around gold_length
        beam_starts = gold_length - (length_beam_size - 1) // 2
        beam_ends = gold_length + length_beam_size // 2 + 1
        beam = torch.stack(
            [torch.arange(beam_starts[batch], beam_ends[batch], device=beam_starts.device) for batch in
             range(gold_length.size(0))], dim=0)
    else:
        # Search top-k lengths
        beam = predicted_lengths.topk(length_beam_size, dim=1)[1]
    beam[beam < 2] = 2
    return beam


class MaskPredictTransformer(Any2SentenceTransformer):
    def __init__(self, params):
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configContainer.fetch_config(params)
        params["bidirectional_decoding"] = True  # Important!
        super().__init__(params)

        self.params = params
        self.target_vocab = params['target_vocab']

    def forward(self, source, source_mask, target, target_mask, positional_encoding=True):
        target_mask, masked_idx = self.regenerate_mask(target_mask)
        result, scores, encoding = super().forward(source, source_mask, target, target_mask, positional_encoding)
        return result, masked_idx

    def regenerate_mask(self, target_mask):
        target_length = target_mask.sum(-1)
        max_length = target_mask.size(-1)
        mask_num = (torch.rand_like(target_length) * target_length).long()
        mask_idx = random_mask(target_length, max_length, mask_num)
        batch_idx = torch.arange(target_mask.size(0)).view(-1, 1)
        new_mask = copy.deepcopy(target_mask)
        new_mask[batch_idx, mask_idx] = 0
        return new_mask, mask_idx

    def preprocess_target(self, batch_size, length_beam_size, max_len, beam):
        target_tokens = torch.zeros(batch_size, length_beam_size, max_len).fill_(self.target_vocab.MASK)
        target_pad_mask = torch.triu(torch.ones(max_len, max_len).long(), 1)
        target_pad_mask = torch.stack([target_pad_mask[beam[batch] - 1] for batch in range(batch_size)], dim=0)
        # beam is in shape (bs, lbs), length_mask is in shape (bs, lbs, max_len)
        target_tokens = (1 - target_pad_mask) * target_tokens + target_pad_mask * self.target_vocab.PAD
        target_tokens = target_tokens.view(batch_size * length_beam_size, max_len)
        target_pad_mask = target_pad_mask.view(batch_size * length_beam_size, max_len)
        return target_tokens, (1 - target_pad_mask)

    def initial_generation(self, target_token, target_mask, encoding, encoding_mask, positional_encoding):
        target_emb = F.embedding(target_token, self.embedding_weight)
        result = self.decoder(target_emb, target_mask, encoding, encoding_mask, positional_encoding)
        target_probs, target_token = F.softmax(self.out(result), dim=-1).max(dim=-1)
        target_token.masked_fill_(1 - target_mask, self.target_vocab.MASK)
        target_probs.masked_fill_(1 - target_mask, 1.0)
        return target_token, target_probs

    def iterative_generation(self, counter, iteration, target_token, target_probs, target_mask,
                             encoding, encoding_mask, positional_encoding):
        batch_size = encoding.size(0)
        batch_idx = torch.arange(batch_size).view(-1, 1)

        target_lengths = target_mask.sum(1)
        mask_num = (target_lengths.float() * (1.0 - (counter / iteration))).long()
        target_probs.masked_fill_(1 - target_mask, 1.0)
        masks_idx = self.select_worst(target_probs, mask_num)
        target_token[batch_idx, masks_idx] = self.target_vocab.MASK
        target_token.masked_fill_(1 - target_mask, self.target_vocab.PAD)
        target_emb = F.embedding(target_token, self.embedding_weight)
        result = self.decoder(target_emb, target_mask, encoding, encoding_mask, positional_encoding)
        new_target_probs, new_target_token = F.softmax(self.out(result), dim=-1).max(dim=-1)
        target_token[batch_idx, masks_idx] = new_target_token[batch_idx, masks_idx]
        target_probs[batch_idx, masks_idx] = new_target_probs[batch_idx, masks_idx]
        target_token.masked_fill_(1 - target_mask, self.target_vocab.MASK)
        target_probs.masked_fill_(1 - target_mask, 1.0)
        return target_token, target_probs

    def select_worst(self, target_probs, mask_num):
        batch_size, max_len = target_probs.size()
        # Pick out at least ONE worst element
        masks_idx = [target_probs[batch, :].topk(max(1, mask_num[batch]), largest=False, sorted=False)[1]
                     for batch in range(batch_size)]
        masks_idx = [torch.cat([mask_idx,
                                torch.tensor([mask_idx[0]] * (max_len - mask_idx.size(0)))])
                     for mask_idx in masks_idx]
        return torch.stack(masks_idx, dim=0)

    def generate(self, source, source_mask, length_beam_size, gold_target_len, iteration=None,
                 positional_encoding=True):
        batch_size = source.size(0)
        encoding, predicted_len = self.encoder.encoding_with_length(source, source_mask, positional_encoding)
        # encoding is in shape (bs, src_len, emb_dim)
        beam = predict_length_beam(gold_target_len, predicted_len, length_beam_size)
        max_len = beam.max().item()
        target_token, target_mask = self.preprocess_target(batch_size, length_beam_size, max_len, beam)
        iteration = max_len if iteration is None else iteration
        encoding = encoding.unsqueeze(1).repeat(1, length_beam_size, 1, 1).view(batch_size * length_beam_size, -1,
                                                                                encoding.size(-1))
        encoding_mask = source_mask.unsqueeze(1).repeat(1, length_beam_size, 1).view(batch_size * length_beam_size, -1)
        # encoding in shape (bs * lbs, src_len, emb_dim), encoding_mask in shape (bs * lbs, src_len)
        # target_tokens in shape (bs * lbs, max_len)
        target_token, target_probs = self.initial_generation(target_token, target_mask, encoding, encoding_mask,
                                                             positional_encoding)
        for counter in range(1, iteration):
            target_token, target_probs = self.iterative_generation(counter, iteration, target_token,
                                                                   target_probs, target_mask, encoding,
                                                                   encoding_mask, positional_encoding)
        log_probs = target_probs.log().sum(-1).view(batch_size, length_beam_size)
        hypotheses = target_token.view(batch_size, length_beam_size, max_len)
        target_lengths = target_mask.sum(-1).view(batch_size, length_beam_size).float()
        avg_log_probs = log_probs / target_lengths
        best_lengths = avg_log_probs.max(-1)[1]
        hypotheses = torch.stack([hypotheses[batch, length] for batch, length in enumerate(best_lengths)], dim=0)
        return hypotheses

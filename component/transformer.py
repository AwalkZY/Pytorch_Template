import copy

import torchsnooper
from torch import nn
import math
import torch
import torch.nn.functional as F

from layers.attention_layers import SelfAttentionWithMultiHead, MultiHeadAttention
from layers.position_layers import PositionWiseFeedForward, PositionEncoder
from utils.asserter import assert_param
from utils.beam_search import beam_search
from utils.container import configContainer
from utils.helper import clones, make_unidirectional_mask, random_mask, no_peak_mask, sequence_mask
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

    def forward(self, x, target_mask, source, source_mask):
        for layer in self.layers:
            x = layer(x, target_mask, source, source_mask)
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

    def forward(self, x, mask):
        """
        :param x: in shape (batch_size, seq_len, input_dim)
        :param mask: in shape (batch_size, seq_len)
        :return: encoding in shape (batch_size, seq_len, input_dim)
        """
        x += self.position_encoder(x)
        return self.core(x, mask)

    def encoding_with_length(self, x, mask):
        assert self.length_embedding is not None, "Invalid model configuration!"
        x += self.position_encoder(x)
        len_tokens = torch.zeros(x.size(0), 1).long().to(x.device)
        len_embeddings = self.length_embedding(len_tokens).view(x.size(0), 1, -1)
        x = torch.cat([len_embeddings, x], dim=1)
        mask = torch.cat([torch.zeros(x.size(0), 1).to(x.device), mask], dim=-1)
        encoding = self.core(x, mask)
        len_embed_weight = self.length_embedding.weight.transpose(0, 1)
        predicted_length_logits = torch.matmul(encoding[:, 0, :], len_embed_weight).float()
        predicted_length_logits[:, 0] += float('-inf')
        return encoding[:, 1:, :], predicted_length_logits


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
                MultiHeadAttention(params),
                PositionWiseFeedForward(params)
            ),
            params["decoder_layer_num"])

    def forward(self, target, target_mask, source=None, source_mask=None):
        target += self.position_encoder(target)
        return self.core(target, target_mask, source, source_mask)


default_params = {
    "head_num": 4,
    "encoder_layer_num": 6,
    "decoder_layer_num": 6,
    "causal": True,
    "share_parameter": False,
    "share_embedding": True,
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

        if not self.params["share_parameter"]:
            self.encoder = TransformerEncoder(self.params)
            self.decoder = TransformerDecoder(self.params)
        else:
            self.encoder = self.decoder = TransformerDecoder(self.params)

    def forward(self, source, source_mask, target, target_mask):
        encoding = self.encoder(source, source_mask)  # [nb, len1, hid]
        if self.params["causal"]:
            target_mask = make_unidirectional_mask(target_mask)
        x = self.decoder(target, target_mask, encoding, source_mask)  # [nb, len2, hid]
        return x, encoding

    def encode(self, source, source_mask):
        return self.encoder(source, source_mask)


class Any2SentenceTransformer(TransformerCore):
    def __init__(self, params):
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configContainer.fetch_config(params)
        if "share_embedding" not in params:
            params["share_embedding"] = True

        self.params = params

        super().__init__(params)

        assert_param(param=params, field='input_dim', field_type=int)
        assert_param(param=params, field='target_vocab', field_type=Vocabulary)

        self.out = nn.Linear(params['input_dim'], params['target_vocab'].word_num)

    def forward(self, source, source_mask, target, target_mask):
        embedding_weight = self.out.weight * math.sqrt(self.params['input_dim'])
        target = F.embedding(target, embedding_weight)
        result, encoding = super().forward(source, source_mask, target, target_mask)
        result = self.out(result)
        scores = F.softmax(result, dim=-1)
        return result, scores, encoding

    def generate(self, source, source_mask, k, max_len):
        return beam_search(self, source, source_mask, self.params["target_vocab"], k, max_len)


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
        if "share_embedding" not in params:
            params["share_embedding"] = True

        self.params = default_params
        self.params.update(params)

        self.transformer1 = TransformerCore({
            "head_num": self.params["head_num"],
            "input_dim": self.params["input_dim"],
            "decoder_layer_num": self.params["layer_number_1"],
            "hidden_dim": self.params["hidden_dim"],
            "share_parameter": self.params["share_parameter"],
            "causal": self.params["causal"]
        })

        self.transformer2 = TransformerCore({
            "head_num": self.params["head_num"],
            "input_dim": self.params["input_dim"],
            "decoder_layer_num": self.params["layer_number_2"],
            "hidden_dim": self.params["hidden_dim"],
            "share_parameter": self.params["share_parameter"],
            "causal": self.params["causal"]
        })

    def forward(self, source1, source_mask1, source2, source_mask2, decoder_id, encoding=None):
        if decoder_id == 1:
            if encoding is None:
                encoding = self.transformer2.encode(source2, source_mask2)
            result, _ = self.transformer1(source1, source_mask1, encoding, source_mask2)
        elif decoder_id == 2:
            if encoding is None:
                encoding = self.transformer1.encode(source1, source_mask1)
            result, _ = self.transformer2(source2, source_mask2, encoding, source_mask1)
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


def generate_mask(target, target_mask, mask_token):
    target_length = target_mask.sum(-1)
    max_length = target.size(1)
    valid_mask = sequence_mask(target_length, max_length)
    mask_num = (torch.rand(target_length.size()).to(target.device) * target_length).floor().clamp_min(1.0).long()
    prob = torch.rand(target.size()).to(target.device).masked_fill(~valid_mask, 0.0)
    _, idx = torch.sort(prob, descending=True)
    new_mask = torch.zeros(target.size()).to(target.device)
    new_targets = copy.deepcopy(target)
    mask_idx = [idx[batch][:mask_num[batch]] for batch in range(target.size(0))]
    mask_idx = torch.stack([torch.cat([mask,
                                       torch.tensor([mask[0]] * (max_length - mask_num[batch])).to(target.device)])
                            for batch, mask in enumerate(mask_idx)], dim=0)
    batch_idx = torch.arange(target.size(0)).view(-1, 1)
    new_targets[batch_idx, mask_idx] = mask_token
    new_mask[batch_idx, mask_idx] = 1
    return new_targets, mask_idx, new_mask.bool(), mask_num


class MaskPredictTransformer(Any2SentenceTransformer):
    def __init__(self, params):
        assert type(params) in [dict, str], "Invalid Parameter Type!"
        if type(params) is str:
            params = configContainer.fetch_config(params)
        params["causal"] = False  # Important!

        self.params = default_params
        self.params.update(params)

        super().__init__(params)
        self.target_vocab = params['target_vocab']

    def forward(self, source, source_mask, target, target_mask):
        _, lengths_logits = self.encoder.encoding_with_length(source, source_mask)
        masked_target, masked_idx, mask, mask_num = generate_mask(target, target_mask, self.target_vocab.stoi('<MASK>'))
        result, scores, encoding = super().forward(source, source_mask, masked_target, target_mask)
        return result, masked_idx, lengths_logits

    def preprocess_target(self, batch_size, length_beam_size, max_len, beam):
        target_tokens = torch.zeros(batch_size, length_beam_size, max_len).fill_(self.target_vocab.MASK).long()
        target_pad_mask = torch.triu(torch.ones(max_len, max_len).long(), 1)
        target_pad_mask = torch.stack([target_pad_mask[beam[batch] - 1] for batch in range(batch_size)], dim=0).bool()
        # beam is in shape (bs, lbs), length_mask is in shape (bs, lbs, max_len)
        target_tokens = (~target_pad_mask) * target_tokens + target_pad_mask * self.target_vocab.PAD
        target_tokens = target_tokens.view(batch_size * length_beam_size, max_len)
        target_pad_mask = target_pad_mask.view(batch_size * length_beam_size, max_len)
        return target_tokens, (~target_pad_mask)

    def initial_generation(self, target_token, target_mask, encoding, encoding_mask):
        target_emb = F.embedding(target_token, self.embedding_weight)
        result = self.decoder(target_emb, target_mask, encoding, encoding_mask)
        target_probs, target_token = F.softmax(self.out(result), dim=-1).max(dim=-1)
        target_token.masked_fill_(~target_mask, self.target_vocab.PAD)
        target_probs.masked_fill_(~target_mask, 1.0)
        return target_token, target_probs

    def iterative_generation(self, counter, iteration, target_token, target_probs, target_mask,
                             encoding, encoding_mask):
        batch_size = encoding.size(0)
        batch_idx = torch.arange(batch_size).view(-1, 1)

        target_lengths = target_mask.sum(1)
        mask_num = (target_lengths.float() * (1.0 - (counter / iteration))).long()
        target_probs.masked_fill_(~target_mask, 1.0)
        masks_idx = self.select_worst(target_probs, mask_num)
        target_token[batch_idx, masks_idx] = self.target_vocab.MASK
        target_token.masked_fill_(~target_mask, self.target_vocab.PAD)
        target_emb = F.embedding(target_token, self.embedding_weight)
        result = self.decoder(target_emb, target_mask, encoding, encoding_mask)
        new_target_probs, new_target_token = F.softmax(self.out(result), dim=-1).max(dim=-1)
        target_token[batch_idx, masks_idx] = new_target_token[batch_idx, masks_idx]
        target_probs[batch_idx, masks_idx] = new_target_probs[batch_idx, masks_idx]
        target_token.masked_fill_(~target_mask, self.target_vocab.PAD)
        target_probs.masked_fill_(~target_mask, 1.0)
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

    def generate(self, source, source_mask, length_beam_size, gold_target_len, iteration=None):
        alpha = 0.7
        batch_size = source.size(0)
        encoding, predicted_len = self.encoder.encoding_with_length(source, source_mask)
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
        target_token, target_probs = self.initial_generation(target_token, target_mask, encoding, encoding_mask)
        for counter in range(1, iteration):
            target_token, target_probs = self.iterative_generation(counter, iteration, target_token,
                                                                   target_probs, target_mask, encoding,
                                                                   encoding_mask)
        log_probs = target_probs.log().sum(-1).view(batch_size, length_beam_size)
        hypotheses = target_token.view(batch_size, length_beam_size, max_len)
        target_lengths = target_mask.sum(-1).view(batch_size, length_beam_size).float()
        avg_log_probs = log_probs / (target_lengths ** alpha)
        best_score, best_lengths = avg_log_probs.max(-1)
        hypotheses = torch.stack([hypotheses[batch, length] for batch, length in enumerate(best_lengths)], dim=0)
        return hypotheses, best_score, best_lengths

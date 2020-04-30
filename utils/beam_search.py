import math

import torch

from component.transformer import Any2SentenceTransformer
from utils.helper import no_peak_mask
from utils.text_processor import Vocabulary


def init_vars(model, source, source_mask, target_vocab, k, max_len):
    assert source.size(0) == 1, "Invalid Number of sentences!"
    assert type(target_vocab) is Vocabulary, "Invalid type of target vocabulary!"
    assert type(model) is Any2SentenceTransformer, "Invalid type of model!"
    # Target sentence begins with Begin of Sentence
    init_token = target_vocab.stoi('<BOS>')
    target = torch.tensor([[init_token]]).long()
    target_mask = no_peak_mask(1)
    result, scores, encoding = model.forward(source, source_mask, target, target_mask, positional_encoding=True)
    # scores in shape (batch_size == 1, sentence_length, vocab_size)
    # Here we do the slice operation on the "sentence_length" dimension
    k_scores, idx = scores[:, -1].data.topk(k)
    log_scores = torch.tensor([math.log(score) for score in k_scores.data[0]]).unsqueeze(0)
    target = torch.zeros(k, max_len).long()
    target[:, 0] = init_token
    # Info: len(idx[0]) == k
    target[:, 1] = idx[0]
    encodings = torch.zeros(k, encoding.size(-2), encoding.size(-1))
    encodings[:, :] = encoding[0]
    return target, encodings, log_scores


def find_k_best(targets, scores, last_scores, i, k):
    word_k_scores, word_idx = scores[:, -1].data.topk(k)
    # k_scores & idx are in shape (k, k)
    log_scores = torch.log(word_k_scores) + last_scores.transpose(0, 1)
    # use "log" operation to transform multiplication into logarithmic addition
    group_k_scores, group_idx = log_scores.view(-1).topk(k)
    row = group_idx // k
    col = group_idx % k
    # Regenerate sentence sequence
    targets[:, :i] = targets[row, :i]
    targets[:, i] = word_idx[row, col]
    log_scores = group_k_scores.unsqueeze(0)
    return targets, log_scores


def beam_search(model, source, source_mask, target_vocab, k, max_len):
    assert type(model) is Any2SentenceTransformer, "Invalid type of model!"
    targets, encodings, log_scores = init_vars(model, source, source_mask, target_vocab, k, max_len)
    eos_token = target_vocab.stoi("<EOS>")
    idx = None
    for i in range(2, max_len):
        target_mask = no_peak_mask(i)
        result, scores, encoding = model.forward(source, source_mask, targets[:, :i], target_mask, positional_encoding=True)
        targets, log_scores = find_k_best(targets, scores, log_scores, i, k)
        end_pos = (targets == eos_token).nonzero()
        sentence_lengths = torch.zeros(len(targets), dtype=torch.long).to(source.device)
        for pos in end_pos:
            beam_i = pos[0]
            if sentence_lengths[beam_i] == 0:
                sentence_lengths[beam_i] = pos[1]
        finished_num = len([s for s in sentence_lengths if s > 0])
        if finished_num == k:
            alpha = 0.7
            ans = log_scores / (sentence_lengths,type_as(log_scores) ** alpha)
            _, idx = torch.max(ans, 1)
            idx = idx.data[0]
            break
    if idx is None:
        idx = 0
    length = (targets[idx] == eos_token).nonzero()[0]
    return ' '.join([
        target_vocab.itos[token] for token in targets[idx][1:length]
    ])



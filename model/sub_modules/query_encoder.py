from torch import nn

from component.transformer import TransformerEncoder
from layers.dynamic_rnn import DynamicGRU


class QueryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, encoder_type):
        super(QueryEncoder, self).__init__()
        self.reshaping = nn.Linear(input_size, hidden_size)
        self.encoder_type = encoder_type
        if encoder_type == "GRU":
            self.txt_agg = DynamicGRU(hidden_size, hidden_size // 2, num_layers=1,
                                      bidirectional=True, batch_first=True)
        elif encoder_type == "Transformer":
            self.txt_agg = TransformerEncoder({
                "input_dim": hidden_size,
                "encoder_layer_num": 1
            })

    def reset_parameters(self):
        self.txt_gru.reset_parameters()

    def forward(self, textual_input, textual_mask):
        textual_len = textual_mask.sum(-1)
        if self.encoder_type == "GRU":
            txt_h = self.txt_agg(textual_input, textual_len)
        else:
            txt_h = self.txt_agg(textual_input, textual_mask)
        return txt_h


def build_query_encoder(cfg) -> QueryEncoder:
    return QueryEncoder(cfg.INPUT_SIZE, cfg.HIDDEN_SIZE, cfg.ENCODER_TYPE)

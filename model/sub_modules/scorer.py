from torch import nn
import torch
import torch.nn.functional as F


def get_padded_mask_and_weight(*args):
    if len(args) == 2:
        mask, conv = args
        masked_weight = torch.round(F.conv3d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(),
                                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
    else:
        raise NotImplementedError

    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight


class MapConv(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, stride, padding, dilation):
        super(MapConv, self).__init__()
        self.conv_layer = nn.Conv3d(in_channels=input_size, out_channels=hidden_size,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation)
        self.pred_layer = nn.Conv2d(in_channels=hidden_size,
                                    out_channels=1,
                                    kernel_size=1)

    def forward(self, map_h, map_mask, props):
        map_h = torch.relu_(self.conv_layer(map_h))
        # 这里从5层压缩到3层， size: [B, C, Depth, Height, Width]
        scores = []
        for depth in range(map_h.size(2)):
            padded_mask = map_mask[:, :, depth]
            x = map_h[:, :, depth]
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, self.conv_layer)
            x = x * masked_weight
            x = self.pred_layer(x)
            scores.append(x[:, 0, props[:, 0], props[:, 1]])
        return torch.stack(scores, dim=1)
        # [B, Depth, Height, Width]

    def reset_parameters(self):
        self.conv_layer.reset_parameters()
        self.pred_layer.reset_parameters()


def build_scorer(cfg) -> MapConv:
    return MapConv(cfg.INPUT_SIZE, cfg.HIDDEN_SIZE, cfg.KERNEL_SIZE, cfg.STRIDE, cfg.PADDING, cfg.DILATION)

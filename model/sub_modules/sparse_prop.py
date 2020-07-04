from torch import nn
import numpy as np
import torch


# def get_ends(now_len):
#     start = np.reshape(np.repeat(np.arange(0, now_len)[:, np.newaxis], axis=1, repeats=now_len), [-1])
#     end = np.reshape(np.repeat(np.arange(1, now_len + 1)[np.newaxis, :], axis=0, repeats=now_len), [-1])
#     props = np.stack([start, end], -1)
#     idx = props[:, 0] < props[:, 1]
#     return torch.from_numpy(props[idx])


class Prop3D(nn.Module):
    def __init__(self, head_scale, rear_scale, max_clip_num):
        super(Prop3D, self).__init__()
        self.head_scale = head_scale
        self.rear_scale = rear_scale
        self.bases = [2 ** i for i in range(head_scale - 1, rear_scale + 2)]
        self.steps = [max_clip_num // base for base in self.bases]
        self.scale_range = rear_scale - head_scale + 2
        self.layers = nn.ModuleList()
        for step in self.step:
            first_layer = nn.MaxPool1d(1, 1)
            rest_layers = [nn.MaxPool1d(2, 1) for _ in range(1, step)]
            scale_layers = nn.ModuleList([first_layer] + rest_layers)
            self.layers.append(scale_layers)

    def forward(self, x):
        """
        :param x: [nb, ns, dim, nc]
        :return:
        """
        batch_size, scale_num, hidden_size, num_clips = x.size()
        map_hidden = x.new_zeros(batch_size, hidden_size, self.scale_range, num_clips, num_clips + 1)
        map_mask = x.new_zeros(batch_size, hidden_size, self.scale_range, num_clips, num_clips + 1)
        # 每一个scale对应一张单独的特征图和掩码图
        for rel_idx, (base, step) in enumerate(zip(self.bases, self.steps)):
            abs_idx = rel_idx + self.head_scale - 1
            now_x = x[:, abs_idx]
            for step_idx, step_layer in enumerate(self.layers[rel_idx], 1):
                now_x = step_layer[step_idx](now_x)
                start_idx = list(range(0, num_clips - step * base, base))
                end_idx = [s_idx + step * base for s_idx in start_idx]
                map_hidden[:, :, rel_idx, start_idx, end_idx] = now_x
                map_mask[:, :, rel_idx, start_idx, end_idx] = 1
        return map_hidden, map_mask


def build_sparse_prop(cfg) -> Prop3D:
    return Prop3D(cfg.STEP, cfg.STRIDES, cfg.BASES)
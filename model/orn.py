#   ==========================================================================================  #
#    Occlusion Recovery Network
#   ==========================================================================================  #
#    Author: Boyu Chen
#   ==========================================================================================  #
#    Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#   ==========================================================================================  #
#    Ref: MOTR - QIM
#   ==========================================================================================  #

import torch
from torch import nn
from triton.profiler import activate


def bounding_box_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class OcclusionRecoveryNetwork(nn.Module):
    """Occlusion Recovery Network:
       Activate when the occlusion detection, Pathchlisation the instance and do Cross-Attention.
       IN: Instance / Bounding box
       IN: Frame t and Frame t-1
       OUT: Completed bounding box
    """
    def __init__(self, dim_in, instance, window_size, num_heads):
        super().__init__()
        self.instance = instance
        self.dim_in = dim_in
        self.window_size = window_size
        self.num_heads = num_heads

        self.norm1 = nn.LayerNorm(dim_in)

    def forward(self, x):
        print(x)

def build(args, layer_name, dim_in, hidden_dim, dim_out):
    interaction_layers = {
        'QIM': OcclusionRecoveryNetwork,
    }
    assert layer_name in interaction_layers, 'invalid query interaction layer: {}'.format(layer_name)
    return interaction_layers[layer_name](args, dim_in, hidden_dim, dim_out)

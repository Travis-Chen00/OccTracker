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
        :unmatched_instances: Tracks unmatched
        :gt_instances: Ground Truth
        :frame_idx: Current frame index
        :d_model, d_ffn: Dimension of hidden layers
        :num_heads, num_layers: Attention block parameters
    """
    def __init__(self, unmatched_instances, gt_instances, frame_idx, d_model, d_ffn, num_heads=4, num_layers=4):
        super().__init__()
        self.unmatched_instances = unmatched_instances
        self.gt_instances = gt_instances

        self._window_size = self.unmatched_instances[frame_idx]._image_size         # Image Size

    def forward(self):
        print(1)

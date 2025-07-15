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
import torch.nn as nn
import torch.nn.functional as F

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

def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            return (x[0], x[0])
        elif len(x) >= 2:
            return (x[0], x[1])
        else:
            raise ValueError("Input sequence is empty")
    else:
        return (x, x)

def resize_frames(frames, size=(96, 96)):
    # frames: shape (B, C, H, W)
    return F.interpolate(frames, size=size, mode='bilinear', align_corners=False)

def bbox_cut(image, image_size, coord):
    """
    image: numpy or torch.Tensor, [H,W,C] or [C,H,W]
    image_size: (H, W)
    coord: torch.tensor/list, [4] 归一化 [cx, cy, w, h]

    返回:
        cut_bbox: 实例区域 (同image类型)
        cut_size: (height, width)
    """
    # 归一化中心转成左上右下像素
    H, W = image_size
    cx, cy, bw, bh = coord
    x1 = int(max(0, (cx - bw/2) * W))
    y1 = int(max(0, (cy - bh/2) * H))
    x2 = int(min(W, (cx + bw/2) * W))
    y2 = int(min(H, (cy + bh/2) * H))

    # 注意image格式，假如是Tensor且 [C,H,W]
    if isinstance(image, torch.Tensor):
        if image.shape[0] == 3:  # [C,H,W]
            crop = image[:, y1:y2, x1:x2]
            cut_size = (y2 - y1, x2 - x1)
        else:  # [H,W,C]
            crop = image[y1:y2, x1:x2, :]
            cut_size = (y2 - y1, x2 - x1)
    else:  # numpy [H,W,C]
        crop = image[y1:y2, x1:x2, :]
        cut_size = (y2 - y1, x2 - x1)

    return crop, cut_size

class AdaptivePatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        missing_instance: Cutted image from current frame
        patch_size (int): Patch token size. Default: 4.
        patch_num: Fixed number of patches, easy for further attention
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size_ = to_2tuple(patch_size)
        self.img_size = None
        self.patch_size = patch_size_
        self.patches_resolution = None
        self.num_patches = None

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x, coord):
        # x shape: (B, C, H, W)
        B, C, H, W = x.shape

        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        object_cut, size_cut = bbox_cut(image=x, image_size=(H, W), coord=coord)
        self.img_size = resize_frames(size_cut)
        self.patches_resolution = [self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1]]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]

        out = self.proj(object_cut).flatten(2).transpose(1, 2)  # (B, Ph*Pw, C)
        if self.norm is not None:
            out = self.norm(out)
        return out

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

class OcclusionRecoveryNetwork(nn.Module):
    """Occlusion Recovery Network:
    Activate when the occlusion detection, Pathchlisation the instance and do Cross-Attention.
        :unmatched_instances: Tracks unmatched at Time t
        :prev_instances: Tracks at Time t-1
        :frame_idx: Current frame index
        :d_model, d_ffn: Dimension of hidden layers
        :num_heads, num_layers: Attention block parameters
    """
    def __init__(self, unmatched_instances, prev_instances,
                 frame_idx,
                 patch_size, in_channel, embed_dim,
                 d_model, d_ffn, num_heads=4, num_layers=4,
                 norm_layer=None):
        super().__init__()
        self.unmatched_instances = unmatched_instances
        self.prev_instances = prev_instances
        self.frame_idx = frame_idx

        self._window_size = self.unmatched_instances[frame_idx]._image_size         # Image Size
        # Two layers simple FFN
        # split image into non-overlapping patches
        self.patch_embed = AdaptivePatchEmbed(patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

    def forward(self, cur_frame, prev_frame):
        # 两个dict 对应unmatched instances 在不同帧
        cur_instance_idxes = self.unmatched_instances[self.frame_idx].obj_ids.detach().cpu().numpy().tolist()
        prev_instance_idxes = self.prev_instances[self.frame_idx].obj_ids.detach().cpu().numpy().tolist()

        assert len(cur_instance_idxes) == len(prev_instance_idxes), "Instances number are not matched!!"
        print(1)

        # Return：结合过的instances, 主要是坐标的更新
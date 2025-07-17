#   ==========================================================================================  #
#    Occlusion Recovery Network
#   ==========================================================================================  #
#    Author: Boyu Chen
#   ==========================================================================================  #
#    Ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
#    Copyright (c) 2021 Microsoft
#   ==========================================================================================  #
#    Ref: MOTR - QIM
#   ==========================================================================================  #
import numpy as np
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

def resize_frames(frames, size=(225, 225)):
    # frames: shape (B, C, H, W)
    return F.interpolate(frames, size=size, mode='bilinear', align_corners=False)

def bbox_cut(image, image_size, coord):
    """
    image: numpy or torch.Tensor, [H,W,C] or [C,H,W]
    image_size: (H, W)
    coord: torch.tensor/list, [4] 归一化 [cx, cy, w, h]

    返回:
        cut_bbox: Instance image
        cut_size: (height, width)
    """
    H, W = image_size
    cx, cy, bw, bh = coord
    x1 = int(max(0, (cx - bw/2) * W))
    y1 = int(max(0, (cy - bh/2) * H))
    x2 = int(min(W, (cx + bw/2) * W))
    y2 = int(min(H, (cy + bh/2) * H))

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
        patch_size: Patch token size. Default: 4.
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


class PatchReconstruction(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim

        # 更灵活的归一化和降维
        self.norm = norm_layer(4 * dim) if norm_layer is not None else nn.Identity()
        self.reduction = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        """
        x: B, L, C
        input_resolution: (H, W)
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        # 确保输入尺寸匹配
        assert L == H * W, f"Input feature has wrong size. Expected {H * W}, got {L}"
        assert H % 2 == 0 and W % 2 == 0, f"Input resolution ({H}*{W}) must be even"

        # 重塑为2D
        x = x.view(B, H, W, C)

        # 四分块重构
        x0 = x[:, 0::2, 0::2, :]  # 左上
        x1 = x[:, 1::2, 0::2, :]  # 右上
        x2 = x[:, 0::2, 1::2, :]  # 左下
        x3 = x[:, 1::2, 1::2, :]  # 右下

        # 拼接四个块
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B (H/2*W/2) 4*C

        # 归一化和降维
        x = self.norm(x)
        x = self.reduction(x)  # B (H/2*W/2) C

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = 0
        # Normalization
        flops += H * W * self.dim * 4
        # Reduction
        flops += (H // 2) * (W // 2) * 4 * self.dim * self.dim
        return flops


class PatchCrossAttention(nn.Module):
    def __init__(self, dim=256, num_heads=8, qkv_bias=True, qk_scale=None,
                 attn_drop=0.0, proj_drop=0.1, use_relative_pos=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 为交叉注意力设计：Q来自当前帧，K/V来自前一帧
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)  # Query projection for current frame
        self.kv_proj = nn.Linear(dim, dim * 2, bias=qkv_bias)  # Key/Value projection for previous frame

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 可选：为patch之间的相对位置添加位置编码
        # 这里假设我们知道patch的空间排列信息
        self.use_relative_pos = use_relative_pos
        if self.use_relative_pos:
            # 假设最大相对距离为31x31的网格
            max_relative_dist = 31
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * max_relative_dist - 1) * (2 * max_relative_dist - 1), num_heads)
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        self.softmax = nn.Softmax(dim=-1)

    def get_relative_position_bias(self, cur_pos, prev_pos):
        """
        计算当前帧和前一帧patch之间的相对位置偏置
        Args:
            cur_pos: (N_cur, 2) 当前帧patch的位置坐标
            prev_pos: (N_prev, 2) 前一帧patch的位置坐标
        """
        if not self.use_relative_pos:
            return None

        # 计算相对位置
        relative_coords = cur_pos[:, None, :] - prev_pos[None, :, :]  # (N_cur, N_prev, 2)
        relative_coords[:, :, 0] += 30  # shift to start from 0, assuming max dist is 30
        relative_coords[:, :, 1] += 30
        relative_coords[:, :, 0] *= 61  # 2*30+1
        relative_position_index = relative_coords.sum(-1)  # (N_cur, N_prev)

        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(
            cur_pos.shape[0], prev_pos.shape[0], -1)  # (N_cur, N_prev, num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (num_heads, N_cur, N_prev)

        return relative_position_bias

    def forward(self, cur_patches, prev_patches, cur_pos=None, prev_pos=None, mask=None):
        """
        Args:
            cur_patches: (B, N_cur, C) 当前帧的patches
            prev_patches: (B, N_prev, C) 前一帧的patches
            cur_pos: (N_cur, 2) 当前帧patch位置坐标 (可选)
            prev_pos: (N_prev, 2) 前一帧patch位置坐标 (可选)
            mask: (B, N_cur, N_prev) 注意力掩码 (可选)
        Returns:
            reconstructed_patches: (B, N_cur, C) 重建的当前帧patches
        """
        B, N_cur, C = cur_patches.shape
        _, N_prev, _ = prev_patches.shape

        # 生成 Q, K, V
        # Q 来自当前帧，K, V 来自前一帧
        q = self.q_proj(cur_patches).reshape(B, N_cur, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_proj(prev_patches).reshape(B, N_prev, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1,
                                                                                                           4)
        k, v = kv[0], kv[1]  # (B, num_heads, N_prev, head_dim)

        # 缩放查询
        q = q * self.scale

        # 计算注意力分数
        attn = (q @ k.transpose(-2, -1))  # (B, num_heads, N_cur, N_prev)

        # 添加相对位置偏置
        if self.use_relative_pos and cur_pos is not None and prev_pos is not None:
            relative_position_bias = self.get_relative_position_bias(cur_pos, prev_pos)
            if relative_position_bias is not None:
                attn = attn + relative_position_bias.unsqueeze(0)  # (B, num_heads, N_cur, N_prev)

        # 应用掩码
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # (B, num_heads, N_cur, N_prev)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Softmax 归一化
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # 计算加权值
        x = (attn @ v).transpose(1, 2).reshape(B, N_cur, C)  # (B, N_cur, C)

        # 最终投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, num_heads={self.num_heads}'

    def flops(self, N_cur, N_prev):
        """计算FLOPs"""
        flops = 0
        # q projection
        flops += N_cur * self.dim * self.dim
        # kv projection
        flops += N_prev * self.dim * 2 * self.dim
        # attention computation
        flops += self.num_heads * N_cur * (self.dim // self.num_heads) * N_prev
        # weighted sum
        flops += self.num_heads * N_cur * N_prev * (self.dim // self.num_heads)
        # output projection
        flops += N_cur * self.dim * self.dim
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
    def __init__(self,
                 d_model, num_heads,
                 dropout, patch_size, in_channel, embed_dim,
                 norm_layer=None):
        super().__init__()
        # Two layers simple FFN
        # split image into non-overlapping patches
        self.patch_embed = AdaptivePatchEmbed(patch_size=patch_size, in_chans=in_channel, embed_dim=embed_dim,
            norm_layer=norm_layer if norm_layer else None)

        self.patchCrossAttn = PatchCrossAttention(dim=d_model, num_heads=num_heads, qkv_bias=True, qk_scale=None,
                                                  attn_drop=dropout, proj_drop=dropout, use_relative_pos=False)


    def forward(self, unmatched_instances, prev_instances, cur_frame, prev_frame, frame_idx):
        # 两个dict 对应unmatched instances 在不同帧
        cur_instance_idxes = unmatched_instances[frame_idx].obj_ids.detach().cpu().numpy().tolist()
        prev_instance_idxes = prev_instances[frame_idx].obj_ids.detach().cpu().numpy().tolist()

        # Ensure the cur is the subset of prev
        mask = np.isin(cur_instance_idxes, prev_instance_idxes)
        matched_cur_idxes = cur_instance_idxes[mask]
        filtered_prev_idxes = np.array([idx for idx in matched_cur_idxes if idx in prev_instance_idxes])
        assert len(cur_instance_idxes) == len(filtered_prev_idxes), "Instances number are not matched!!"

        for idx in cur_instance_idxes:
            cur_obj = self.patch_embed(cur_frame,
                                             self.unmatched_instances[frame_idx][idx].pred_box.tolist())
            prev_obj = self.patch_embed(prev_frame,
                                              self.prev_instances[frame_idx][idx].pred_box.tolist())

            _window_size = unmatched_instances[frame_idx][idx]._image_size
            cur_reconstruction = self.patchCrossAttn(cur_obj, prev_obj)
            patch_reconstructor = PatchReconstruction(input_resolution=(_window_size), dim=self.d_model)
            reconstructed_patches = patch_reconstructor(cur_reconstruction)

            refined_bbox = self.refine_bbox_from_patches(reconstructed_patches, cur_obj)
            self.unmatched_instances[frame_idx][idx].pred_box = refined_bbox

        # 不用返回物体，直接更新了instance
        return None


    def refine_bbox_from_patches(self, reconstructed_patches, original_bbox):
        """
        从重建的patches中提取bbox信息

        Args:
            reconstructed_patches: 重建后的patches特征 (B, L, C)
            original_bbox: 原始bbox [x1, y1, x2, y2]

        Returns:
            refined_bbox: 优化后的bbox [x1, y1, x2, y2]
        """
        # 方法1: 使用注意力权重
        spatial_weights = self.compute_spatial_weights(reconstructed_patches)

        # 方法2: 使用重心偏移
        bbox_center = [(original_bbox[0] + original_bbox[2]) / 2,
                       (original_bbox[1] + original_bbox[3]) / 2]

        # 基于重建patches的特征计算中心点偏移
        center_offset = self.estimate_center_offset(reconstructed_patches)

        # 计算新的bbox中心
        new_center = [
            bbox_center[0] + center_offset[0],
            bbox_center[1] + center_offset[1]
        ]

        # 保持原始bbox的尺寸
        width = original_bbox[2] - original_bbox[0]
        height = original_bbox[3] - original_bbox[1]

        # 构建新的bbox
        refined_bbox = [
            new_center[0] - width / 2,
            new_center[1] - height / 2,
            new_center[0] + width / 2,
            new_center[1] + height / 2
        ]

        return refined_bbox

    def compute_spatial_weights(self, patches):
        """
        计算patches的空间权重

        Args:
            patches: 重建的patches特征 (B, L, C)

        Returns:
            spatial_weights: 空间权重 (B, L)
        """
        # 例如可以使用注意力机制或特征的范数
        spatial_weights = torch.norm(patches, dim=-1)
        spatial_weights = F.softmax(spatial_weights, dim=-1)
        return spatial_weights

    def estimate_center_offset(self, patches):
        """
        估计bbox中心的偏移

        Args:
            patches: 重建的patches特征 (B, L, C)

        Returns:
            center_offset: [dx, dy]
        """
        # 可以使用神经网络或启发式方法估计
        # 这里是一个简单的示例
        mean_patch = torch.mean(patches, dim=0)
        offset_net = nn.Sequential(
            nn.Linear(self.d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        center_offset = offset_net(mean_patch)
        return center_offset.detach().cpu().numpy()
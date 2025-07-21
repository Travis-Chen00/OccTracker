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
from torch.nn.init import trunc_normal_


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

def window_partition(feature, window_size):
    """
    Args:
        feature: (B, H, W, C) -- Feature maps (Crop)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = feature.shape
    feature = feature.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = feature.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_recover(windows, window_size, H, W):
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

def bbox_feature_cut(features, image_size, coord, target_size=224):
    """
    从多层特征图中根据归一化坐标切割实例，并进行填充

    Args:
        features (list/dict): 多层特征图
        image_size (tuple): 原始图像尺寸 (H, W)
        coord (torch.Tensor/list): 归一化坐标 [cx, cy, w, h]
        target_size (int): 目标填充大小

    Returns:
        feature_instances (list): 填充后的特征图列表
        cutted_size (tuple): 裁剪尺寸
    """
    H, W = image_size
    cx, cy, bw, bh = coord

    # 计算边界框坐标
    x1 = int(max(0, (cx - bw/2) * W))
    y1 = int(max(0, (cy - bh/2) * H))
    x2 = int(min(W, (cx + bw/2) * W))
    y2 = int(min(H, (cy + bh/2) * H))

    # 特征图裁剪
    feature_crops = []
    for feature_map in features:
        # 计算特征图上的对应坐标
        # 注意特征图可能是下采样的
        feat_H, feat_W = feature_map.shape[-2:]
        stride_h = H / feat_H
        stride_w = W / feat_W

        # 特征图上的坐标
        f_x1 = max(0, int(x1 / stride_w))
        f_y1 = max(0, int(y1 / stride_h))
        f_x2 = min(feat_W, int(np.ceil(x2 / stride_w)))
        f_y2 = min(feat_H, int(np.ceil(y2 / stride_h)))

        # 裁剪特征图
        if feature_map.dim() == 4:  # [B, C, H, W]
            feature_crop = feature_map[0, :, f_y1:f_y2, f_x1:f_x2]
        else:  # [C, H, W]
            feature_crop = feature_map[:, f_y1:f_y2, f_x1:f_x2]

        # 特征图填充到目标大小
        feature_crop = F.interpolate(
            feature_crop.unsqueeze(0),
            size=(target_size, target_size),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)

        feature_crops.append(feature_crop)

    return feature_crops, (target_size, target_size)


class DeformableCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_points=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.n_points = n_points

        self.reference_points = nn.Linear(dim, 2 * num_heads * n_points)
        self.offset_attn = nn.Linear(dim, num_heads * n_points * 2)

        self.value_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)

        # 初始化权重
        nn.init.xavier_uniform_(self.reference_points.weight)
        nn.init.xavier_uniform_(self.offset_attn.weight)
        nn.init.constant_(self.reference_points.bias, 0.)
        nn.init.constant_(self.offset_attn.bias, 0.)

    def forward(self, query, key, value):
        """
        Args:
            query: 当前帧tokens [N, C]
            key: 前一帧tokens [M, C]
            value: 前一帧tokens [M, C]
        """
        N, C = query.shape
        M, _ = key.shape

        ref_points = self.reference_points(query).view(N, self.num_heads, self.n_points, 2)
        ref_points = torch.sigmoid(ref_points)  # 归一化到 [0, 1]

        offset = self.offset_attn(query).view(N, self.num_heads, self.n_points, 2)

        value_proj = self.value_proj(value)
        value_proj = value_proj.view(M, self.num_heads, C // self.num_heads)

        # 关键步骤：空间变形采样
        sampled_values = self._deformable_sampling(
            value_proj,
            ref_points,
            offset
        )

        attn_weights = self._compute_attention_weights(query, key)
        output = torch.einsum('nhmp,nhc->nmc', attn_weights * sampled_values, value_proj)
        output = output.reshape(N, C)

        return self.output_proj(output)

    def _deformable_sampling(self, value, ref_points, offset):
        """
        可变形采样

        Args:
            value: [M, num_heads, C//num_heads]
            ref_points: [N, num_heads, n_points, 2]
            offset: [N, num_heads, n_points, 2]

        Returns:
            sampled_values: [N, num_heads, n_points, C//num_heads]
        """
        sampling_locations = ref_points + offset
        sampled_values = self._bilinear_interpolate(value, sampling_locations)

        return sampled_values

    def _bilinear_interpolate(self, value, locations):
        """
        双线性插值

        Args:
            value: [M, num_heads, C//num_heads]
            locations: [N, num_heads, n_points, 2]

        Returns:
            interpolated_values: [N, num_heads, n_points, C//num_heads]
        """
        # 实现双线性插值逻辑
        # 这里是一个简化版本，实际实现会更复杂
        N, num_heads, n_points, _ = locations.shape
        M, _, C_per_head = value.shape

        # 将坐标映射到value的网格
        x = locations[..., 0] * (M - 1)
        y = locations[..., 1] * (M - 1)

        # 双线性插值的四个角点
        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        # 约束在有效范围
        x0 = torch.clamp(x0, 0, M - 1)
        x1 = torch.clamp(x1, 0, M - 1)
        y0 = torch.clamp(y0, 0, M - 1)
        y1 = torch.clamp(y1, 0, M - 1)

        # 计算插值权重
        wx = x - x0.float()
        wy = y - y0.float()

        # 双线性插值
        # 这里是一个简化实现，实际应该更复杂
        interpolated = (
                (1 - wx) * (1 - wy) * value[x0, :, :] +
                wx * (1 - wy) * value[x1, :, :] +
                (1 - wx) * wy * value[x0, :, :] +
                wx * wy * value[x1, :, :]
        )

        return interpolated

class WindowCuttingPosition(nn.Module):
    def __init__(self, window_size, dim, heads, dropout):
        super().__init__()

        self.proj_qkv = nn.Linear(dim, 3 * dim)
        self.heads = heads
        head_dim = heads // 2

        self.scale = head_dim ** -0.5
        self.proj_out = nn.Linear(dim, dim)

        self.window_size = window_size
        self.dropout = nn.Dropout(dropout)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.deformable_cross_attn = DeformableCrossAttention(
            dim=dim,
            num_heads=heads,
            n_points=4
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, image_size, features, coord):
        if len(features) == 1:
            prev_feature = cur_feature = features[0]
            prev_size = cur_size = image_size[0]
            prev_coord = cur_coord = coord[0]
        else:
            prev_feature, cur_feature = features
            prev_size, cur_size = image_size
            prev_coord, cur_coord = coord

        prev_feature_crop, prev_cutted_size = bbox_feature_cut(prev_feature, prev_size, prev_coord)
        cur_feature_crop, cur_cutted_size = bbox_feature_cut(cur_feature, cur_size, cur_coord)

        # 获取两帧的特帧，并平接好token
        prev_feature_windows = window_partition(prev_feature_crop,
                                           self.window_size)  # (num_windows*B, window_size, window_size, C)
        num_windows, window_h, window_w, C = prev_feature_windows.shape
        prev_tokens = prev_feature_windows.view(num_windows, window_h * window_w, C)
        prev_tokens = prev_tokens.view(-1, C)

        cur_feature_windows = window_partition(cur_feature_crop, self.window_size)
        num_windows, window_h, window_w, C = cur_feature_windows.shape
        cur_tokens = cur_feature_windows.view(num_windows, window_h * window_w, C)
        cur_tokens = cur_tokens.view(-1, C)

        # Deformable Cross Attention
        temporal_tokens = self.deformable_cross_attn(
            query=cur_tokens,   # 当前帧tokens
            key=prev_tokens,    # 前一帧tokens
            value=prev_tokens   # 前一帧tokens
        )

        completed_tokens = self.layer_norm(temporal_tokens + cur_tokens)
        return completed_tokens


class OcclusionRecoveryNetwork(nn.Module):
    """Occlusion Recovery Network:
    Activate when the occlusion detection, Pathchlisation the instance and do Cross-Attention.
        :unmatched_instances: Tracks unmatched at Time t
        :prev_instances: Tracks at Time t-1
        :frame_idx: Current frame index
        :d_model, d_ffn: Dimension of hidden layers
        :num_heads, num_layers: Attention block parameters
    """
    def __init__(self, window_size, dim, heads, dropout, cls_head, bbox_head):
        super().__init__()
        self.windowAttn = WindowCuttingPosition(window_size, dim, heads, dropout)
        self.cls_head = cls_head
        self.bbox_head = bbox_head


    def forward(self, unmatched_instances, prev_instances, frame_idx, features):
        # 两个dict 对应unmatched instances 在不同帧
        cur_instance_idxes = unmatched_instances[frame_idx].obj_ids.detach().cpu().numpy().tolist()
        prev_instance_idxes = prev_instances[frame_idx].obj_ids.detach().cpu().numpy().tolist()

        # Ensure the cur is the subset of prev
        mask = np.isin(cur_instance_idxes, prev_instance_idxes)
        matched_cur_idxes = cur_instance_idxes[mask]
        filtered_prev_idxes = np.array([idx for idx in matched_cur_idxes if idx in prev_instance_idxes])
        assert len(cur_instance_idxes) == len(filtered_prev_idxes), "Instances number are not matched!!"

        for idx in cur_instance_idxes:
            if frame_idx == 0:
                image_size = [self.unmatched_instances[frame_idx][idx]._image_size,
                              self.unmatched_instances[frame_idx][idx]._image_size]
                coords = [self.unmatched_instances[frame_idx][idx].pred_box.tolist(),
                          self.unmatched_instances[frame_idx][idx].pred_box.tolist()]

            else:
                image_size = [self.prev_instances[frame_idx][idx]._image_size,
                              self.unmatched_instances[frame_idx][idx]._image_size]
                coords = [self.prev_instances[frame_idx][idx].pred_box.tolist(),
                          self.unmatched_instances[frame_idx][idx].pred_box.tolist()]

            completed_token = self.windowAttn(image_size, features, coords)
            outputs_class = self.class_embed(completed_token)   # Classification
            tmp = self.bbox_embed(completed_token)              # Bounding box
            outputs_coord = tmp.sigmoid()

            # 更新结果
            self.unmatched_instances[frame_idx][idx].pred_logits = outputs_class[-1]
            self.unmatched_instances[frame_idx][idx].pred_boxes = outputs_coord[-1]

        # 不用返回物体，直接更新了instance
        # 返回loss 做新的loss计算，更新了instance的 bbox 坐标 在CA模块中
        return None
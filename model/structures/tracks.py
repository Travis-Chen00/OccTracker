from typing import Any
import torch
import torch.nn as nn
from .instances import Instances

class Tracks(Instances):
    """
    Inherit from class Instances.
    Rewrite to add batch dimension into the instances.
    """
    def __init__(self, image_size, hidden_dim=256, num_queries=10, **kwargs):
        """
            Image_size: Height, Width
            hidden_dim: Hidden dimension
            num_classes: Number of classes
        """
        super().__init__(image_size, **kwargs)
        self._image_size = image_size
        self._hidden_dim = hidden_dim
        self._num_queries = num_queries

    def to(self, *args: Any, **kwargs: Any):
        res = Tracks(self.image_size, self._hidden_dim, self._num_queries)
        for k, v in self._fields.items():  # 拷贝已有字段
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            res.set(k, v)
        return res

    def __getitem__(self, item):
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = self.__class__(self._image_size)
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    @staticmethod
    def _init_tracks(image_size, hidden_dim, num_queries, mem_bank_len, device):
        """
        初始化 Tracks 对象。
        第一个 set 的字段固定 Instances 长度，避免 length mismatch 的断言错误。
        """
        tracks = Tracks(image_size=image_size, hidden_dim=hidden_dim, num_queries=num_queries)

        # 用第一个字段固定长度（num_queries）
        tracks.set('obj_idxes', torch.full((num_queries,), -1, dtype=torch.long, device=device))

        # 添加其他字段，必须和 num_queries 对齐
        tracks.set('ref_pts', torch.zeros(num_queries, 4, device=device))
        tracks.set('query_pos', torch.zeros(num_queries, 2 * hidden_dim, device=device))
        tracks.set('output_embedding', torch.zeros((num_queries, hidden_dim >> 1), device=device))
        tracks.set('matched_gt_idxes', torch.full((num_queries,), -1, dtype=torch.long, device=device))
        tracks.set('missing_gt_idxes', torch.full((num_queries,), -1, dtype=torch.long, device=device))
        tracks.set('disappear_time', torch.zeros((num_queries,), dtype=torch.long, device=device))
        tracks.set('iou', torch.zeros((num_queries,), dtype=torch.float, device=device))
        tracks.set('scores', torch.zeros((num_queries,), dtype=torch.float, device=device))
        tracks.set('track_scores', torch.zeros((num_queries,), dtype=torch.float, device=device))
        tracks.set('pred_boxes', torch.zeros((num_queries, 4), dtype=torch.float, device=device))
        tracks.set('pred_logits',
                   torch.zeros((num_queries, num_queries), dtype=torch.float, device=device))  # 注意这里第二维可能是 num_classes

        # 记忆模块
        tracks.set('mem_bank', torch.zeros((num_queries, mem_bank_len, hidden_dim // 2),
                                           dtype=torch.float32, device=device))
        tracks.set('mem_padding_mask', torch.ones((num_queries, mem_bank_len),
                                                  dtype=torch.bool, device=device))
        tracks.set('save_period', torch.zeros((num_queries,), dtype=torch.float32, device=device))

        return tracks




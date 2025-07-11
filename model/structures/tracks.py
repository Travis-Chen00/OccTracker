from typing import Any
import torch
import torch.nn as nn
from .instances import Instances

class Tracks(Instances):
    """
    Inherit from class Instances.
    Rewrite to add batch dimension into the instances.
    """
    def __init__(self, image_size, hidden_dim=256, num_classes=10, **kwargs):
        """
            Image_size: Height, Width
            hidden_dim: Hidden dimension
            num_classes: Number of classes
        """
        super(Instances).__init__(image_size, **kwargs)
        self._height = image_size[0]
        self._width = image_size[1]
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.query_embed = nn.Embedding(num_classes, 2 * hidden_dim)
        self.query_pos = self.query_embed.weight
        self.ref_pts = torch.zeros((0, 4))

        self.obj_idx = torch.zeros((0,), dtype=torch.long)
        self.scores = torch.zeros((0,), dtype=torch.float)

        self.boxes = torch.zeros((0, 4))
        self.logits = torch.zeros((0, self.num_classes), dtype=torch.float)
        self.iou = torch.zeros((0, ), dtype=torch.float)
        self.matched_idx = torch.zeros((0,), dtype=torch.long)
        self.missing_period = torch.zeros((0,), dtype=torch.long)   # Missing period to delete tracks
        self.last_output = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.long_memory = torch.zeros((0, self.hidden_dim), dtype=torch.float)
        self.last_appear_boxes = torch.zeros((0, 4))


    def to(self, *args: Any, **kwargs: Any):
        res = Track(self.image_size, self.hidden_dim, self.num_classes)
        for k, v in kwargs.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)

            res.set(k, v)
        return res

    @staticmethod
    def _init_tracks(batch, hidden_dim, num_classes, device):
        shapes = [img[0].shape[-2:] for img in batch["imgs"]]  # (H, W)
        # h_max, w_max = map(max, zip(*shapes))

        tracks = [
            Track(
                image_size=(h, w),                              # 潜在归一化问题！
                hidden_dim=hidden_dim,
                num_classes=num_classes,
            ).to(device)
            for h, w in shapes
        ]

        return tracks


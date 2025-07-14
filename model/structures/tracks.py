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
        res = Tracks(self.image_size, self.hidden_dim, self._num_queries)
        for k, v in kwargs.items():
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

        ret = self.__class__(self._image_size)          # Class Tracks
        for k, v in self._fields.items():
            ret.set(k, v[item])
        return ret

    @staticmethod
    def _init_tracks(image_size, hidden_dim, num_queries, mem_bank_len, device):
        tracks = Tracks(image_size=image_size, hidden_dim=hidden_dim, num_queries=num_queries)

        tracks.ref_pts = torch.zeros(0, 4)
        tracks.query_pos = torch.zeros(num_queries, 2 * hidden_dim)
        tracks.output_embedding = torch.zeros((num_queries, hidden_dim >> 1), device=device)
        tracks.obj_idxes = torch.full((len(tracks),), -1,
                                               dtype=torch.long, device=device) # -1 means unmatched objects
        tracks.matched_gt_idxes = torch.full((len(tracks),), -1,
                                                      dtype=torch.long, device=device)
        tracks.missing_gt_idxes = torch.full((len(tracks),), -1,
                                                      dtype=torch.long, device=device)
        tracks.disappear_time = torch.zeros((len(tracks), ),
                                                     dtype=torch.long, device=device)
        tracks.iou = torch.zeros((len(tracks),),
                                          dtype=torch.float, device=device)
        tracks.scores = torch.zeros((len(tracks),),
                                             dtype=torch.float, device=device)
        tracks.track_scores = torch.zeros((len(tracks),),
                                                   dtype=torch.float, device=device)
        tracks.pred_boxes = torch.zeros((len(tracks), 4),
                                                 dtype=torch.float, device=device)
        tracks.pred_logits = torch.zeros((len(tracks), num_queries),
                                                  dtype=torch.float, device=device)

        tracks.mem_bank = torch.zeros((len(tracks), mem_bank_len, hidden_dim // 2),
                                               dtype=torch.float32, device=device)
        tracks.mem_padding_mask = torch.ones((len(tracks), mem_bank_len),
                                                      dtype=torch.bool, device=device)
        tracks.save_period = torch.zeros((len(tracks), ),
                                                  dtype=torch.float32, device=device)


        return tracks


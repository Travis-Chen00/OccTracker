# Modified from MeMOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) Ruopeng Gao. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
import torch
import copy

import torch.nn as nn
import torch.nn.functional as F
import torch.distributed

from .matcher import build_matcher
from .structures.instances import Instances
from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
from utils.misc import is_dist_avail_and_initialized, get_world_size, accuracy
from .util import sigmoid_focal_loss
from utils import box_ops
from .structures import Boxes, matched_boxlist_iou
from .structures.tracks import Tracks
from .orn import OcclusionRecoveryNetwork
from utils.utils import is_distributed, distributed_world_size

class FrameMatcher:
    def __init__(self, corrector, num_classes, matcher, weights, losses, focal_loss=True):
        """
                Parameters:
            corrector: Occlusion Recovery Network
            num_classes: number of object categories, omitting the special no-object category
            matcher: HungarianMatcher from DETR
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: include "box_l1_loss", "box_giou_loss", "label_focal_loss"
            focal_loss: using Focal Loss
        """
        self.num_classes = num_classes
        self.matcher = matcher
        self.weights = weights

        self.loss_name = losses
        self.losses = {}
        self.focal_loss = focal_loss

        self.gt_instances = None
        self.num_samples = 0
        self.device = None
        self._current_frame_idx = 0
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.corrector = corrector
        self.prev_instance = None
        self.n_gts = []

    def initialize_for_single_clip(self, gt_instances):
        self.losses = {}
        self.device = None
        self.num_samples = 0
        self.gt_instances = gt_instances
        self.n_gts = []

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def get_sum_loss_dict(self, loss_dict: dict):
        def get_weight(loss_name):
            if "box_l1_loss" in loss_name:
                return self.losses["box_l1_loss"]
            elif "box_giou_loss" in loss_name:
                return self.losses["box_giou_loss"]
            elif "label_focal_loss" in loss_name:
                return self.losses["label_focal_loss"]
            else:
                return 1.0

        # 使用生成器表达式和 sum()
        weighted_losses = [
            torch.tensor(get_weight(k), dtype=torch.float32) *
            (v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
            for k, v in loss_dict.items()
        ]

        if weighted_losses:
            # 确保所有张量在同一设备上
            device = weighted_losses[0].device if hasattr(weighted_losses[0], 'device') else None
            if device:
                weighted_losses = [loss.to(device) for loss in weighted_losses]
            return sum(weighted_losses)
        else:
            return torch.tensor(0.0, requires_grad=True)

    def get_mean_by_n_gts(self):
        total_n_gts = sum(self.n_gts)
        total_n_gts = torch.as_tensor(total_n_gts, dtype=torch.float, device=self.device)
        n_gts = torch.as_tensor(self.n_gts, dtype=torch.float, device=self.device)
        if is_distributed():
            torch.distributed.all_reduce(total_n_gts)
            torch.distributed.all_reduce(n_gts)
        total_n_gts = torch.clamp(total_n_gts / distributed_world_size(), min=1).item()
        n_gts = torch.clamp(n_gts / distributed_world_size(), min=1).tolist()
        loss = {}
        for k in self.losses:
            loss[k] = self.losses[k] / total_n_gts
        log = {}
        # for k in self.log:
        #     for i in range(len(n_gts)):
        #         if f"frame{i}" in k:
        #             log[k] = (self.log[k] / n_gts[i], 1)
        #             break
        return loss, log

    def calc_loss_for_track_scores(self, track_instances):
        frame_id = self._current_frame_idx - 1
        gt_instances = self.gt_instances[frame_id]
        outputs = {
            'pred_logits': track_instances.track_scores[None],
        }
        device = track_instances.track_scores.device

        num_tracks = len(track_instances)
        src_idx = torch.arange(num_tracks, dtype=torch.long, device=device)
        tgt_idx = track_instances.matched_gt_idxes  # -1 for FP tracks and disappeared tracks

        track_losses = self.get_loss('labels',
                                     outputs=outputs,
                                     gt_instances=[gt_instances],
                                     indices=[(src_idx, tgt_idx)],
                                     num_boxes=1)
        self.losses.update(
            {'frame_{}_track_{}'.format(frame_id, key): value for key, value in
             track_losses.items()})

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def get_loss(self, loss, outputs, gt_instances, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,  # Classification
            'cardinality': self.loss_cardinality,  # Number of boxes
            'boxes': self.loss_boxes,  # Bounding box
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, gt_instances, indices, num_boxes)

    def loss_labels(self, outputs, gt_instances, indices, num_boxes, log=True):
        """
            Classification loss (NLL)
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # The matched gt for disappear track query is set -1.
        labels = []

        for gt_img, (_, id) in zip(gt_instances, indices):
            labels_img = torch.ones_like(id)

            if len(gt_img) > 0:
                # set labels of track-appear slots to 0.
                labels_img[id != -1] = gt_img.labels[id[id != 1]]

            labels.append(labels_img)
        target_classes_o = torch.cat(labels)
        target_classes[idx] = target_classes_o

        if self.focal_loss:
            gt_labels_target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[:, :,
                               :-1]  # no loss for the last (background) class
            gt_labels_target = gt_labels_target.to(src_logits)
            loss_ce = sigmoid_focal_loss(src_logits.flatten(1),
                                         gt_labels_target.flatten(1),
                                         alpha=0.25,
                                         gamma=2,
                                         num_boxes=num_boxes, mean_in_dim1=False)
            loss_ce = loss_ce.sum()
        else:
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_boxes(self, outputs, gt_instances, indices, num_boxes):
        """
            Bounding box regression loss (NLL)
            : Using L1 regression loss and GIoU loss
        """
        assert 'pred_boxes' in outputs
        # Filtering Out Invalid Matches
        filtered_idx = []
        for src_img, tgt_img in indices:
            keep = tgt_img != -1
            filtered_idx.append((src_img[keep], tgt_img[keep]))
        indices = filtered_idx

        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([gt_img.boxes[i] for gt_img, (_, i) in zip(gt_instances, indices)], dim=0)

        # for pad target, don't calculate regression loss, judged by whether obj_id=-1
        target_obj_ids = torch.cat([gt_img.obj_ids[i] for gt_img, (_, i) in zip(gt_instances, indices)],
                                   dim=0)  # size(16)
        mask = (target_obj_ids != -1)

        loss_bbox = F.l1_loss(src_boxes[mask], target_boxes[mask], reduction='none')
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes[mask]),
            box_ops.box_cxcywh_to_xyxy(target_boxes[mask])))

        losses = {'loss_bbox': loss_bbox.sum() / num_boxes, 'loss_giou': loss_giou.sum() / num_boxes}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
            Compute the cardinality error,
            ie the absolute error in the number of predicted non-empty boxes
            This is not really a loss, it is intended for logging purposes only.
            It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)

        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())

        losses = {'cardinality_error': card_err}
        return losses

    def regress_for_single_frame(self, outputs, frame_idx, device, features, cls_head, box_head, occ_threshold=0.5):
        """
            Match all tracks and calculates losses for a single frame.
            The unmatched tracks will be further input into partition networks (New!)
            Args:
                outputs: The model forward results.
                frame_idx: frame_idx t.
                device: cuda basically.
                occ_threshold: For unmatched tracks split.
            Returns:
                track_instances: (It contains losses)
                Unmatched tracks
        """
        # Step 1. Get all ground truths for current (i-th) frame
        gt_instances = self.gt_instances[frame_idx][0]
        track_instances = outputs['track_instances']
        unmatched_instances = None                              # Unmatched Scenarios: 1. Totally missing. 2. Occlusion

        if frame_idx == 0:
            self.prev_instance = track_instances

        # Step 2. Initialise loss dict
        det_loss = {
            "pred_logits": track_instances.pred_logits.unsqueeze(0),    # shape: [1, N, num_classes]
            "pred_boxes": track_instances.pred_boxes.unsqueeze(0),      # shape: [1, N, 4]
        }  # Model outputs for i-th frame

        # Step 3. Get all (ground truth) object index in i-th frame
        obj_idxes_list = gt_instances["obj_ids"].detach().cpu().numpy().tolist()
        obj_gt_mapping = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        # Step 4. Inherit and updating tracked instances.
        # Possible a merge function here _merge_tracks(outputs, tracked_instances)
        # Return tracked_instances
        num_disappear_tracks = 0                                        # Used for calculate total object loss
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()

            if obj_id >= 0:  # Exist tracking
                # Need to refine for object ID storage
                if obj_id not in obj_gt_mapping:
                    num_disappear_tracks += 1  # May not be used for loss calculating
                    track_instances.matched_gt_idxes[j] = -1
                    # track_instances.missing_gt_idxes[j] = obj_gt_mapping[obj_id]
                else:
                    track_instances.matched_gt_idxes[j] = obj_gt_mapping[obj_id]
            else:
                track_instances.matched_gt_idxes[j] = -1  # Missing objects

        # Step 4.1 Matched GT
        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(device)      # [0, 1, ..., N]
        matched_track_idxes = (track_instances.obj_idxes >= 0)                                  # [F, F, T, ..., T]
        # （M, 2) Matched Obj, a table indicating track idx with GT idx
        # prev_matched_track_idxes = torch.stack(
        #                             [full_track_idxes[matched_track_idxes],                     # [[track_idx, gt_idx]]
        #                             track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(device)

        # Step 4.2: Unmatched Objects for whole process (globally)
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # Step 4.3: Untracked objects for current frame
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances["labels"])).to(device)
        tgt_state[tgt_indexes] = 1
        gt_instances["labels"] = gt_instances["labels"].to(device)
        gt_instances["boxes"] = gt_instances["boxes"].to(device)
        untracked_tgt_indexes = torch.arange(len(gt_instances["labels"])).to(device)[tgt_state == 0]
        # untracked_gt_instances = gt_instances[untracked_tgt_indexes]

        unmatched_targets = [{
            'labels': gt_instances["labels"][untracked_tgt_indexes],  # tensor [num_untracked]
            'boxes': gt_instances["boxes"][untracked_tgt_indexes],  # tensor [num_untracked, 4]
        }]

        # Step 5: Hungarian Matching for the unmatched objects
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }                                                                               # Unmatched loss dict
        matching_res = self.matcher(outputs=det_loss, targets=[unmatched_targets])       # Batch = 1 for target （Tuple)
        matching_res = [list(res) for res in matching_res]

        # Step 5.1: Split new object or occluded and Update result:
        new_indices, occ_indices = [], []

        for match in matching_res:
            if match:
                pred_indices = match[0]  # tensor([p1, p2, ...])
                for idx in pred_indices:
                    idx = int(idx)  # 转成 Python int
                    score_val = float(track_instances.scores[idx])
                    if score_val >= occ_threshold:
                        new_indices.append(idx)
                    else:
                        occ_indices.append(idx)

        occluded_tracks = track_instances[occ_indices] if occ_indices else None

        # Step 6: Preparing loss computation
        def matcher_res_for_gt_idx(res, valid_indices, unmatched_track_idxes, untracked_tgt_indexes, device):
            valid_matches = [(src, tgt) for i, (src, tgt) in enumerate(zip(res[0][0], res[0][1]))
                             if i in valid_indices]

            if not valid_matches:
                # 空匹配：返回 shape (0, 2)
                return torch.empty((0, 2), dtype=torch.long, device=device)

            src_idx, tgt_idx = zip(*valid_matches)  # tuple of tensors
            src_idx = torch.as_tensor(src_idx, dtype=torch.long, device=device)
            tgt_idx = torch.as_tensor(tgt_idx, dtype=torch.long, device=device)

            new_matched_indices = torch.stack([
                unmatched_track_idxes[src_idx],
                untracked_tgt_indexes[tgt_idx]
            ], dim=1)  # 保证是 (N, 2)

            return new_matched_indices

        if new_indices:
            new_matched_indices = matcher_res_for_gt_idx(matching_res, new_indices, unmatched_track_idxes, untracked_tgt_indexes, device)
            # 确保将所有张量移动到相同的设备
            new_matched_indices = new_matched_indices.to(track_instances.obj_idxes.device)
            gt_obj_ids = gt_instances["obj_ids"].to(new_matched_indices.device)[new_matched_indices[:, 1]].long()
            gt_obj_ids = gt_obj_ids.to(track_instances.obj_idxes.device)

            # 然后进行赋值
            track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_obj_ids
            track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # Step 6*: Occlusion Recovery Mechanism
        """
            New module here
            Input: Occluded tracks, prev_matched_track_idxes 当前信息和上一帧的信息
            Returns: loss
        """
        if occluded_tracks is not None:
            self.corrector(occluded_tracks, self.prev_instance[occ_indices], frame_idx, features, cls_head, box_head)

        # Step 7: loss computation
        self.sample_device = device

        # 保证 new_indices 是 Tensor
        if not torch.is_tensor(new_indices):
            new_indices = torch.as_tensor(new_indices, dtype=torch.long, device=device)

        for loss in self.loss_name:

            if not torch.is_tensor(new_indices):
                new_indices = torch.as_tensor(new_indices, dtype=torch.long, device=device)

            # 保证维度 (N, 2)
            if new_indices.ndim == 1:
                if new_indices.numel() == 0:
                    new_indices = new_indices.view(0, 2)  # 空匹配
                elif new_indices.numel() == 2:
                    new_indices = new_indices.view(1, 2)  # 单个匹配对
                else:
                    raise ValueError(f"Unexpected new_indices shape: {new_indices.shape}")

            indices = [(new_indices[:, 0], new_indices[:, 1])]

            new_track_loss = self.get_loss(
                loss,
                outputs=outputs,
                gt_instances=[gt_instances],
                indices=indices,
                num_boxes=1
            )
            self.losses.update(
                {
                    f'frame_{self._current_frame_idx}_{key}': value
                    for key, value in new_track_loss.items()
                }
            )

        self.prev_instance = track_instances
        return track_instances

    # def update_track(self, model_res, tracked_instance):
    #     # Update tracks from previous frame(s)
    #     track_instance = Instances((1, 1))
    #     for id in range(len(tracked_instance)):
    #         track_instance[id]['track_instances'] = tracked_instance
    #     return track_instances


def build_criterion(args):
    matcher = build_matcher(args)
    num_frames_per_batch = max(args.SAMPLER_LENGTHS)
    weight_dict = {}
    corrector = OcclusionRecoveryNetwork(window_size=args.WINDOW_SIZE, dim=args.HIDDEN_DIM,
                                         heads=args.NUM_HEADS, dropout=args.DROPOUT)
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.CLS_LOSS_COEF,
                            'frame_{}_loss_bbox'.format(i): args.BBOX_LOSS_COEF,
                            'frame_{}_loss_giou'.format(i): args.GIOU_LOSS_COEF,
                            })
    return FrameMatcher(
        corrector=corrector,
        num_classes=args.NUM_CLASSES,
        matcher=matcher,
        weights=weight_dict,
        losses=['labels', 'boxes'],
        focal_loss=args.FOCAL_LOSS,
    )

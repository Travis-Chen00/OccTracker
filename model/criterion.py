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
from Demos.win32cred_demo import target

from .matcher import build_matcher
from .structures.instances import Instances
from utils.box_ops import generalized_box_iou, box_cxcywh_to_xyxy
from utils.misc import is_dist_avail_and_initialized, get_world_size, accuracy
from .util import sigmoid_focal_loss
from utils import box_ops
from .structures import Boxes, matched_boxlist_iou
from .structures.tracks import Tracks


class FrameMatcher:
    def __init__(self, num_classes, matcher, weights, losses, focal_loss=True):
        """
                Parameters:
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

    def initialize_for_single_clip(self, gt_instances):
        self.losses = {}
        self.device = None
        self.num_samples = 0
        self.gt_instances = gt_instances

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

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

    def match_for_single_frame(self, outputs):
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        track_instances = outputs_without_aux['track_instance']

        gt_instances_ith = self.gt_instances[self._current_frame_idx]  # i-th image
        pred_logits_ith = track_instances.pred_logits  # predicted logits of i-th image.
        pred_boxes_ith = track_instances.pred_boxes  # predicted boxes of i-th image.

        obj_idxes = gt_instances_ith.obj_ids
        obj_idxes_list = obj_idxes.detach().cpu().numpy().tolist()
        # Set mapping from object ID to ground truth ID
        obj_map_gt = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        outputs_ith = {
            'pred_logits': pred_logits_ith.unsqueeze(0),
            'pred_boxes': pred_boxes_ith.unsqueeze(0),
        }

        # step1. inherit and update the previous tracks.
        num_disappear_track = 0
        for t in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[t].item()  # Object ID

            if obj_id >= 0:
                if obj_id not in obj_map_gt:
                    num_disappear_track += 1
                    track_instances.matched_gt_idxes[t] -= 1
                else:
                    track_instances.matched_gt_idxes[t] = obj_map_gt[obj_id]  # Assign the id
            else:
                track_instances.matched_gt_idxes[t] -= 1  # Not matched

        matched_track_idxes = (track_instances.obj_idxes >= 0)  # occu
        full_track_idxes = torch.arange(len(track_instances),
                                        dtype=torch.long).to(pred_logits_ith.device)
        prev_matched_indices = torch.stack(
            [full_track_idxes[matched_track_idxes],
             track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(
            pred_logits_ith.device)

        # step2. select the unmatched slots.
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # Do partition this step

        # Step 3. New track
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances_ith)).to(pred_logits_ith.device)
        untracked_tgt_indexes = torch.arange(len(gt_instances_ith)).to(
            pred_logits_ith.device)[tgt_state == 0]
        untracked_gt_instances = gt_instances_ith[untracked_tgt_indexes]

        # step4. do matching between the unmatched slots and GTs.
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }
        new_matched_indices = match_for_single_decoder_layer(unmatched_outputs, self.matcher,
                                                             untracked_gt_instances,
                                                             unmatched_track_idxes, untracked_tgt_indexes,
                                                             pred_logits_ith)

        # step5. update obj_idxes according to the new matching result.
        track_instances.obj_idxes[new_matched_indices[:, 0]] = (
            gt_instances_ith.obj_ids[new_matched_indices[:, 1]].long())
        track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # step6. calculate iou.
        active_idxes = (track_instances.obj_idxes >= 0) & (track_instances.matched_gt_idxes >= 0)
        active_track_boxes = track_instances.pred_boxes[active_idxes]
        if len(active_track_boxes) > 0:
            gt_boxes = gt_instances_ith.boxes[track_instances.matched_gt_idxes[active_idxes]]
            active_track_boxes = box_ops.box_cxcywh_to_xyxy(active_track_boxes)
            gt_boxes = box_ops.box_cxcywh_to_xyxy(gt_boxes)
            track_instances.iou[active_idxes] = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. calculate the loss
        self.num_samples += len(gt_instances_ith) + num_disappear_track
        self.sample_device = pred_logits_ith.device
        for loss in self.losses:
            new_track_loss = self.get_loss(loss,
                                           outputs=outputs_ith,
                                           gt_instances=[gt_instances_ith],
                                           indices=[(matched_indices[:, 0], matched_indices[:, 1])],
                                           num_boxes=1)
            self.losses_dict.update(
                {'frame_{}_{}'.format(self._current_frame_idx, key): value for key, value in new_track_loss.items()})

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                unmatched_outputs_layer = {
                    'pred_logits': aux_outputs['pred_logits'][0, unmatched_track_idxes].unsqueeze(0),
                    'pred_boxes': aux_outputs['pred_boxes'][0, unmatched_track_idxes].unsqueeze(0),
                }
                new_matched_indices_layer = match_for_single_decoder_layer(unmatched_outputs_layer, self.matcher)
                matched_indices_layer = torch.cat([new_matched_indices_layer, prev_matched_indices], dim=0)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss,
                                           aux_outputs,
                                           gt_instances=[gt_instances_ith],
                                           indices=[(matched_indices_layer[:, 0], matched_indices_layer[:, 1])],
                                           num_boxes=1, )
                    self.losses_dict.update(
                        {'frame_{}_aux{}_{}'.format(self._current_frame_idx, i, key): value for key, value in
                         l_dict.items()})

        return track_instances

    def regress_for_single_frame(self, outputs, frame_idx, device, occ_threshold=0.5):
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
        gt_instances = self.gt_instances[frame_idx]
        track_instances = outputs['track_instance']
        unmatched_instances = None                              # Unmatched Scenarios: 1. Totally missing. 2. Occlusion

        # Step 2. Initialise loss dict
        det_loss = {
            "pred_logits": track_instances.pred_logits.unsqueeze(0),    # shape: [1, N, num_classes]
            "pred_boxes": track_instances.pred_boxes.unsqueeze(0),      # shape: [1, N, 4]
        }  # Model outputs for i-th frame

        # Step 3. Get all (ground truth) object index in i-th frame
        obj_idxes_list = gt_instances.obj_ids.detach().cpu().numpy().tolist()
        obj_gt_mapping = {obj_idx: gt_idx for gt_idx, obj_idx in enumerate(obj_idxes_list)}

        # Step 4. Inherit and updating tracked instances.
        # Possible a merge function here _merge_tracks(outputs, tracked_instances)
        # Return tracked_instances
        num_disappear_tracks = 0                                        # Used for calculate total object loss
        for j in range(len(track_instances)):
            obj_id = track_instances.obj_idxes[j].item()

            if obj_id >= 0:  # Exist tracking
                if obj_id not in obj_gt_mapping:
                    num_disappear_tracks += 1  # May not be used for loss calculating
                    track_instances.matched_gt_idxes[j] = -1
                    track_instances.missing_gt_idxes[j] = obj_gt_mapping[obj_id]
                else:
                    track_instances.matched_gt_idxes[j] = obj_gt_mapping[obj_id]
            else:
                track_instances.matched_gt_idxes[j] = -1  # Missing objects

        # Step 4.1 Matched GT
        full_track_idxes = torch.arange(len(track_instances), dtype=torch.long).to(device)      # [0, 1, ..., N]
        matched_track_idxes = (track_instances.obj_idxes >= 0)                                  # [F, F, T, ..., T]
        # （M, 2) Matched Obj, a table indicating track idx with GT idx
        prev_matched_track_idxes = torch.stack(
                                    [full_track_idxes[matched_track_idxes],                     # [[track_idx, gt_idx]]
                                    track_instances.matched_gt_idxes[matched_track_idxes]], dim=1).to(device)

        # Step 4.2: Unmatched Objects for whole process (globally)
        unmatched_track_idxes = full_track_idxes[track_instances.obj_idxes == -1]

        # Step 4.3: Untracked objects for current frame
        tgt_indexes = track_instances.matched_gt_idxes
        tgt_indexes = tgt_indexes[tgt_indexes != -1]

        tgt_state = torch.zeros(len(gt_instances)).to(device)
        tgt_state[tgt_indexes] = 1
        untracked_tgt_indexes = torch.arange(len(gt_instances)).to(device)[tgt_state == 0]
        untracked_gt_instances = gt_instances[untracked_tgt_indexes]


        # Step 5: Hungarian Matching for the unmatched objects
        unmatched_outputs = {
            'pred_logits': track_instances.pred_logits[unmatched_track_idxes].unsqueeze(0),
            'pred_boxes': track_instances.pred_boxes[unmatched_track_idxes].unsqueeze(0),
        }                                                                               # Unmatched loss dict
        matching_res = self.matcher(outputs=det_loss, target=[unmatched_outputs])       # Batch = 1 for target （Tuple)
        matching_res = [list(res) for res in matching_res]

        # Step 5.1: Split new object or occluded and Update result:
        new_indices, occ_indices = [], []

        for match in matching_res:
            if match:
                idx = match[0]
                if track_instances.scores[idx] >= occ_threshold:
                    new_indices.append(idx)                                             # 理论不需要单独切片 (Merge)
                else:
                    occ_indices.append(idx)                                             # 只针对遮挡物体进行处理

        occluded_tracks = track_instances[occ_indices] if occ_indices else None

        # Step 6: Preparing loss computation
        def matcher_res_for_gt_idx(res, valid_indices):
            # 只处理new_indices对应的匹配
            valid_matches = [(src, tgt) for i, (src, tgt) in enumerate(zip(res[0][0], res[0][1]))
                             if i in valid_indices]

            if not valid_matches:
                return torch.empty((0, 2), dtype=torch.long)

            src_idx, tgt_idx = zip(*valid_matches)
            new_matched_indices = torch.stack([
                unmatched_track_idxes[list(src_idx)],
                untracked_tgt_indexes[list(tgt_idx)]
            ], dim=1).to(device)
            return new_matched_indices

        if new_indices:
            new_matched_indices = matcher_res_for_gt_idx(matching_res, new_indices)
            track_instances.obj_idxes[new_matched_indices[:, 0]] = gt_instances.obj_ids[
                new_matched_indices[:, 1]].long()
            track_instances.matched_gt_idxes[new_matched_indices[:, 0]] = new_matched_indices[:, 1]

        # Step 6*: Occlusion Recovery Mechanism
        """
            New module here
            Input: Occluded tracks, prev_matched_track_idxes 当前信息和上一帧的信息
            Returns: loss 或者更新的框
        """


        # Step 7: loss computation


        return track_instances, unmatched_instances

    # def update_track(self, model_res, tracked_instance):
    #     # Update tracks from previous frame(s)
    #     track_instance = Instances((1, 1))
    #     for id in range(len(tracked_instance)):
    #         track_instance[id]['track_instances'] = tracked_instance
    #     return track_instances


def match_for_single_decoder_layer(unmatched_outputs, matcher, untracked_gt_instances,
                                   unmatched_track_idxes, untracked_tgt_indexes, pred_logits_i):
    new_track_indices = matcher(unmatched_outputs,
                                [untracked_gt_instances])  # list[tuple(src_idx, tgt_idx)]

    src_idx = new_track_indices[0][0]
    tgt_idx = new_track_indices[0][1]
    # concat src and tgt.
    new_matched_indices = torch.stack([unmatched_track_idxes[src_idx],
                                       untracked_tgt_indexes[tgt_idx]],
                                      dim=1).to(pred_logits_i.device)
    return new_matched_indices


def build_criterion(args):
    matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {}
    for i in range(num_frames_per_batch):
        weight_dict.update({"frame_{}_loss_ce".format(i): args.cls_loss_coef,
                            'frame_{}_loss_bbox'.format(i): args.bbox_loss_coef,
                            'frame_{}_loss_giou'.format(i): args.giou_loss_coef,
                            })
    return FrameMatcher(
        num_classes=args.num_classes,
        matcher=matcher,
        weights=weight_dict,
        losses=['labels', 'boxes'],
        focal_loss=args.focal_loss,
    )

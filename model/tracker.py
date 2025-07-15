import torch
from torch import nn
import torch.nn.functional as F
from utils.misc import nested_tensor_from_tensor_list, NestedTensor, inverse_sigmoid
from utils import checkpoint
from model.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou

from .backbone import build_backbone
from .transformer.deformable_transformer import build_deforamble_transformer
from .util import MLP, TrackRecorder
from .criterion import build_criterion
from .structures.tracks import Tracks

class Tracker(nn.Module):
    def __init__(self, args,
                 backbone,
                 transformer,
                 criterion,
                 num_classes,
                 num_queries,
                 num_feature_levels,
                 memory_bank=None,
                 use_checkpoint=False):
        super(Tracker, self).__init__()
        self.args = args
        self.backbone = backbone
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.criterion = criterion

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)

        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        self.use_checkpoint = use_checkpoint
        self.memory_bank = memory_bank
        self.track_recorder = TrackRecorder()
        self.mem_bank_len = 0 if memory_bank is None else memory_bank.max_his_length
        self.image_size = (1,1)

    def _generate_empty_tracks(self, image_size):
        num_queries, dim = self.query_embed.weight.shape  # (300, 512)
        device = self.query_embed.weight.device

        track_instances = Tracks._init_tracks(image_size, dim, num_queries, self.mem_bank_len, device)

        return track_instances.to(self.query_embed.weight.device)

    def _process_single_frame(self, frame, track_instances):
        # Will be changed into forward(), Train the model frame by frame
        # print("Processing single frame")

        features, pos = self.backbone(frame)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs, masks = [], []
        for level, feature in enumerate(features):
            src, mask = feature.decompose()
            srcs.append(self.input_proj[level](src))
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(srcs):
            for level in range(len(srcs), self.num_feature_levels):
                if level == len(srcs):
                    src = self.input_proj[level](features[-1].tensors)
                else:
                    src = self.input_proj[level](srcs[-1])
                m = frame.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        (hs, init_reference,
         inter_references, enc_outputs_class,
         enc_outputs_coord) = self.transformer(srcs, masks,
                                               pos_embed=pos,
                                               query_embed=track_instances.query_pos,
                                               ref_pts=track_instances.ref_pts)
        # hs: result
        # init_reference: Reference points (layer 0)
        # inter_reference: Reference points (layer > 0)

        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            outputs_class = self.class_embed[lvl](hs[lvl])  # Classification
            tmp = self.bbox_embed[lvl](hs[lvl])             # Bounding box
            outputs_coord = tmp.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        ref_pts_all = torch.cat([init_reference[None], inter_references[:, :, :, :2]], dim=0)
        out = {'hs': hs[-1],
               'pred_logits': outputs_class[-1],
               'pred_boxes': outputs_coord[-1],
               'ref_pts': ref_pts_all[5]}
        # if self.aux_loss:
        #     out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    def _matching_frames_with_orm(self, frame_res, track_instances, is_last, frame_idx, cur_frame, prev_frame):
        print("Matching frames with ORM")
        last_frame = False if is_last != 0 else True

        # Using output coordinates to crop the instance
        # TODO: Matching predicted boxes with previous results, if not do next step
        # Bounding box size: coord_w * W, coord_h * H
        # Unmatched box: Disappear or Occluded. --> Occlusion Recovery Module
        with torch.no_grad():
            if self.training:
                track_scores = frame_res['pred_logits'][0, :].sigmoid().max(dim=-1).values
            else:
                track_scores = frame_res['pred_logits'][0, :, 0].sigmoid()

        track_instances.scores = track_scores
        track_instances.pred_logits = frame_res['pred_logits'][0]
        track_instances.pred_boxes = frame_res['pred_boxes'][0]
        track_instances.output_embedding = frame_res['hs'][0]
        if self.training:
            # the track id will be assigned by the mather.
            frame_res['track_instances'] = track_instances
            track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            # each track will be assigned an unique global id by the track base.
            self.track_recorder.update(track_instances)


        out = 1
        return out


    def forward(self, frames):
        outputs = {
            'pred_logits': [],
            'pred_boxes': [],
        }

        _image_size = frames[0][-2:]                            # (H,W)
        track_instances = self._generate_empty_tracks(_image_size)
        keys = list(track_instances._fields.keys())

        for idx, frame in enumerate(frames):
            prev_frame = frame
            frame.requires_grad = False
            _, frame_h, frame_w = frame.shape

            if self.use_checkpoint and idx < len(frames) - 2:
                # Continue the training
                def fn(frame, *args):
                    frame = nested_tensor_from_tensor_list([frame])
                    tmp = Instances((1, 1), **dict(zip(keys, args)))
                    frame_res = self._process_single_frame(frame, tmp)
                    return (
                        frame_res['pred_logits'],
                        frame_res['pred_boxes'],
                        frame_res['ref_pts'],
                        frame_res['hs'],
                        *[aux['pred_logits'] for aux in frame_res['aux_outputs']],
                        *[aux['pred_boxes'] for aux in frame_res['aux_outputs']]
                    )

                args = [frame] + [track_instances.get(k) for k in keys]
                params = tuple((p for p in self.parameters() if p.requires_grad))
                tmp = checkpoint.CheckpointFunction.apply(fn, len(args), *args, *params)
                frame_res = {
                    'pred_logits': tmp[0],
                    'pred_boxes': tmp[1],
                    'ref_pts': tmp[2],
                    'hs': tmp[3],
                    'aux_outputs': [{
                        'pred_logits': tmp[4+i],
                        'pred_boxes': tmp[4+5+i],
                    } for i in range(5)],
                }
            else:
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._process_single_frame(frame, track_instances)
            frame_res = self._matching_frames_with_orm(frame_res, track_instances,
                                                       len(frames) - idx, idx, frame, prev_frame)

            track_instances = frame_res['track_instances']
            outputs['pred_logits'].append(frame_res['pred_logits'])
            outputs['pred_boxes'].append(frame_res['pred_boxes'])

        if not self.training:
            outputs['track_instances'] = track_instances
        else:
            outputs['losses_dict'] = self.criterion.losses_dict
        return outputs

def build_tracker(args):
    dataset_to_num_classes = {
        'nuscences': 10,
    }

    backbone = build_backbone(args)
    transformer = build_deforamble_transformer(args)

    criterion = build_criterion(args)

    model = Tracker(
        backbone=backbone,
        transformer=transformer,
        criterion=criterion,)
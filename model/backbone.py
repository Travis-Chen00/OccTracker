# ------------------------------------------------------------------------
# Modified from MOTR (https://github.com/megvii-research/MOTR/)
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from utils.misc import NestedTensor, is_main_process
from model.position_encoding import build_position_encoding


class FrozenBatchNorm2d(nn.Module):
    """
        BatchNorm2d where the batch statistics and the affine parameters are fixed.

        Copy-paste from torchvision.misc.ops with added eps before rqsrt,
        without which any other models than torchvision.models.resnet[18,34,50,101]
        produce nans.
    """
    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # 加载预训练权重
        # 避免重复训练
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Backbone(nn.Module):
    """
        ResNet with frozen BatchNorm as backbone.
    """
    def __init__(self, backbone_name, train_backbone, return_interm_layers, dilation):
        super(Backbone, self).__init__()
        assert backbone_name == "resnet50", f"Backbone do not support '{backbone_name}'."
        backbone = getattr(torchvision.models, backbone_name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)

        for name, parameter in backbone.named_parameters():
            if not train_backbone or ("layer2" not in name and "layer3" not in name and "layer4" not in name):
                parameter.requires_grad_(False)

        if return_interm_layers:
            return_layers = {
                "layer2": "0",
                "layer3": "1",
                "layer4": "2"
            }
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {"layer4", "0"}
            self.strides = [32]
            self.num_channels = [2048]

        if dilation:
            self.strides[-1] = self.strides[-1] // 2
        self.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list):
        results = self.backbone(tensor_list.tensors)
        out = {}
        for name, output in results.items():
            masks = tensor_list.masks
            assert masks is not None, "Masks should be NOT NONE."
            masks = F.interpolate(masks[None].float(), size=output.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(output, masks)
        return out


class FeatureExtractor(nn.Module):
    """
        Backbone with Position Embedding.
        Input: NestedTensor in (B, C, H, W)
        Output: Multi layer (B, C, H, W) as Image Features, 
                multi layer (B, 2*num_pos_feats, H, W) as Position Embedding
    """
    def __init__(self, backbone, position_embedding):
        super(FeatureExtractor, self).__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list):
        b_out = self.backbone(tensor_list)
        features = []
        pos = []

        # Image Features
        for _, output in sorted(b_out.items()):
            features.append(output)
        # Position Embedding
        for feature in features:
            pos.append(self.position_embedding(feature))

        return features, pos

    def n_inter_layers(self):
        return len(self.strides)

    def n_inter_channels(self):
        return self.num_channels


def build_backbone(args):
    position_embedding = build_position_encoding(args)

    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks or (args.num_feature_levels > 1)

    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    return FeatureExtractor(backbone, position_embedding)

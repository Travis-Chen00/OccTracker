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
import math
import torch.nn as nn

from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from typing import List

from .deformable_encoder import DeformableEncoderLayer, DeformableEncoder
from .deformable_decoder import DeformableDecoderLayer, DeformableDecoder
from ..ops.modules import MSDeformAttn


class DeformableTransformer(nn.Module):
    def __init__(self, device, d_model=256, d_ffn=1024,
                 num_feature_levels=4, num_heads=8,
                 num_enc_points=4, num_dec_points=4,
                 num_enc_layers=6, num_dec_layers=6,
                 dropout=0.1, activation="relu",
                 return_intermediate_dec=False,
                 decoder_self_cross=True,
                 two_stage=False, two_stage_num_proposals=300,
                 use_checkpoint = False,
                 checkpoint_level = 2):
        super(DeformableTransformer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = DeformableEncoderLayer(d_model=d_model, d_ffn=d_ffn,
                                               dropout=dropout, activation=activation,
                                               num_levels=num_feature_levels, num_heads=num_heads,
                                               num_points=num_enc_points,
                                               sigmoid_attn=False)

        self.encoder = DeformableEncoder(encoder_layer, num_enc_layers, use_checkpoint, device)

        decoder_layer = DeformableDecoderLayer(d_model=d_model, d_ffn=d_ffn,
                                               dropout=dropout, activation=activation,
                                               num_levels=num_feature_levels, num_heads=num_heads,
                                               num_points=num_dec_points,
                                               self_cross=decoder_self_cross, sigmoid_attn=False)

        self.decoder = DeformableDecoder(decoder_layer, num_dec_layers, return_intermediate_dec, device)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.reference_points = nn.Linear(d_model, 2)

        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    @staticmethod
    def get_proposal_pos_embed(proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def forward(self, srcs, masks, pos_embeds, query_embed, ref_pts):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        encode_res = self.encoder(src_flatten, spatial_shapes,
                              level_start_index, valid_ratios,
                              lvl_pos_embed_flatten, mask_flatten)

        bs, _, c = encode_res.shape
        # prepare input for decoder
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(encode_res, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference = reference_points
            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)

            if ref_pts is None:
                reference_points = self.reference_points(query_embed).sigmoid()
            else:
                reference_points = ref_pts.unsqueeze(0).repeat(bs, 1, 1).sigmoid()

            init_reference = reference_points
        # decoder
        hs, inter_references = self.decoder(tgt, reference_points, encode_res,
                                            spatial_shapes, level_start_index, valid_ratios, query_embed, mask_flatten)

        if self.two_stage:
            return hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact

        return hs, init_reference, inter_references, None, None


def build_deforamble_transformer(args):
    return DeformableTransformer(
        d_model=args.HIDDEN_DIM,
        d_ffn=args.FFN_DIM,
        num_feature_levels=args.FEATURE_LEVEL,
        num_heads=args.NUM_HEADS,
        num_enc_layers=args.NUM_ENC_LAYERS,
        num_dec_layers=args.NUM_DEC_LAYERS,
        num_enc_points=args.NUM_ENC_POINTS,
        num_dec_points=args.NUM_DEC_POINTS,
        dropout=args.DROPOUT,
        activation="relu",
        return_intermediate_dec=True,
        two_stage=args.TWO_STAGES,
        two_stage_num_proposals=args.NUM_QUERIES,
        decoder_self_cross=not args.DECODER_CROSS_SELF,
        use_checkpoint=args.USE_CHECKPOINT,
        checkpoint_level=args.CHECKPOINT_LEVEL,
        device=args.DEVICE
    )

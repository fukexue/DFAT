import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

from geotransformer.modules.ops import point_to_node_partition, index_select
from geotransformer.modules.registration import get_node_correspondences
from geotransformer.modules.sinkhorn import LearnableLogOptimalTransport
from geotransformer.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)

from backbone import KPConvFPN

from geotransformer.modules.maskautoencoder import PositionEmbeddingCoordsSine, PositionEmbeddingLearned
from geotransformer.modules.maskautoencoder import CreateTransformerDecoder, MaskRegressor, se3_inv, Group, GeometricTransformerDecoder
from geotransformer.modules.lineartransformer import LocalFeatureTransformer
from geotransformer.utils.registration import get_correspondences


class GeoTransformer(nn.Module):
    def __init__(self, cfg):
        super(GeoTransformer, self).__init__()
        self.num_points_in_patch = cfg.model.num_points_in_patch
        self.matching_radius = cfg.model.ground_truth_matching_radius

        self.backbone = KPConvFPN(
            cfg.backbone.input_dim,
            cfg.backbone.output_dim,
            cfg.backbone.init_dim,
            cfg.backbone.kernel_size,
            cfg.backbone.init_radius,
            cfg.backbone.init_sigma,
            cfg.backbone.group_norm,
        )

        self.transformer = GeometricTransformer(
            cfg.geotransformer.input_dim,
            cfg.geotransformer.output_dim,
            cfg.geotransformer.hidden_dim,
            cfg.geotransformer.num_heads,
            cfg.geotransformer.blocks,
            cfg.geotransformer.sigma_d,
            cfg.geotransformer.sigma_a,
            cfg.geotransformer.angle_k,
            reduction_a=cfg.geotransformer.reduction_a,
        )

        # Select GT SuperPoint Correspondences for Training
        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        # Generate Estimated SuperPoint Correspondences for Testing
        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

        #  Local correspondences and Global correspondences to get Global Transform
        self.fine_matching = LocalGlobalRegistration(
            cfg.fine_matching.topk,
            cfg.fine_matching.acceptance_radius,
            mutual=cfg.fine_matching.mutual,
            confidence_threshold=cfg.fine_matching.confidence_threshold,
            use_dustbin=cfg.fine_matching.use_dustbin,
            use_global_score=cfg.fine_matching.use_global_score,
            correspondence_threshold=cfg.fine_matching.correspondence_threshold,
            correspondence_limit=cfg.fine_matching.correspondence_limit,
            num_refinement_steps=cfg.fine_matching.num_refinement_steps,
        )

        # Coarse matching guidance to get local-to-local correspondences
        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

        # Mask AutoEncoder
        self.use_maskautoencoder = cfg.loss.weight_rec_loss > 0
        if self.use_maskautoencoder:
            self.choice = 2
            if self.choice == 1:
                self.group_size = 32
                self.group = Group(32, self.group_size)
                self.proj_mae = nn.Sequential(nn.Linear(cfg.geotransformer.output_dim, cfg.geotransformer.output_dim))
                self.pos_embed = PositionEmbeddingCoordsSine(3, cfg.geotransformer.output_dim, scale=1.0)
                self.transformer_decoder = CreateTransformerDecoder(cfg.geotransformer.output_dim)
                self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.geotransformer.output_dim))
                self.mask_decoder = MaskRegressor(cfg.geotransformer.output_dim, self.group_size)
            else:
                self.group_size = 32
                self.group = Group(32, self.group_size)
                self.proj_mae = nn.Sequential(nn.Linear(cfg.geotransformer.output_dim, cfg.geotransformer.output_dim))
                self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.geotransformer.output_dim))
                self.mask_decoder = MaskRegressor(cfg.geotransformer.output_dim, self.group_size)
                self.transformer_decoder = GeometricTransformerDecoder(cfg.geotransformer.output_dim,
                                                                       cfg.geotransformer.output_dim,
                                                                       cfg.geotransformer.hidden_dim,
                                                                       cfg.geotransformer.num_heads,
                                                                       ['self'],
                                                                       cfg.geotransformer.sigma_d,
                                                                       cfg.geotransformer.sigma_a,
                                                                       cfg.geotransformer.angle_k,
                                                                       reduction_a=cfg.geotransformer.reduction_a)

        # LoFTR
        self.use_loftr = cfg.fine_matching.loftr
        if self.use_loftr:
            # self.register_parameter('loftr_point_scale', torch.nn.Parameter(torch.tensor(1.0)))
            self.localtransformer = LocalFeatureTransformer(d_model=cfg.geotransformer.output_dim, nhead=4, sigma_d=0.05, sigma_a=15, angle_k=3, num_points_in_patch=self.num_points_in_patch)

    def forward(self, data_dict):
        output_dict = {}

        # Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()

        ref_length_c = data_dict['lengths'][-1][0].item()
        ref_length_f = data_dict['lengths'][1][0].item()
        ref_length = data_dict['lengths'][0][0].item()
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][1].detach()
        points = data_dict['points'][0].detach()

        ref_points_c = points_c[:ref_length_c]
        src_points_c = points_c[ref_length_c:]
        ref_points_f = points_f[:ref_length_f]
        src_points_f = points_f[ref_length_f:]
        ref_points = points[:ref_length]
        src_points = points[ref_length:]

        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f
        output_dict['ref_points'] = ref_points
        output_dict['src_points'] = src_points

        # 1. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_node_knn_points,
            src_node_knn_points,
            transform,
            self.matching_radius,
            ref_masks=ref_node_masks,
            src_masks=src_node_masks,
            ref_knn_masks=ref_node_knn_masks,
            src_knn_masks=src_node_knn_masks,
        )

        output_dict['gt_node_corr_indices'] = gt_node_corr_indices
        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps

        # 2. KPFCNN Encoder
        feats_list = self.backbone(feats, data_dict)

        feats_c = feats_list[-1]
        feats_f = feats_list[0]

        # 3. Conditional Transformer
        ref_feats_c = feats_c[:ref_length_c]
        src_feats_c = feats_c[ref_length_c:]
        ref_feats_c, src_feats_c = self.transformer(
            ref_points_c.unsqueeze(0),
            src_points_c.unsqueeze(0),
            ref_feats_c.unsqueeze(0),
            src_feats_c.unsqueeze(0),
            data_dict
        )
        ref_feats_c_norm = F.normalize(ref_feats_c.squeeze(0), p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c.squeeze(0), p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 5. Head for fine level matching
        ref_feats_f = feats_f[:ref_length_f]
        src_feats_f = feats_f[ref_length_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        # 6. Select topk nearest node correspondences
        with torch.no_grad():
            ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_matching(
                ref_feats_c_norm, src_feats_c_norm, ref_points_c, src_points_c, ref_node_masks, src_node_masks
            )

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 7 Random select ground truth node correspondences during training
            if self.training:
                ref_node_corr_indices, src_node_corr_indices, node_corr_scores = self.coarse_target(
                    gt_node_corr_indices, gt_node_corr_overlaps
                )

        # 7.2 Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]  # (P, K)
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]  # (P, K)
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]  # (P, K, 3)
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]  # (P, K, 3)

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)  # (P, K, C)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)  # (P, K, C)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 7.3 add local transformer
        if self.use_loftr:
            # ref_node_corr_knn_points_incenter = (ref_node_corr_knn_points - ref_points_c[ref_node_corr_indices].unsqueeze(1))*self.loftr_point_scale
            # src_node_corr_knn_points_incenter = (src_node_corr_knn_points - src_points_c[src_node_corr_indices].unsqueeze(1))*self.loftr_point_scale
            ref_node_corr_knn_feats, src_node_corr_knn_feats = self.localtransformer(ref_node_corr_knn_feats, src_node_corr_knn_feats,
                                                                                     ref_node_corr_knn_masks, src_node_corr_knn_masks)

        # 8. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)  # (P, K, K)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)

        output_dict['matching_scores'] = matching_scores

        # 9. Generate final correspondences during testing
        with torch.no_grad():
            if not self.fine_matching.use_dustbin:
                matching_scores = matching_scores[:, :-1, :-1]

            ref_corr_points, src_corr_points, corr_scores, estimated_transform = self.fine_matching(
                ref_node_corr_knn_points,
                src_node_corr_knn_points,
                ref_node_corr_knn_masks,
                src_node_corr_knn_masks,
                matching_scores,
                node_corr_scores,
            )

            output_dict['ref_corr_points'] = ref_corr_points
            output_dict['src_corr_points'] = src_corr_points
            output_dict['corr_scores'] = corr_scores
            output_dict['estimated_transform'] = estimated_transform

        corr_indices = get_correspondences(ref_points.detach().cpu(), src_points.detach().cpu(), estimated_transform.detach().cpu(), self.matching_radius/2)
        output_dict['ref_corr_indices'] = corr_indices[:, 0]
        output_dict['src_corr_indices'] = corr_indices[:, 1]

        # Mask AutoEncoder
        if self.use_maskautoencoder:
            if self.choice == 1:
                pose = data_dict['transform']
                inv_pose = se3_inv(pose)
                src_neighbor, src_center = self.group(src_points_c.unsqueeze(0), pose.unsqueeze(0))
                tgt_neighbor, tgt_center = self.group(ref_points_c.unsqueeze(0), inv_pose.unsqueeze(0))

                src_num_mask = tgt_center.shape[1]
                tgt_num_mask = src_center.shape[1]

                src_mask_tokens = self.mask_token.repeat(src_num_mask, 1, 1)
                tgt_mask_tokens = self.mask_token.repeat(tgt_num_mask, 1, 1)

                src_mask_pos = self.pos_embed(tgt_center).transpose(0, 1)
                tgt_mask_pos = self.pos_embed(src_center).transpose(0, 1)

                b, g_src, _, _ = src_neighbor.shape
                b, g_tgt, _, _ = tgt_neighbor.shape

                src_mask_xyz = tgt_neighbor.reshape(b * g_tgt, -1, 3)
                tgt_mask_xyz = src_neighbor.reshape(b * g_src, -1, 3)

                src_masked_mask = torch.zeros(1, src_num_mask, device=ref_points_c.device).to(torch.bool)
                tgt_masked_mask = torch.zeros(1, tgt_num_mask, device=ref_points_c.device).to(torch.bool)

                src_key_padding_mask = torch.zeros(1, src_points_c.shape[0], device=ref_points_c.device).to(torch.bool)
                tgt_key_padding_mask = torch.zeros(1, ref_points_c.shape[0], device=ref_points_c.device).to(torch.bool)

                src_key_padding_mask = torch.cat([src_key_padding_mask, src_masked_mask], dim=1)
                tgt_key_padding_mask = torch.cat([tgt_key_padding_mask, tgt_masked_mask], dim=1)

                src_pe = self.pos_embed(src_points_c.unsqueeze(0)).transpose(0, 1)
                tgt_pe = self.pos_embed(ref_points_c.unsqueeze(0)).transpose(0, 1)

                src_feats_c_tmp = self.proj_mae(src_feats_c)
                ref_feats_c_tmp = self.proj_mae(ref_feats_c)

                src_mask_feats, tgt_mask_feats = self.transformer_decoder(
                    src_feats_c_tmp.permute(1, 0, 2), ref_feats_c_tmp.permute(1, 0, 2),
                    src_mask_token=src_mask_tokens,
                    tgt_mask_token=tgt_mask_tokens,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    src_pos=src_pe,
                    tgt_pos=tgt_pe,
                    src_mask_pos=src_mask_pos,
                    tgt_mask_pos=tgt_mask_pos,
                )

                src_mask_corr, tgt_mask_corr = self.mask_decoder(src_mask_feats, tgt_mask_feats, src_mask_xyz,
                                                                 tgt_mask_xyz)

                output_dict['src_mask_kp'] = src_mask_xyz
                output_dict['src_mask_kp_warped'] = src_mask_corr
                output_dict['tgt_mask_kp'] = tgt_mask_xyz
                output_dict['tgt_mask_kp_warped'] = tgt_mask_corr

            else:
                pose = data_dict['transform']
                inv_pose = se3_inv(pose)
                src_neighbor, src_center = self.group(src_points_c.unsqueeze(0), pose.unsqueeze(0))
                tgt_neighbor, tgt_center = self.group(ref_points_c.unsqueeze(0), inv_pose.unsqueeze(0))

                src_num_mask = tgt_center.shape[1]
                tgt_num_mask = src_center.shape[1]

                src_mask_tokens = self.mask_token.repeat(1, src_num_mask, 1)
                tgt_mask_tokens = self.mask_token.repeat(1, tgt_num_mask, 1)

                b, g_src, _, _ = src_neighbor.shape
                b, g_tgt, _, _ = tgt_neighbor.shape

                src_mask_xyz = tgt_neighbor.reshape(b*g_tgt, -1, 3)
                tgt_mask_xyz = src_neighbor.reshape(b*g_src, -1, 3)

                src_feats_c_tmp = self.proj_mae(src_feats_c)
                ref_feats_c_tmp = self.proj_mae(ref_feats_c)

                tgt_mask_feats, src_mask_feats = self.transformer_decoder(
                    ref_points_c.unsqueeze(0), src_points_c.unsqueeze(0),
                    ref_feats_c_tmp, src_feats_c_tmp,
                    src_center, tgt_center,
                    tgt_mask_tokens, src_mask_tokens,
                )

                src_mask_corr, tgt_mask_corr = self.mask_decoder(src_mask_feats.permute(1, 0, 2).unsqueeze(0),
                                                                 tgt_mask_feats.permute(1, 0, 2).unsqueeze(0),
                                                                 src_mask_xyz, tgt_mask_xyz)

                output_dict['src_mask_kp'] = src_mask_xyz
                output_dict['src_mask_kp_warped'] = src_mask_corr
                output_dict['tgt_mask_kp'] = tgt_mask_xyz
                output_dict['tgt_mask_kp_warped'] = tgt_mask_corr

        return output_dict


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import default_parse_args

    args, cfg = default_parse_args('train')
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()

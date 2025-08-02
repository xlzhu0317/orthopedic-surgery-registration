import torch
import torch.nn as nn
import torch.nn.functional as F

from backbone import KPConvFPN
from src.modules.geotransformer import (
    GeometricTransformer,
    SuperPointMatching,
    SuperPointTargetGenerator,
    LocalGlobalRegistration,
)
from src.modules.ops import point_to_node_partition, index_select
from src.modules.registration import get_node_correspondences
from src.modules.sinkhorn import LearnableLogOptimalTransport


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

        self.coarse_target = SuperPointTargetGenerator(
            cfg.coarse_matching.num_targets, cfg.coarse_matching.overlap_threshold
        )

        self.coarse_matching = SuperPointMatching(
            cfg.coarse_matching.num_correspondences, cfg.coarse_matching.dual_normalization
        )

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

        self.optimal_transport = LearnableLogOptimalTransport(cfg.model.num_sinkhorn_iterations)

    def forward(self, data_dict):
        output_dict = {}

        # 1. Downsample point clouds
        feats = data_dict['features'].detach()
        transform = data_dict['transform'].detach()
        batch_size = data_dict['batch_size']

        lengths_c_flat = data_dict['lengths'][-1].view(-1)
        lengths_f_flat = data_dict['lengths'][0].view(-1)
        points_c = data_dict['points'][-1].detach()
        points_f = data_dict['points'][0].detach()


        ref_lengths_c = lengths_c_flat[:batch_size]
        src_lengths_c = lengths_c_flat[batch_size:]
        total_ref_len_c = torch.sum(ref_lengths_c).item()
        ref_points_c = points_c[:total_ref_len_c]
        src_points_c = points_c[total_ref_len_c:]

        ref_lengths_f = lengths_f_flat[:batch_size]
        src_lengths_f = lengths_f_flat[batch_size:]
        total_ref_len_f = torch.sum(ref_lengths_f).item()
        ref_points_f = points_f[:total_ref_len_f]
        src_points_f = points_f[total_ref_len_f:]


        output_dict['ref_lengths_c'] = ref_lengths_c
        output_dict['src_lengths_c'] = src_lengths_c
        output_dict['ref_points_c'] = ref_points_c
        output_dict['src_points_c'] = src_points_c

        output_dict['ref_lengths_f'] = ref_lengths_f
        output_dict['src_lengths_f'] = src_lengths_f
        output_dict['ref_points_f'] = ref_points_f
        output_dict['src_points_f'] = src_points_f

        output_dict['ref_points'] = ref_points_f
        output_dict['src_points'] = src_points_f


        # 2. Generate ground truth node correspondences
        _, ref_node_masks, ref_node_knn_indices, ref_node_knn_masks = point_to_node_partition(
            ref_points_f, ref_points_c, self.num_points_in_patch, ref_lengths_f, ref_lengths_c
        )
        _, src_node_masks, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points_f, src_points_c, self.num_points_in_patch, src_lengths_f, src_lengths_c
        )

        ref_padded_points_f = torch.cat([ref_points_f, torch.zeros_like(ref_points_f[:1])], dim=0)
        src_padded_points_f = torch.cat([src_points_f, torch.zeros_like(src_points_f[:1])], dim=0)
        ref_node_knn_points = index_select(ref_padded_points_f, ref_node_knn_indices, dim=0)
        src_node_knn_points = index_select(src_padded_points_f, src_node_knn_indices, dim=0)

        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences(
            ref_points_c,
            src_points_c,
            ref_lengths_c,
            src_lengths_c,
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
        ref_feats_c = feats_c[:total_ref_len_c]
        src_feats_c = feats_c[total_ref_len_c:]

        ref_points_c_list = ref_points_c.split(ref_lengths_c.cpu().tolist())
        src_points_c_list = src_points_c.split(src_lengths_c.cpu().tolist())
        ref_feats_c_list = ref_feats_c.split(ref_lengths_c.cpu().tolist())
        src_feats_c_list = src_feats_c.split(src_lengths_c.cpu().tolist())

        ref_feats_c_b_list, src_feats_c_b_list = [], []
        for i in range(batch_size):
            ref_fc_i, src_fc_i = self.transformer(
                ref_points_c_list[i].unsqueeze(0),
                src_points_c_list[i].unsqueeze(0),
                ref_feats_c_list[i].unsqueeze(0),
                src_feats_c_list[i].unsqueeze(0)
            )
            ref_feats_c_b_list.append(ref_fc_i.squeeze(0))
            src_feats_c_b_list.append(src_fc_i.squeeze(0))

        ref_feats_c = torch.cat(ref_feats_c_b_list, dim=0)
        src_feats_c = torch.cat(src_feats_c_b_list, dim=0)

        ref_feats_c_norm = F.normalize(ref_feats_c, p=2, dim=1)
        src_feats_c_norm = F.normalize(src_feats_c, p=2, dim=1)

        output_dict['ref_feats_c'] = ref_feats_c_norm
        output_dict['src_feats_c'] = src_feats_c_norm

        # 4. Head for fine level matching
        ref_feats_f = feats_f[:total_ref_len_f]
        src_feats_f = feats_f[total_ref_len_f:]
        output_dict['ref_feats_f'] = ref_feats_f
        output_dict['src_feats_f'] = src_feats_f

        ref_feats_c_norm_list = ref_feats_c_norm.split(ref_lengths_c.cpu().tolist())
        src_feats_c_norm_list = src_feats_c_norm.split(src_lengths_c.cpu().tolist())
        ref_node_masks_list = ref_node_masks.split(ref_lengths_c.cpu().tolist())
        src_node_masks_list = src_node_masks.split(src_lengths_c.cpu().tolist())

        ref_node_corr_indices_list, src_node_corr_indices_list, node_corr_scores_list = [], [], []
        ref_offset, src_offset = 0, 0

        # 5. Coarse_matching
        with torch.no_grad():
            # 5.1 Select topk nearest node correspondences
            for i in range(batch_size):
                ref_node_corr_indices_i, src_node_corr_indices_i, node_corr_scores_i = self.coarse_matching(
                    ref_feats_c_norm_list[i], src_feats_c_norm_list[i],
                    ref_node_masks_list[i], src_node_masks_list[i]
                )

                ref_node_corr_indices_list.append(ref_node_corr_indices_i + ref_offset)
                src_node_corr_indices_list.append(src_node_corr_indices_i + src_offset)
                node_corr_scores_list.append(node_corr_scores_i)

                ref_offset += ref_lengths_c[i]
                src_offset += src_lengths_c[i]

            ref_node_corr_indices = torch.cat(ref_node_corr_indices_list, dim=0)
            src_node_corr_indices = torch.cat(src_node_corr_indices_list, dim=0)
            node_corr_scores = torch.cat(node_corr_scores_list, dim=0)

            output_dict['ref_node_corr_indices'] = ref_node_corr_indices
            output_dict['src_node_corr_indices'] = src_node_corr_indices

            # 5.2 Random select ground truth node correspondences during training
            # if self.training:
            #     gt_corr_lengths = []
            #     src_node_boundaries = torch.cumsum(src_lengths_c, dim=0)
            #     start_node_index = 0
            #     for boundary in src_node_boundaries:
            #         mask = (gt_node_corr_indices[:, 1] >= start_node_index) & (gt_node_corr_indices[:, 1] < boundary)
            #         gt_corr_lengths.append(torch.sum(mask).item())
            #         start_node_index = boundary
            #
            #     ref_offset, src_offset = 0, 0
            #     ref_node_corr_indices_final_list, src_node_corr_indices_final_list, node_corr_scores_final_list = [], [], []
            #
            #     for i in range(batch_size):
            #         gt_ref_mask_i = (gt_node_corr_indices[:, 0] >= ref_offset) & (gt_node_corr_indices[:, 0] < ref_offset + ref_lengths_c[i])
            #         gt_src_mask_i = (gt_node_corr_indices[:, 1] >= src_offset) & (gt_node_corr_indices[:, 1] < src_offset + src_lengths_c[i])
            #         gt_mask_i = gt_ref_mask_i & gt_src_mask_i
            #
            #         gt_corr_indices_i = gt_node_corr_indices[gt_mask_i]
            #         gt_corr_overlaps_i = gt_node_corr_overlaps[gt_mask_i]
            #
            #         gt_corr_indices_i[:, 0] -= ref_offset
            #         gt_corr_indices_i[:, 1] -= src_offset
            #
            #         ref_indices_i, src_indices_i, scores_i = self.coarse_target(gt_corr_indices_i, gt_corr_overlaps_i)
            #
            #         ref_node_corr_indices_final_list.append(ref_indices_i + ref_offset)
            #         src_node_corr_indices_final_list.append(src_indices_i + src_offset)
            #         node_corr_scores_final_list.append(scores_i)
            #
            #         ref_offset += ref_lengths_c[i]
            #         src_offset += src_lengths_c[i]
            #
            #     ref_node_corr_indices = torch.cat(ref_node_corr_indices_final_list, dim=0)
            #     src_node_corr_indices = torch.cat(src_node_corr_indices_final_list, dim=0)
            #     node_corr_scores = torch.cat(node_corr_scores_final_list, dim=0)

        # 6. Generate batched node points & feats
        ref_node_corr_knn_indices = ref_node_knn_indices[ref_node_corr_indices]
        src_node_corr_knn_indices = src_node_knn_indices[src_node_corr_indices]
        ref_node_corr_knn_masks = ref_node_knn_masks[ref_node_corr_indices]
        src_node_corr_knn_masks = src_node_knn_masks[src_node_corr_indices]
        ref_node_corr_knn_points = ref_node_knn_points[ref_node_corr_indices]
        src_node_corr_knn_points = src_node_knn_points[src_node_corr_indices]

        ref_padded_feats_f = torch.cat([ref_feats_f, torch.zeros_like(ref_feats_f[:1])], dim=0)
        src_padded_feats_f = torch.cat([src_feats_f, torch.zeros_like(src_feats_f[:1])], dim=0)
        ref_node_corr_knn_feats = index_select(ref_padded_feats_f, ref_node_corr_knn_indices, dim=0)
        src_node_corr_knn_feats = index_select(src_padded_feats_f, src_node_corr_knn_indices, dim=0)

        output_dict['ref_node_corr_knn_points'] = ref_node_corr_knn_points
        output_dict['src_node_corr_knn_points'] = src_node_corr_knn_points
        output_dict['ref_node_corr_knn_masks'] = ref_node_corr_knn_masks
        output_dict['src_node_corr_knn_masks'] = src_node_corr_knn_masks

        # 7. Optimal transport
        matching_scores = torch.einsum('bnd,bmd->bnm', ref_node_corr_knn_feats, src_node_corr_knn_feats)
        matching_scores = matching_scores / feats_f.shape[1] ** 0.5
        matching_scores = self.optimal_transport(matching_scores, ref_node_corr_knn_masks, src_node_corr_knn_masks)
        output_dict['matching_scores'] = matching_scores

        # 8. Generate final correspondences during testing
        with torch.no_grad():
            corr_lengths = []
            src_node_boundaries = torch.cumsum(src_lengths_c, dim=0)
            start_node_index = 0
            for boundary in src_node_boundaries:
                mask = (src_node_corr_indices >= start_node_index) & (src_node_corr_indices < boundary)
                num_corrs = torch.sum(mask).item()
                corr_lengths.append(num_corrs)
                start_node_index = boundary

            ref_knn_points_list = ref_node_corr_knn_points.split(corr_lengths)
            src_knn_points_list = src_node_corr_knn_points.split(corr_lengths)
            ref_knn_masks_list = ref_node_corr_knn_masks.split(corr_lengths)
            src_knn_masks_list = src_node_corr_knn_masks.split(corr_lengths)
            scores_list = matching_scores.split(corr_lengths)
            node_scores_list = node_corr_scores.split(corr_lengths)

            ref_corr_points_list = []
            src_corr_points_list = []
            corr_scores_list = []
            est_transform_list = []

            for i in range(batch_size):
                if corr_lengths[i] == 0:
                    ref_corr_points_list.append(torch.empty(0, 3, device=transform.device))
                    src_corr_points_list.append(torch.empty(0, 3, device=transform.device))
                    corr_scores_list.append(torch.empty(0, device=transform.device))
                    est_transform_list.append(torch.eye(4, device=transform.device))
                    continue

                scores_i = scores_list[i]
                if not self.fine_matching.use_dustbin: scores_i = scores_i[:, :-1, :-1]

                ref_corr_points_i, src_corr_points_i, corr_scores_i, est_transform_i = self.fine_matching(
                    ref_knn_points_list[i],
                    src_knn_points_list[i],
                    ref_knn_masks_list[i],
                    src_knn_masks_list[i],
                    scores_i,
                    node_scores_list[i]
                )
                ref_corr_points_list.append(ref_corr_points_i)
                src_corr_points_list.append(src_corr_points_i)
                corr_scores_list.append(corr_scores_i)
                est_transform_list.append(est_transform_i)

            fine_corr_lengths = torch.tensor([len(x) for x in ref_corr_points_list], device=transform.device)
            output_dict['fine_corr_lengths'] = fine_corr_lengths
            output_dict['ref_corr_points'] = torch.cat(ref_corr_points_list, dim=0)
            output_dict['src_corr_points'] = torch.cat(src_corr_points_list, dim=0)
            output_dict['corr_scores'] = torch.cat(corr_scores_list, dim=0)
            output_dict['estimated_transform'] = torch.stack(est_transform_list, dim=0)

        return output_dict


def create_model(config):
    model = GeoTransformer(config)
    return model


def main():
    from config import make_cfg

    cfg = make_cfg()
    model = create_model(cfg)
    print(model.state_dict().keys())
    print(model)


if __name__ == '__main__':
    main()

import torch
import torch.nn as nn

from src.modules.loss import WeightedCircleLoss
from src.modules.ops.transformation import apply_transform
from src.modules.registration.metrics import isotropic_transform_error
from src.modules.ops.pairwise_distance import pairwise_distance


class CoarseMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(CoarseMatchingLoss, self).__init__()
        self.weighted_circle_loss = WeightedCircleLoss(
            cfg.coarse_loss.positive_margin,
            cfg.coarse_loss.negative_margin,
            cfg.coarse_loss.positive_optimal,
            cfg.coarse_loss.negative_optimal,
            cfg.coarse_loss.log_scale,
        )
        self.positive_overlap = cfg.coarse_loss.positive_overlap

    def forward(self, output_dict):
        ref_feats = output_dict['ref_feats_c']
        src_feats = output_dict['src_feats_c']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]

        feat_dists = torch.sqrt(pairwise_distance(ref_feats, src_feats, normalized=True))

        overlaps = torch.zeros_like(feat_dists)
        overlaps[gt_ref_node_corr_indices, gt_src_node_corr_indices] = gt_node_corr_overlaps
        pos_masks = torch.gt(overlaps, self.positive_overlap)
        neg_masks = torch.eq(overlaps, 0)
        pos_scales = torch.sqrt(overlaps * pos_masks.float())

        loss = self.weighted_circle_loss(pos_masks, neg_masks, feat_dists, pos_scales)

        return loss


class FineMatchingLoss(nn.Module):
    def __init__(self, cfg):
        super(FineMatchingLoss, self).__init__()
        self.positive_radius = cfg.fine_loss.positive_radius

    def forward(self, output_dict, data_dict):
        ref_node_corr_knn_points = output_dict['ref_node_corr_knn_points']
        src_node_corr_knn_points = output_dict['src_node_corr_knn_points']
        ref_node_corr_knn_masks = output_dict['ref_node_corr_knn_masks']
        src_node_corr_knn_masks = output_dict['src_node_corr_knn_masks']
        matching_scores = output_dict['matching_scores']
        transform = data_dict['transform']
        batch_size = transform.shape[0]

        # ----- START OF CORE MODIFICATION -----
        # 使用與 model.py 中完全一致的、基於 src 索引的穩健邏輯來計算 corr_lengths

        # FineMatchingLoss 作用於 coarse_matching 或 coarse_target 的輸出
        # 因此我們需要用它們輸出的 src_node_corr_indices
        src_node_corr_indices = output_dict['src_node_corr_indices']
        src_lengths_c = output_dict['src_lengths_c']

        corr_lengths = []
        src_node_boundaries = torch.cumsum(src_lengths_c, dim=0)
        start_node_index = 0
        for boundary in src_node_boundaries:
            mask = (src_node_corr_indices >= start_node_index) & (src_node_corr_indices < boundary)
            num_corrs = torch.sum(mask).item()
            corr_lengths.append(num_corrs)
            start_node_index = boundary
        # ----- END OF CORE MODIFICATION -----

        ref_points_list, src_points_list = ref_node_corr_knn_points.split(corr_lengths), src_node_corr_knn_points.split(
            corr_lengths)
        ref_masks_list, src_masks_list = ref_node_corr_knn_masks.split(corr_lengths), src_node_corr_knn_masks.split(
            corr_lengths)
        scores_list = matching_scores.split(corr_lengths)

        total_loss, num_samples = 0.0, 0
        for i in range(batch_size):
            if corr_lengths[i] == 0: continue

            transformed_src_points_i = apply_transform(src_points_list[i], transform[i])
            dists_i = pairwise_distance(ref_points_list[i], transformed_src_points_i)
            gt_masks_i = torch.logical_and(ref_masks_list[i].unsqueeze(2), src_masks_list[i].unsqueeze(1))
            gt_corr_map_i = torch.lt(dists_i, self.positive_radius ** 2)
            gt_corr_map_i = torch.logical_and(gt_corr_map_i, gt_masks_i)
            slack_row_labels_i = torch.logical_and(torch.eq(gt_corr_map_i.sum(2), 0), ref_masks_list[i])
            slack_col_labels_i = torch.logical_and(torch.eq(gt_corr_map_i.sum(1), 0), src_masks_list[i])

            labels_i = torch.zeros_like(scores_list[i], dtype=torch.bool, device=scores_list[i].device)
            labels_i[:, :-1, :-1] = gt_corr_map_i
            labels_i[:, :-1, -1] = slack_row_labels_i
            labels_i[:, -1, :-1] = slack_col_labels_i

            if labels_i.any():
                total_loss += -scores_list[i][labels_i].mean()
                num_samples += 1

        return total_loss / num_samples if num_samples > 0 else torch.tensor(0.0, device=transform.device)


class OverallLoss(nn.Module):
    def __init__(self, cfg):
        super(OverallLoss, self).__init__()
        self.coarse_loss = CoarseMatchingLoss(cfg)
        self.fine_loss = FineMatchingLoss(cfg)
        self.weight_coarse_loss = cfg.loss.weight_coarse_loss
        self.weight_fine_loss = cfg.loss.weight_fine_loss

    def forward(self, output_dict, data_dict):
        coarse_loss = self.coarse_loss(output_dict)
        fine_loss = self.fine_loss(output_dict, data_dict)

        loss = self.weight_coarse_loss * coarse_loss + self.weight_fine_loss * fine_loss

        return {
            'loss': loss,
            'c_loss': coarse_loss,
            'f_loss': fine_loss,
        }


class Evaluator(nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c, src_length_c = output_dict['ref_points_c'].shape[0], output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps, gt_node_corr_indices = output_dict['gt_node_corr_overlaps'], output_dict[
            'gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices, gt_src_node_corr_indices = gt_node_corr_indices[:, 0], gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c, device=gt_node_corr_indices.device)
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0
        ref_node_corr_indices, src_node_corr_indices = output_dict['ref_node_corr_indices'], output_dict[
            'src_node_corr_indices']
        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()
        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform']
        ref_corr_points, src_corr_points = output_dict['ref_corr_points'], output_dict['src_corr_points']
        batch_size = transform.shape[0]
        if batch_size == 1:
            src_corr_points = apply_transform(src_corr_points, transform.squeeze(0))
            corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
            return torch.lt(corr_distances, self.acceptance_radius).float().mean()

        fine_corr_lengths = output_dict['fine_corr_lengths']
        ref_corr_points_list, src_corr_points_list = ref_corr_points.split(
            fine_corr_lengths.cpu().tolist()), src_corr_points.split(fine_corr_lengths.cpu().tolist())

        total_precision, num_samples = 0.0, 0
        for i in range(batch_size):
            if fine_corr_lengths[i] == 0: continue
            src_corr_points_i = apply_transform(src_corr_points_list[i], transform[i])
            corr_distances_i = torch.linalg.norm(ref_corr_points_list[i] - src_corr_points_i, dim=1)
            total_precision += torch.lt(corr_distances_i, self.acceptance_radius).float().mean()
            num_samples += 1
        return total_precision / num_samples if num_samples > 0 else torch.tensor(1.0, device=transform.device)

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        transform, est_transform = data_dict['transform'], output_dict['estimated_transform']
        rre, rte = isotropic_transform_error(transform, est_transform)

        batch_size = transform.shape[0]
        src_points_f = output_dict['src_points_f']  # Use fine points for RMSE
        src_lengths_f = output_dict['src_lengths_f']
        src_points_f_list = src_points_f.split(src_lengths_f.cpu().tolist())

        total_rmse, total_recall = 0.0, 0.0
        for i in range(batch_size):
            realignment_transform_i = torch.matmul(torch.inverse(transform[i]), est_transform[i])
            realigned_src_points_f_i = apply_transform(src_points_f_list[i], realignment_transform_i)
            rmse_i = torch.linalg.norm(realigned_src_points_f_i - src_points_f_list[i], dim=1).mean()
            total_rmse += rmse_i
            total_recall += torch.lt(rmse_i, self.acceptance_rmse).float()

        return rre.mean(), rte.mean(), total_rmse / batch_size, total_recall / batch_size

    def forward(self, output_dict, data_dict):
        c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, rmse, recall = self.evaluate_registration(output_dict, data_dict)
        return {'PIR': c_precision, 'IR': f_precision, 'RRE': rre, 'RTE': rte, 'RMSE': rmse, 'RR': recall}


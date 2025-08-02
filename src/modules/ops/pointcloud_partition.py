import warnings

import torch

from src.modules.ops.pairwise_distance import pairwise_distance
from src.modules.ops.index_select import index_select


def get_point_to_node_indices(points: torch.Tensor, nodes: torch.Tensor, return_counts: bool = False):

    sq_dist_mat = pairwise_distance(points, nodes)
    indices = sq_dist_mat.min(dim=1)[1]
    if return_counts:
        unique_indices, unique_counts = torch.unique(indices, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()
        node_sizes[unique_indices] = unique_counts
        return indices, node_sizes
    else:
        return indices


@torch.no_grad()
def knn_partition(points: torch.Tensor, nodes: torch.Tensor, k: int, return_distance: bool = False):
    r"""k-NN partition of the point cloud.

    Find the k nearest points for each node.

    Args:
        points: torch.Tensor (num_point, num_channel)
        nodes: torch.Tensor (num_node, num_channel)
        k: int
        return_distance: bool

    Returns:
        knn_indices: torch.Tensor (num_node, k)
        knn_indices: torch.Tensor (num_node, k)
    """
    k = min(k, points.shape[0])
    sq_dist_mat = pairwise_distance(nodes, points)
    knn_sq_distances, knn_indices = sq_dist_mat.topk(dim=1, k=k, largest=False)
    if return_distance:
        knn_distances = torch.sqrt(knn_sq_distances)
        return knn_distances, knn_indices
    else:
        return knn_indices


def _point_to_node_partition_single(
        points: torch.Tensor,
        nodes: torch.Tensor,
        point_limit: int,
):
    """
    Original point_to_node_partition logic, for a single, non-batched sample.
    """
    # Ensure all tensors are on the same device
    device = points.device

    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool, device=device)
    node_masks.index_fill_(0, point_to_node, True)

    matching_masks = torch.zeros_like(sq_dist_mat, dtype=torch.bool)  # (M, N)
    point_indices = torch.arange(points.shape[0], device=device)  # (N,)
    matching_masks[point_to_node, point_indices] = True  # (M, N)
    sq_dist_mat.masked_fill_(~matching_masks, 1e12)  # (M, N)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, k)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0], device=device).unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def point_to_node_partition(
        points: torch.Tensor,
        nodes: torch.Tensor,
        point_limit: int,
        points_lengths: torch.Tensor,
        nodes_lengths: torch.Tensor,
        return_count: bool = False,
):
    batch_size = nodes_lengths.shape[0]

    # Split the concatenated tensors into lists of tensors, one for each sample in the batch
    points_list = points.split(points_lengths.cpu().tolist())
    nodes_list = nodes.split(nodes_lengths.cpu().tolist())

    # Lists to store results from each sample
    point_to_node_list = []
    node_masks_list = []
    node_knn_indices_list = []
    node_knn_masks_list = []

    # Offsets for concatenating indices later
    points_offset = 0
    nodes_offset = 0

    # Process each sample in the batch individually
    for i in range(batch_size):
        points_i = points_list[i]
        nodes_i = nodes_list[i]

        # Call the single-sample helper function
        p2n_i, n_masks_i, n_knn_indices_i, n_knn_masks_i = _point_to_node_partition_single(
            points_i, nodes_i, point_limit
        )

        # Apply offsets to indices to make them valid for the concatenated tensors
        p2n_i += nodes_offset

        # Adjust both the valid indices and the sentinel value for knn_indices
        sentinel_mask_i = (n_knn_indices_i == points_i.shape[0])
        n_knn_indices_i += points_offset
        n_knn_indices_i[sentinel_mask_i] = points.shape[0]  # Use the global sentinel value

        # Append results to lists
        point_to_node_list.append(p2n_i)
        node_masks_list.append(n_masks_i)
        node_knn_indices_list.append(n_knn_indices_i)
        node_knn_masks_list.append(n_knn_masks_i)

        # Update offsets for the next sample
        points_offset += points_i.shape[0]
        nodes_offset += nodes_i.shape[0]

    point_to_node = torch.cat(point_to_node_list, dim=0)
    node_masks = torch.cat(node_masks_list, dim=0)
    node_knn_indices = torch.cat(node_knn_indices_list, dim=0)
    node_knn_masks = torch.cat(node_knn_masks_list, dim=0)

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long, device=nodes.device)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def point_to_node_partition_bug(
    points: torch.Tensor,
    nodes: torch.Tensor,
    point_limit: int,
    return_count: bool = False,
):
    r"""Point-to-Node partition to the point cloud.

    BUG: this implementation ignores point_to_node indices when building patches. However, the points that do not
    belong to a superpoint should be masked out.


    Args:
        points (Tensor): (N, 3)
        nodes (Tensor): (M, 3)
        point_limit (int): max number of points to each node
        return_count (bool=False): whether to return `node_sizes`

    Returns:
        point_to_node (Tensor): (N,)
        node_sizes (LongTensor): (M,)
        node_masks (BoolTensor): (M,)
        node_knn_indices (LongTensor): (M, K)
        node_knn_masks (BoolTensor) (M, K)
    """
    warnings.warn('There is a bug in this implementation. Use `point_to_node_partition` instead.')
    sq_dist_mat = pairwise_distance(nodes, points)  # (M, N)

    point_to_node = sq_dist_mat.min(dim=0)[1]  # (N,)
    node_masks = torch.zeros(nodes.shape[0], dtype=torch.bool).cuda()  # (M,)
    node_masks.index_fill_(0, point_to_node, True)

    node_knn_indices = sq_dist_mat.topk(k=point_limit, dim=1, largest=False)[1]  # (M, K)
    node_knn_node_indices = index_select(point_to_node, node_knn_indices, dim=0)  # (M, K)
    node_indices = torch.arange(nodes.shape[0]).cuda().unsqueeze(1).expand(-1, point_limit)  # (M, K)
    node_knn_masks = torch.eq(node_knn_node_indices, node_indices)  # (M, K)
    node_knn_indices.masked_fill_(~node_knn_masks, points.shape[0])

    if return_count:
        unique_indices, unique_counts = torch.unique(point_to_node, return_counts=True)
        node_sizes = torch.zeros(nodes.shape[0], dtype=torch.long).cuda()  # (M,)
        node_sizes.index_put_([unique_indices], unique_counts)
        return point_to_node, node_sizes, node_masks, node_knn_indices, node_knn_masks
    else:
        return point_to_node, node_masks, node_knn_indices, node_knn_masks


@torch.no_grad()
def ball_query_partition(
    points: torch.Tensor,
    nodes: torch.Tensor,
    radius: float,
    point_limit: int,
    return_count: bool = False,
):
    node_knn_distances, node_knn_indices = knn_partition(points, nodes, point_limit, return_distance=True)
    node_knn_masks = torch.lt(node_knn_distances, radius)  # (N, k)
    sentinel_indices = torch.full_like(node_knn_indices, points.shape[0])  # (N, k)
    node_knn_indices = torch.where(node_knn_masks, node_knn_indices, sentinel_indices)  # (N, k)

    if return_count:
        node_sizes = node_knn_masks.sum(1)  # (N,)
        return node_knn_indices, node_knn_masks, node_sizes
    else:
        return node_knn_indices, node_knn_masks

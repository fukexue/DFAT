import torch
import torch.nn as nn

from geotransformer.modules.ops import pairwise_distance


def sinkhorn_local(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    out_alpha = torch.exp(log_alpha)

    return out_alpha


class SuperPointMatching(nn.Module):
    def __init__(self, num_correspondences, dual_normalization=True):
        super(SuperPointMatching, self).__init__()
        self.num_correspondences = num_correspondences
        self.dual_normalization = dual_normalization

    def forward(self, ref_feats, src_feats, ref_points, src_points, ref_masks=None, src_masks=None, attention_scores=None):
        r"""Extract superpoint correspondences.

        Args:
            ref_feats (Tensor): features of the superpoints in reference point cloud.
            src_feats (Tensor): features of the superpoints in source point cloud.
            ref_points (Tensor): coordinates of the superpoints in reference point cloud.
            src_points (Tensor): coordinates of the superpoints in source point cloud.
            ref_masks (BoolTensor=None): masks of the superpoints in reference point cloud (False if empty).
            src_masks (BoolTensor=None): masks of the superpoints in source point cloud (False if empty).
            attention_scores (Tensor=None): attention scores of the superpoints in reference point cloud.

        Returns:
            ref_corr_indices (LongTensor): indices of the corresponding superpoints in reference point cloud.
            src_corr_indices (LongTensor): indices of the corresponding superpoints in source point cloud.
            corr_scores (Tensor): scores of the correspondences.
        """
        if ref_masks is None:
            ref_masks = torch.ones(size=(ref_feats.shape[0],), dtype=torch.bool).cuda()
        if src_masks is None:
            src_masks = torch.ones(size=(src_feats.shape[0],), dtype=torch.bool).cuda()
        # remove empty patch
        ref_indices = torch.nonzero(ref_masks, as_tuple=True)[0]
        src_indices = torch.nonzero(src_masks, as_tuple=True)[0]
        ref_feats = ref_feats[ref_indices]
        src_feats = src_feats[src_indices]
        # select top-k proposals
        matching_scores = torch.exp(-pairwise_distance(ref_feats, src_feats, normalized=True))
        if self.dual_normalization:
            ref_matching_scores = matching_scores / matching_scores.sum(dim=1, keepdim=True)
            src_matching_scores = matching_scores / matching_scores.sum(dim=0, keepdim=True)
            matching_scores = ref_matching_scores * src_matching_scores
            # matching_scores = sinkhorn_local(matching_scores.unsqueeze(0), n_iters=50).squeeze(0)
        num_correspondences = min(self.num_correspondences, matching_scores.numel())
        corr_scores, corr_indices = matching_scores.view(-1).topk(k=num_correspondences, largest=True)
        ref_sel_indices = torch.div(corr_indices, matching_scores.shape[1], rounding_mode='trunc')  # corr_indices // matching_scores.shape[1]
        src_sel_indices = corr_indices % matching_scores.shape[1]
        # recover original indices
        ref_corr_indices = ref_indices[ref_sel_indices]
        src_corr_indices = src_indices[src_sel_indices]
        # remove correspondences of must not match
        ref_dist = ((ref_points[ref_indices, None, :] - ref_points[None, ref_indices, :]) ** 2).sum(-1) ** 0.5
        ref_dist[ref_dist == 0] = 1000
        src_dist = ((src_points[src_indices, None, :] - src_points[None, src_indices, :]) ** 2).sum(-1) ** 0.5
        src_dist[src_dist == 0] = 1000
        ref_nearest_indices = torch.argmin(ref_dist, dim=-1)
        src_nearest_indices = torch.argmin(src_dist, dim=-1)
        ref_corr_nearest_indices = ref_nearest_indices[ref_sel_indices]
        src_corr_nearest_indices = src_nearest_indices[src_sel_indices]
        corr_nearest_score = matching_scores[ref_corr_nearest_indices, src_corr_nearest_indices]
        # corr_nearest_dist = ((ref_points[ref_corr_nearest_indices, :] - src_points[src_corr_nearest_indices, :]) ** 2).sum(-1) ** 0.5
        ref_corr_indices = ref_corr_indices[corr_nearest_score > 1/(matching_scores.shape[-1]*matching_scores.shape[-2])]
        src_corr_indices = src_corr_indices[corr_nearest_score > 1/(matching_scores.shape[-1]*matching_scores.shape[-2])]
        corr_scores = corr_scores[corr_nearest_score > 1/(matching_scores.shape[-1]*matching_scores.shape[-2])]

        # ref_full_dist = ((ref_points[:, None, :] - ref_points[None, :, :]) ** 2).sum(-1) ** 0.5
        # ref_full_dist[ref_full_dist == 0] = 1000
        # src_full_dist = ((src_points[:, None, :] - src_points[None, :, :]) ** 2).sum(-1) ** 0.5
        # src_full_dist[src_full_dist == 0] = 1000
        # ref_full_nearest_indices = torch.argmin(ref_full_dist, dim=-1)
        # src_full_nearest_indices = torch.argmin(src_full_dist, dim=-1)
        # ref_full_corr_nearest_indices = ref_full_nearest_indices[ref_corr_indices]
        # src_full_corr_nearest_indices = src_full_nearest_indices[src_corr_indices]
        # attention_score = torch.zeros_like(attention_scores[1][0].mean(1).squeeze(0))
        # for i in range(1, len(attention_scores), 2):
        #     attention_score += (attention_scores[i][0].mean(1).squeeze(0)+attention_scores[i][1].permute(0, 1, 3, 2).mean(1).squeeze(0))*0.5
        # attention_score = attention_score / len(attention_scores)
        # corr_nearest_score1 = attention_score[ref_full_corr_nearest_indices, src_full_corr_nearest_indices]
        #
        # ref_corr_indices = ref_corr_indices[corr_nearest_score1 > 1/max(matching_scores.shape[-1], matching_scores.shape[-2])]
        # src_corr_indices = src_corr_indices[corr_nearest_score1 > 1/max(matching_scores.shape[-1], matching_scores.shape[-2])]

        if ref_corr_indices.shape[0] < 1:
            print('no corr')

        return ref_corr_indices, src_corr_indices, corr_scores

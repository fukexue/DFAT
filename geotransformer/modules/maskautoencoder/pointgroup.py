import torch
import torch.nn as nn
from knn_cuda import KNN
from .se3_torch import se3_inv, se3_transform
from pointnet2_ops import pointnet2_utils


def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, pc, pose):
        """
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        """
        xyz = pc.clone()
        while xyz.shape[1] < self.group_size:
            max_len = min(xyz.shape[1], self.group_size-xyz.shape[1])
            xyz = torch.cat([xyz, torch.zeros_like(xyz[:, :max_len])], dim=1)

        B, N, _ = xyz.shape
        xyz = se3_transform(pose, xyz).contiguous()
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group)  # B G 3
        # knn to get the neighborhood
        _, idx_n = self.knn(xyz, center)  # B G M
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx_n + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


def pad_sequence(sequences, require_padding_mask=False, require_lens=False,
                 batch_first=False):
    """List of sequences to padded sequences

    Args:
        sequences: List of sequences (N, D)
        require_padding_mask:

    Returns:
        (padded_sequence, padding_mask), where
           padded sequence has shape (N_max, B, D)
           padding_mask will be none if require_padding_mask is False
    """
    padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
    padding_mask = None
    padding_lens = None

    if require_padding_mask:
        B = len(sequences)
        seq_lens = list(map(len, sequences))
        padding_mask = torch.zeros((B, padded.shape[0]), dtype=torch.bool, device=padded.device)
        for i, l in enumerate(seq_lens):
            padding_mask[i, l:] = True

    if require_lens:
        padding_lens = [seq.shape[0] for seq in sequences]

    return padded, padding_mask, padding_lens
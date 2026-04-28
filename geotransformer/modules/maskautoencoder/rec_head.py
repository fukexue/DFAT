import torch
import torch.nn as nn


class MaskRegressor(nn.Module):

    def __init__(self, d_embed, group_size):
        super().__init__()

        self.coor_mlp = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Linear(d_embed, 3 * group_size)
        )

    def forward(self, src_feats_padded, tgt_feats_padded, src_xyz, tgt_xyz):
        """

        Args:
            src_feats_padded: Source features ([N_pred,] N_src, B, D)
            tgt_feats_padded: Target features ([N_pred,] N_tgt, B, D)
            src_xyz: List of ([N_pred,] N_src, 3). Ignored
            tgt_xyz: List of ([N_pred,] N_tgt, 3). Ignored

        Returns:

        """
        _, g_src, B, _ = src_feats_padded.shape
        _, g_tgt, B, _ = tgt_feats_padded.shape

        # Decode the coordinates
        src_corr = self.coor_mlp(src_feats_padded).transpose(1, 2)
        tgt_corr = self.coor_mlp(tgt_feats_padded).transpose(1, 2)

        src_corr = src_corr.reshape(g_src*B, -1, 3)
        tgt_corr = tgt_corr.reshape(g_tgt*B, -1, 3)

        return src_corr, tgt_corr

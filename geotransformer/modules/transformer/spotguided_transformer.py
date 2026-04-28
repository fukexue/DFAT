r"""SpotGuided Transformer without positional embeddings.

The shape of input tensor should be (B, N, C). Implemented with `nn.Linear` and `nn.LayerNorm` (with affine).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from geotransformer.modules.layers import build_dropout_layer
from geotransformer.modules.transformer.output_layer import AttentionOutput
from geotransformer.modules.ops import index_select


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(MultiHeadAttention, self).__init__()
        if d_model % num_heads != 0:
            raise ValueError('`d_model` ({}) must be a multiple of `num_heads` ({}).'.format(d_model, num_heads))

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_model_per_head = d_model // num_heads

        self.proj_q = nn.Linear(self.d_model, self.d_model)
        self.proj_k = nn.Linear(self.d_model, self.d_model)
        self.proj_v = nn.Linear(self.d_model, self.d_model)

        self.dropout = build_dropout_layer(dropout)

    def forward(
        self, input_q, input_k, input_v, key_weights=None, key_masks=None, attention_factors=None, attention_masks=None
    ):
        """Vanilla Self-attention forward propagation.

        Args:
            input_q (Tensor): input tensor for query (B, N, C)
            input_k (Tensor): input tensor for key (B, M, C)
            input_v (Tensor): input tensor for value (B, M, C)
            key_weights (Tensor): soft masks for the keys (B, M)
            key_masks (BoolTensor): True if ignored, False if preserved (B, M)
            attention_factors (Tensor): factors for attention matrix (B, N, M)
            attention_masks (BoolTensor): True if ignored, False if preserved (B, N, M)

        Returns:
            hidden_states: torch.Tensor (B, C, N)
            attention_scores: intermediate values
                'attention_scores': torch.Tensor (B, H, N, M), attention scores before dropout
        """
        q = rearrange(self.proj_q(input_q), 'b n (h c) -> b h n c', h=self.num_heads)
        k = rearrange(self.proj_k(input_k), 'b m (h c) -> b h m c', h=self.num_heads)
        v = rearrange(self.proj_v(input_v), 'b m (h c) -> b h m c', h=self.num_heads)

        attention_scores = torch.einsum('bhnc,bhmc->bhnm', q, k) / self.d_model_per_head ** 0.5
        if attention_factors is not None:
            attention_scores = attention_factors.unsqueeze(1) * attention_scores
        if key_weights is not None:
            attention_scores = attention_scores * key_weights.unsqueeze(1).unsqueeze(1)
        if key_masks is not None:
            attention_scores = attention_scores.masked_fill(key_masks.unsqueeze(1).unsqueeze(1), float('-inf'))
        if attention_masks is not None:
            attention_scores = attention_scores.masked_fill(attention_masks, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_scores = self.dropout(attention_scores)

        hidden_states = torch.matmul(attention_scores, v)

        hidden_states = rearrange(hidden_states, 'b h n c -> b n (h c)')

        return hidden_states, attention_scores


class AttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None):
        super(AttentionLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = build_dropout_layer(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        input_states,
        memory_states,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        hidden_states, attention_scores = self.attention(
            input_states,
            memory_states,
            memory_states,
            key_weights=memory_weights,
            key_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        hidden_states = self.linear(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output_states = self.norm(hidden_states + input_states)
        return output_states, attention_scores


class SpotGuidedTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=None, activation_fn='ReLU'):
        super(SpotGuidedTransformerLayer, self).__init__()
        self.attention = AttentionLayer(d_model, num_heads, dropout=dropout)
        self.output = AttentionOutput(d_model, dropout=dropout, activation_fn=activation_fn)
        self.topk = 3

    def forward(
        self,
        input_states,
        memory_states,
        data_dict,
        memory_weights=None,
        memory_masks=None,
        attention_factors=None,
        attention_masks=None,
    ):
        feat_states_in = torch.cat([input_states, memory_states], 1)
        feat_states = feat_states_in.squeeze(0)
        corre_score = torch.matmul(feat_states, feat_states.transpose(0, 1))/(feat_states.shape[-1]**0.5)
        corre_score = F.softmax(corre_score, -1)*F.softmax(corre_score, -2)

        feat_states_pad = torch.cat([feat_states, torch.zeros_like(feat_states[:1, :])], 0)  # (N, D) -> (N+1, D)
        feat_states_pad = index_select(feat_states_pad, data_dict['neighbors'][-1], dim=0)  # (N+1, 3) -> (M, H, 3)
        sim_score = torch.einsum('bnc,bmc->bnm', feat_states_pad[:, 0:1, :], feat_states_pad[:, 1:, :])
        sim_score = sim_score.masked_fill(sim_score==0, float('-inf'))
        sim_score = F.softmax(sim_score, -1).squeeze(1)
        corre_score_rowmax = corre_score.max(dim=-1, keepdim=True)[0]
        corre_score_rowmax_id = corre_score.max(dim=-1, keepdim=True)[1]
        corre_score_rowmax = torch.cat([corre_score_rowmax, torch.zeros_like(corre_score_rowmax[0:1, :])], dim=0)
        corre_score_rowmax_id = torch.cat([corre_score_rowmax_id, torch.zeros_like(corre_score_rowmax_id[0:1, :])], dim=0)
        conf_score = corre_score_rowmax[data_dict['neighbors'][-1], :].squeeze(-1)
        conf_score_id = corre_score_rowmax_id[data_dict['neighbors'][-1], :].squeeze(-1)
        sel_score = sim_score * conf_score[:, 1:]    # 视为特征相似度叠加空间结构相似度比重，空间结构相似度越高代表显著性越高。
        sel_spot = sel_score.topk(self.topk, dim=-1)
        sel_spot_id = conf_score_id[:, 1:].gather(-1, sel_spot.indices)
        sel_spot_id = torch.cat([conf_score_id[:, 0:1], sel_spot_id], dim=-1)

        source_feat = feat_states_pad[sel_spot_id, :, :] #可能会存在Bug,有可能数组越界
        source_feat = source_feat.mean(dim=-2)

        hidden_states, attention_scores = self.attention(
            feat_states_in.transpose(0, 1),
            source_feat,
            memory_weights=memory_weights,
            memory_masks=memory_masks,
            attention_factors=attention_factors,
            attention_masks=attention_masks,
        )
        output_states = self.output(hidden_states.transpose(0, 1))
        return output_states[:, :input_states.shape[1], :], output_states[:, input_states.shape[1]:, :]


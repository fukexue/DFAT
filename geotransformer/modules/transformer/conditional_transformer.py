import torch
import torch.nn as nn

from geotransformer.modules.transformer.lrpe_transformer import LRPETransformerLayer
from geotransformer.modules.transformer.pe_transformer import PETransformerLayer
from geotransformer.modules.transformer.rpe_transformer import RPETransformerLayer
from geotransformer.modules.transformer.vanilla_transformer import TransformerLayer
from geotransformer.modules.transformer.spotguided_transformer import SpotGuidedTransformerLayer
from geotransformer.modules.kpconv import nearest_upsample


def _check_block_type(block):
    if block not in ['self', 'cross', 'only_cross']:
        raise ValueError('Unsupported block type "{}".'.format(block))


class VanillaConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(VanillaConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class PEConditionalTransformer(nn.Module):
    def __init__(self, blocks, d_model, num_heads, dropout=None, activation_fn='ReLU', return_attention_scores=False):
        super(PEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(PETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, embeddings0, embeddings1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, embeddings1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class RPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
        parallel=False,
    ):
        super(RPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        layers_sga = []
        # proj_up = []
        # proj_down = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(RPETransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                layers_sga.append(nn.Identity())
                # proj_up.append(nn.Linear(d_model, d_model*2))
                # proj_down.append(nn.Linear(d_model*2, d_model))
            elif block == 'only_cross':
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                layers_sga.append(nn.Identity())
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
                layers_sga.append(SpotGuidedTransformerLayer(d_model, num_heads))
                # proj_up.append(nn.Linear(d_model, d_model*2))
                # proj_down.append(nn.Linear(d_model*2, d_model))
        self.layers = nn.ModuleList(layers)
        self.layers_sga = nn.ModuleList(layers_sga)
        # self.proj_up = nn.ModuleList(proj_up)
        # self.proj_down = nn.ModuleList(proj_down)
        self.return_attention_scores = return_attention_scores
        self.parallel = parallel

    def forward(self, feats0, feats1, embeddings0, embeddings1, data_dict, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, embeddings0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, embeddings1, memory_masks=masks1)
                feats0 = self.layers_sga[i](feats0)
                feats1 = self.layers_sga[i](feats1)
                # feats0and1 = self.proj_up[i](nearest_upsample(torch.cat([feats0, feats1], dim=1).squeeze(0), data_dict['upsampling'][2]))
                # feats0_f = feats0_f + feats0and1[:feats0_f.shape[1]].unsqueeze(0)
                # feats1_f = feats1_f + feats0and1[feats0_f.shape[1]:].unsqueeze(0)
                # feats0and1_f = self.proj_down[i](nearest_upsample(torch.cat([feats0_f, feats1_f], dim=1).squeeze(0), data_dict['subsampling'][2]))
                # feats0 = feats0 + feats0and1_f[:feats0.shape[1]].unsqueeze(0)
                # feats1 = feats1 + feats0and1_f[feats0.shape[1]:].unsqueeze(0)
            elif block == 'only_cross':
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                feats0 = self.layers_sga[i](feats0)
                feats1 = self.layers_sga[i](feats1)
            else:
                if self.parallel:
                    new_feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    new_feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    new_feats0, new_feats1 = self.layers_f[i](new_feats0, new_feats1, data_dict)
                    feats0 = new_feats0
                    feats1 = new_feats1
                else:
                    feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                    feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
                    feats0, feats1 = self.layers_sga[i](feats0, feats1, data_dict)
                    # feats0and1 = self.proj_up[i](nearest_upsample(torch.cat([feats0, feats1], dim=1).squeeze(0), data_dict['upsampling'][2]))
                    # feats0_f = feats0_f + feats0and1[:feats0_f.shape[1]].unsqueeze(0)
                    # feats1_f = feats1_f + feats0and1[feats0_f.shape[1]:].unsqueeze(0)
                    # feats0and1_f = self.proj_down[i](nearest_upsample(torch.cat([feats0_f, feats1_f], dim=1).squeeze(0), data_dict['subsampling'][2]))
                    # feats0 = feats0 + feats0and1_f[:feats0.shape[1]].unsqueeze(0)
                    # feats1 = feats1 + feats0and1_f[feats0.shape[1]:].unsqueeze(0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1


class LRPEConditionalTransformer(nn.Module):
    def __init__(
        self,
        blocks,
        d_model,
        num_heads,
        num_embeddings,
        dropout=None,
        activation_fn='ReLU',
        return_attention_scores=False,
    ):
        super(LRPEConditionalTransformer, self).__init__()
        self.blocks = blocks
        layers = []
        for block in self.blocks:
            _check_block_type(block)
            if block == 'self':
                layers.append(
                    LRPETransformerLayer(
                        d_model, num_heads, num_embeddings, dropout=dropout, activation_fn=activation_fn
                    )
                )
            else:
                layers.append(TransformerLayer(d_model, num_heads, dropout=dropout, activation_fn=activation_fn))
        self.layers = nn.ModuleList(layers)
        self.return_attention_scores = return_attention_scores

    def forward(self, feats0, feats1, emb_indices0, emb_indices1, masks0=None, masks1=None):
        attention_scores = []
        for i, block in enumerate(self.blocks):
            if block == 'self':
                feats0, scores0 = self.layers[i](feats0, feats0, emb_indices0, memory_masks=masks0)
                feats1, scores1 = self.layers[i](feats1, feats1, emb_indices1, memory_masks=masks1)
            else:
                feats0, scores0 = self.layers[i](feats0, feats1, memory_masks=masks1)
                feats1, scores1 = self.layers[i](feats1, feats0, memory_masks=masks0)
            if self.return_attention_scores:
                attention_scores.append([scores0, scores1])
        if self.return_attention_scores:
            return feats0, feats1, attention_scores
        else:
            return feats0, feats1

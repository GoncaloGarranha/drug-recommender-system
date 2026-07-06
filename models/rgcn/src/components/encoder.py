from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv


class RGCNEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        h_dim: int,
        num_rels: int,
        num_bases: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_rels = num_rels
        self.num_bases = min(num_bases, num_rels) if num_bases > 0 else None

        self.emb = nn.Embedding(num_nodes, h_dim)

        self.layers = nn.ModuleList([
            RelGraphConv(
                in_feat=h_dim,
                out_feat=h_dim,
                num_rels=num_rels,
                regularizer="basis",
                num_bases=self.num_bases,
                self_loop=True,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(h_dim)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, graph, edge_types: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight

        for layer_idx, layer in enumerate(self.layers):
            residual = x

            h = layer(graph, x, edge_types)
            h = self.norms[layer_idx](h)

            if layer_idx < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)

            x = h + residual

        return x
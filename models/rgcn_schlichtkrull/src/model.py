from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv


class SchlichtkrullRGCN(nn.Module):
    """R-GCN encoder following the Schlichtkrull et al. link prediction setup."""

    def __init__(
        self,
        num_nodes: int,
        h_dim: int,
        num_rels: int,
        num_bases: int,
        num_layers: int,
        dropout: float,
        regularizer: str = "basis",
    ) -> None:
        super().__init__()
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.layers = nn.ModuleList()
        bases = min(num_bases, num_rels) if num_bases > 0 else None

        for _ in range(num_layers):
            self.layers.append(
                RelGraphConv(
                    in_feat=h_dim,
                    out_feat=h_dim,
                    num_rels=num_rels,
                    regularizer=regularizer,
                    num_bases=bases,
                    self_loop=True,
                    dropout=dropout,
                )
            )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, graph, edge_types: torch.Tensor) -> torch.Tensor:
        h = self.emb.weight
        for layer_idx, layer in enumerate(self.layers):
            h = layer(graph, h, edge_types)
            if layer_idx < len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


class DistMult(nn.Module):
    """DistMult decoder used for R-GCN link prediction."""

    def __init__(self, num_rels: int, h_dim: int) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(num_rels, h_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def score(self, head: torch.Tensor, rel: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        return torch.sum(head * rel * tail, dim=-1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        heads: torch.Tensor,
        rels: torch.Tensor,
        tails: torch.Tensor,
    ) -> torch.Tensor:
        return self.score(
            node_embeddings[heads],
            self.rel_emb(rels),
            node_embeddings[tails],
        )

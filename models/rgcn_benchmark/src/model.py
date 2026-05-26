from __future__ import annotations

import math

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
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.dropout_p = dropout
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.layers = nn.ModuleList()
        self.self_loop_layers = nn.ModuleList()
        # For basis decomposition, capping by num_rels is common.
        if num_bases > 0:
            bases = min(num_bases, num_rels)
        else:
            bases = None

        for _ in range(num_layers):
            layer = RelGraphConv(
                in_feat=h_dim,
                out_feat=h_dim,
                num_rels=num_rels,
                regularizer=regularizer,
                num_bases=bases,
                self_loop=True,
                dropout=0.0,
            )
            self.layers.append(layer)
            self.self_loop_layers.append(nn.Linear(h_dim, h_dim, bias=False))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 3.0 / math.sqrt(self.num_nodes + self.h_dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=std)
        for layer in self.self_loop_layers:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, graph, edge_types: torch.Tensor, edge_norm: torch.Tensor | None = None) -> torch.Tensor:
        h = F.relu(self.emb.weight)
        for layer_idx, layer in enumerate(self.layers):
            if edge_norm is not None:
                if edge_norm.dim() == 1:
                    edge_norm = edge_norm.unsqueeze(1)
                message_h = layer(graph, h, edge_types, edge_norm)
            else:
                message_h = layer(graph, h, edge_types)

            self_loop_h = self.self_loop_layers[layer_idx](h)
            self_loop_h = F.dropout(self_loop_h, p=self.dropout_p, training=self.training)
            h = message_h + self_loop_h
            if layer_idx < len(self.layers) - 1:
                h = F.relu(h)
        return h


class DistMult(nn.Module):
    """DistMult decoder used for R-GCN link prediction."""

    def __init__(self, num_rels: int, h_dim: int) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(num_rels, h_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.rel_emb.weight, mean=0.0, std=1.0)

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

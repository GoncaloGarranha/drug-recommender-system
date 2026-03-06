from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv


class RGCNEncoder(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) encoder.

    This module learns node embeddings by propagating information through
    relation-aware graph convolution layers.

    Args:
        num_nodes: Total number of nodes in the graph.
        h_dim: Hidden embedding dimension.
        num_rels: Number of relation types.
        num_bases: Number of bases used in basis decomposition.
        num_layers: Number of R-GCN layers.
        dropout: Dropout probability applied between hidden layers.
    """

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

        # Learnable initial node embeddings.
        self.emb = nn.Embedding(num_nodes, h_dim)

        # Stacked relational graph convolution layers.
        self.layers = nn.ModuleList(
            [
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
            ]
        )

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize learnable parameters.
        """
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, graph, edge_types: torch.Tensor) -> torch.Tensor:
        """
        Compute node embeddings for the full graph.

        Args:
            graph: DGL graph.
            edge_types: Tensor containing the relation type of each edge.

        Returns:
            Tensor of shape [num_nodes, h_dim] with the final node embeddings.
        """
        x = self.emb.weight

        for layer_idx, layer in enumerate(self.layers):
            x = layer(graph, x, edge_types)

            # Apply non-linearity and dropout after every layer except the last.
            if layer_idx < len(self.layers) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        return x
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Backend detection ─────────────────────────────────────────────────────────
try:
    from dgl.nn.pytorch import RelGraphConv
    _BACKEND = "dgl"
except ImportError:
    try:
        from torch_geometric.nn import RGCNConv
        _BACKEND = "pyg"
    except ImportError:
        raise ImportError(
            "Neither DGL nor PyTorch Geometric is installed. "
            "Install one: `pip install dgl` or `pip install torch_geometric`."
        )
# ─────────────────────────────────────────────────────────────────────────────


class SchlichtkrullRGCN(nn.Module):
    """R-GCN encoder following the Schlichtkrull et al. link prediction setup.

    Automatically uses DGL when available (local/Windows), otherwise falls
    back to PyTorch Geometric (Colab/Kaggle). External interface is identical
    in both cases — the rest of the codebase does not need to change.
    """

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
        self._backend = _BACKEND
        self.emb = nn.Embedding(num_nodes, h_dim)
        self.dropout = nn.Dropout(dropout)
        bases = min(num_bases, num_rels) if num_bases > 0 else None

        if _BACKEND == "dgl":
            self.layers = nn.ModuleList([
                RelGraphConv(
                    in_feat=h_dim,
                    out_feat=h_dim,
                    num_rels=num_rels,
                    regularizer=regularizer,
                    num_bases=bases,
                    self_loop=True,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])
        else:  # pyg
            self.layers = nn.ModuleList([
                RGCNConv(
                    in_channels=h_dim,
                    out_channels=h_dim,
                    num_relations=num_rels,
                    num_bases=bases,
                    aggr="mean",
                )
                for _ in range(num_layers)
            ])

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.emb.weight, -1.0, 1.0)

    def forward(
        self,
        graph,
        edge_types: torch.Tensor,
        edge_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.emb.weight

        if self._backend == "dgl":
            for layer_idx, layer in enumerate(self.layers):
                if edge_norm is not None:
                    norm = edge_norm if edge_norm.dim() == 2 else edge_norm.unsqueeze(1)
                    h = layer(graph, h, edge_types, norm)
                else:
                    h = layer(graph, h, edge_types)
                if layer_idx < len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
        else:  # pyg — graph is edge_index [2, E], edge_norm ignored (PyG handles internally)
            for layer_idx, layer in enumerate(self.layers):
                h = layer(h, graph, edge_types)
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
        nn.init.uniform_(self.rel_emb.weight, -1.0, 1.0)

    def score(
        self,
        head: torch.Tensor,
        rel: torch.Tensor,
        tail: torch.Tensor,
    ) -> torch.Tensor:
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
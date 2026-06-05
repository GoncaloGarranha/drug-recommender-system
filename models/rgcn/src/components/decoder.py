from __future__ import annotations

import torch
import torch.nn as nn


class DistMultDecoder(nn.Module):
    """DistMult decoder for scoring knowledge graph triples."""

    def __init__(self, num_rels: int, h_dim: int) -> None:
        super().__init__()
        self.rel_emb = nn.Embedding(num_rels, h_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def score(
        self,
        head_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(head_emb * rel_emb * tail_emb, dim=-1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        src_idx: torch.Tensor,
        rel_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        head_emb = node_embeddings[src_idx]
        rel_emb = self.rel_emb(rel_idx)
        tail_emb = node_embeddings[dst_idx]
        return self.score(head_emb, rel_emb, tail_emb)

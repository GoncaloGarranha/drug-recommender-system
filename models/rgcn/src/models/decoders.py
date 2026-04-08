from __future__ import annotations

import torch
import torch.nn as nn


class DistMultDecoder(nn.Module):
    """
    DistMult decoder for scoring knowledge graph triples.

    Each relation is represented by a learnable embedding vector. Triple scores
    are computed with the DistMult interaction:
        score(h, r, t) = sum(h * r * t)

    Args:
        num_rels: Number of relation types.
        h_dim: Embedding dimension.
    """

    def __init__(self, num_rels: int, h_dim: int) -> None:
        super().__init__()

        self.rel_emb = nn.Embedding(num_rels, h_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Initialize relation embeddings.
        """
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def score(
        self,
        head_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute DistMult scores for batches of triples.

        Args:
            head_emb: Head entity embeddings.
            rel_emb: Relation embeddings.
            tail_emb: Tail entity embeddings.

        Returns:
            A tensor of triple scores.
        """
        return torch.sum(head_emb * rel_emb * tail_emb, dim=-1)

    def forward(
        self,
        node_embeddings: torch.Tensor,
        src_idx: torch.Tensor,
        rel_idx: torch.Tensor,
        dst_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score indexed triples using node and relation embeddings.

        Args:
            node_embeddings: Tensor with all node embeddings.
            src_idx: Head entity indices.
            rel_idx: Relation indices.
            dst_idx: Tail entity indices.

        Returns:
            A tensor of triple scores.
        """
        head_emb = node_embeddings[src_idx]
        rel_emb = self.rel_emb(rel_idx)
        tail_emb = node_embeddings[dst_idx]

        return self.score(head_emb, rel_emb, tail_emb)
"""
DistMult standalone model for link prediction.

Pure bilinear scoring without R-GCN encoder.
Entity embeddings + Relation embeddings → score(h, r, t) = h^T R_r t
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DistMult(nn.Module):
    """
    DistMult decoder: score(h, r, t) = sum_i h_i * r_i * t_i
    
    Corresponds to the bilinear form h^T R_r t where R_r is a diagonal matrix.
    """

    def __init__(self, num_relations: int, emb_dim: int):
        super().__init__()
        self.num_relations = num_relations
        self.emb_dim = emb_dim
        self.rel_emb = nn.Embedding(num_relations, emb_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.rel_emb.weight, -1.0, 1.0)

    def score(
        self,
        head_emb: torch.Tensor,
        rel_emb: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples using DistMult.
        
        Args:
            head_emb: (batch_size, emb_dim) or (num_entities, emb_dim)
            rel_emb: (batch_size, emb_dim) or (num_relations, emb_dim)
            tail_emb: (batch_size, emb_dim) or (num_entities, emb_dim)
        
        Returns:
            scores: (batch_size,) or (num_entities,) depending on input shapes
        """
        # Element-wise product: (batch, emb_dim) * (batch, emb_dim) * (batch, emb_dim)
        # Then sum over embedding dimension
        return torch.sum(head_emb * rel_emb * tail_emb, dim=-1)

    def forward(
        self,
        head_emb: torch.Tensor,
        rel_ids: torch.Tensor,
        tail_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: embed relations and score.
        
        Args:
            head_emb: (batch_size, emb_dim)
            rel_ids: (batch_size,) relation IDs
            tail_emb: (batch_size, emb_dim)
        
        Returns:
            scores: (batch_size,)
        """
        rel_emb = self.rel_emb(rel_ids)
        return self.score(head_emb, rel_emb, tail_emb)


class DistMultStandalone(nn.Module):
    """
    Standalone DistMult model with learnable entity and relation embeddings.
    """

    def __init__(self, num_entities: int, num_relations: int, emb_dim: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.emb_dim = emb_dim

        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.distmult = DistMult(num_relations, emb_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.entity_emb.weight, -1.0, 1.0)
        self.distmult.reset_parameters()

    def forward(
        self,
        head_ids: torch.Tensor,
        rel_ids: torch.Tensor,
        tail_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score triples.
        
        Args:
            head_ids: (batch_size,)
            rel_ids: (batch_size,)
            tail_ids: (batch_size,)
        
        Returns:
            scores: (batch_size,)
        """
        head_emb = self.entity_emb(head_ids)
        tail_emb = self.entity_emb(tail_ids)
        return self.distmult(head_emb, rel_ids, tail_emb)

    def score_all_tails(
        self,
        head_id: torch.Tensor,
        rel_id: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all candidate tails for a (head, rel) pair.
        
        Args:
            head_id: scalar tensor
            rel_id: scalar tensor
            candidate_ids: (num_candidates,)
        
        Returns:
            scores: (num_candidates,)
        """
        head_emb = self.entity_emb(head_id).unsqueeze(0)  # (1, emb_dim)
        rel_emb = self.distmult.rel_emb(rel_id).unsqueeze(0)  # (1, emb_dim)
        tail_embs = self.entity_emb(candidate_ids)  # (num_candidates, emb_dim)

        return self.distmult.score(
            head_emb.expand(len(candidate_ids), -1),
            rel_emb.expand(len(candidate_ids), -1),
            tail_embs,
        )

    def score_all_heads(
        self,
        rel_id: torch.Tensor,
        tail_id: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Score all candidate heads for a (rel, tail) pair.
        
        Args:
            rel_id: scalar tensor
            tail_id: scalar tensor
            candidate_ids: (num_candidates,)
        
        Returns:
            scores: (num_candidates,)
        """
        tail_emb = self.entity_emb(tail_id).unsqueeze(0)  # (1, emb_dim)
        rel_emb = self.distmult.rel_emb(rel_id).unsqueeze(0)  # (1, emb_dim)
        head_embs = self.entity_emb(candidate_ids)  # (num_candidates, emb_dim)

        return self.distmult.score(
            head_embs,
            rel_emb.expand(len(candidate_ids), -1),
            tail_emb.expand(len(candidate_ids), -1),
        )

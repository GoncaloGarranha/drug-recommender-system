from __future__ import annotations

import torch

from .components import DistMultDecoder


def score_batch(
    task_name: str,
    decoder: DistMultDecoder,
    node_embeddings: torch.Tensor,
    queries: torch.Tensor,
    relations: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Score candidate triples using the model's DistMult decoder."""
    rel_emb = decoder.rel_emb(relations)

    if task_name == "disease_to_drug":
        disease_emb = node_embeddings[queries]
        drug_emb = node_embeddings[candidates]
        return torch.sum(disease_emb * rel_emb * drug_emb, dim=1)

    if task_name == "drug_to_disease":
        drug_emb = node_embeddings[queries]
        disease_emb = node_embeddings[candidates]
        return torch.sum(drug_emb * rel_emb * disease_emb, dim=1)

    if task_name == "tail_prediction":
        src_emb = node_embeddings[queries]
        dst_emb = node_embeddings[candidates]
        return torch.sum(src_emb * rel_emb * dst_emb, dim=1)

    raise ValueError(f"Unsupported task name: {task_name}")

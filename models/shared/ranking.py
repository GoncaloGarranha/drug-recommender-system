from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set, Tuple

import dgl
import torch

from .evaluation import filtered_rank
from .metrics import compute_ranking_metrics


def build_training_graph(
    train_target: List[Dict[str, str]],
    non_target_triples: List[Dict[str, str]],
    context_relations: List[str] | str,
    num_nodes: int,
    device: torch.device,
    num_relations: int,
    add_inverse_edges: bool = False,
) -> Tuple[dgl.DGLGraph, torch.Tensor]:
    """Build the DGL graph used by encoder-based models."""
    if context_relations == "ALL":
        graph_rows = list(train_target) + list(non_target_triples)
    else:
        graph_rows = list(train_target)
        allowed_context_relations = set(context_relations)
        for row in non_target_triples:
            relation = (row.get("rel") or "").strip()
            if relation in allowed_context_relations:
                graph_rows.append(row)

    src_ids: List[int] = []
    dst_ids: List[int] = []
    rel_ids: List[int] = []

    for row in graph_rows:
        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        rel_id = int(row["rel_id"])

        # Original edge: src --rel--> dst
        src_ids.append(src_id)
        dst_ids.append(dst_id)
        rel_ids.append(rel_id)

        # Inverse edge: dst --rel_inverse--> src
        if add_inverse_edges:
            src_ids.append(dst_id)
            dst_ids.append(src_id)
            rel_ids.append(rel_id + num_relations)

    graph = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes)
    edge_types = torch.tensor(rel_ids, dtype=torch.long, device=device)

    return graph.to(device), edge_types


def build_candidate_pool(
    entity_types: Dict[str, str],
    entity2id: Dict[str, int],
    candidate_entity_type: str,
    device: torch.device,
    required_entity_ids: Iterable[int] | None = None,
) -> torch.Tensor:
    """Build the candidate entity pool for ranking.

    The pool starts with the configured entity type and is expanded with any
    required ground-truth ids so evaluation never drops the correct answer.
    """
    if candidate_entity_type == "ALL":
        candidate_ids = set(entity2id.values())
    else:
        candidate_ids = {
            entity2id[entity_id]
            for entity_id, entity_type in entity_types.items()
            if entity_type == candidate_entity_type
        }

    if required_entity_ids is not None:
        candidate_ids.update(int(entity_id) for entity_id in required_entity_ids)

    if not candidate_ids:
        raise ValueError(f"No candidate entities found for type: {candidate_entity_type}")

    return torch.tensor(sorted(candidate_ids), dtype=torch.long, device=device)


def score_candidates(
    task_name: str,
    decoder,
    node_embeddings: torch.Tensor,
    relation_id: int,
    query_id: int,
    candidate_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Score all candidate entities for a fixed query entity."""
    rel_emb = decoder.rel_emb(torch.tensor([relation_id], device=device)).repeat(len(candidate_ids), 1)

    if task_name == "disease_to_drug":
        disease_emb = node_embeddings[query_id].unsqueeze(0).repeat(len(candidate_ids), 1)
        drug_emb = node_embeddings[candidate_ids]
        return torch.sum(drug_emb * rel_emb * disease_emb, dim=1)

    if task_name == "drug_to_disease":
        drug_emb = node_embeddings[query_id].unsqueeze(0).repeat(len(candidate_ids), 1)
        disease_emb = node_embeddings[candidate_ids]
        return torch.sum(drug_emb * rel_emb * disease_emb, dim=1)

    if task_name == "tail_prediction":
        src_emb = node_embeddings[query_id].unsqueeze(0).repeat(len(candidate_ids), 1)
        dst_emb = node_embeddings[candidate_ids]
        return torch.sum(src_emb * rel_emb * dst_emb, dim=1)

    raise ValueError(f"Unsupported task name: {task_name}")


def evaluate_ranking(
    encoder,
    decoder,
    graph: dgl.DGLGraph,
    edge_types: torch.Tensor,
    eval_triples: List[Dict[str, str]],
    relation_id: int,
    task_name: str,
    known_positives: Dict[Any, Set[int]],
    candidate_ids: torch.Tensor,
    device: torch.device,
    hits_k: Tuple[int, ...] = (1, 3, 10),
    direction: str = "tail",
) -> Dict[str, float]:
    """
    Evaluate filtered ranking metrics for a configured task.

    direction: one of 'tail', 'head', or 'both'. When 'both' is used the
    function computes metrics for tail and head prediction and returns
    the average of each metric.
    """
    if direction not in ("tail", "head", "both"):
        raise ValueError("direction must be one of: 'tail', 'head', 'both'")

    encoder.eval()
    decoder.eval()

    node_embeddings = encoder(graph, edge_types)
    candidate_list = candidate_ids.detach().cpu().tolist()

    from collections import defaultdict

    def _reverse_known(known: Dict[Any, Set[int]]) -> Dict[Any, Set[int]]:
        rev = defaultdict(set)
        for key, vals in known.items():
            if isinstance(key, tuple) and len(key) == 2:
                q, r = key
                for c in vals:
                    rev[(c, r)].add(q)
            else:
                for c in vals:
                    rev[c].add(key)
        return rev

    def _compute_ranks_for(dir_mode: str) -> List[float]:
        kp = known_positives
        if dir_mode == "head":
            kp = _reverse_known(known_positives)

        ranks: List[float] = []

        for row in eval_triples:
            src_id = int(row["src_id"])
            dst_id = int(row["dst_id"])
            current_relation_id = relation_id

            if task_name == "disease_to_drug":
                query_id = dst_id
                true_candidate_id = src_id
                query_key = query_id
            elif task_name == "drug_to_disease":
                query_id = src_id
                true_candidate_id = dst_id
                query_key = query_id
            elif task_name == "tail_prediction":
                if dir_mode == "tail":
                    query_id = src_id
                    true_candidate_id = dst_id
                else:
                    query_id = dst_id
                    true_candidate_id = src_id

                current_relation_id = int(row["rel_id"])
                query_key = (query_id, current_relation_id)
            else:
                raise ValueError(f"Unsupported task name: {task_name}")

            scores = score_candidates(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                relation_id=current_relation_id,
                query_id=query_id,
                candidate_ids=candidate_ids,
                device=device,
            )

            rank = filtered_rank(
                candidate_ids=candidate_list,
                scores=scores.detach().cpu().tolist(),
                true_candidate_id=true_candidate_id,
                known_positive_ids=kp.get(query_key, set()),
            )
            ranks.append(rank)

        return ranks

    if direction == "both":
        tail_metrics = compute_ranking_metrics(_compute_ranks_for("tail"), hits_k)
        head_metrics = compute_ranking_metrics(_compute_ranks_for("head"), hits_k)

        avg_metrics: Dict[str, float] = {}
        for k in hits_k:
            avg_metrics[f"hits@{k}"] = 0.5 * (tail_metrics[f"hits@{k}"] + head_metrics[f"hits@{k}"])
        avg_metrics["mrr"] = 0.5 * (tail_metrics["mrr"] + head_metrics["mrr"])
        return avg_metrics

    return compute_ranking_metrics(_compute_ranks_for(direction), hits_k)

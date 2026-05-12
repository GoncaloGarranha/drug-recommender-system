from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import dgl
import torch

from .evaluation import filtered_rank
from .metrics import compute_ranking_metrics

TAIL_DIRECTION = "tail"
HEAD_DIRECTION = "head"
LINK_PREDICTION_TASKS = {"tail_prediction", "link_prediction"}


def normalize_prediction_directions(raw: Any, default: Sequence[str] = (TAIL_DIRECTION,)) -> Tuple[str, ...]:
    """Normalize benchmark prediction directions from config values."""
    if raw is None:
        values = list(default)
    elif isinstance(raw, str):
        values = [HEAD_DIRECTION, TAIL_DIRECTION] if raw == "both" else [raw]
    else:
        values = list(raw)

    normalized: List[str] = []
    for value in values:
        direction = str(value).strip().lower()
        if direction == "both":
            for item in (HEAD_DIRECTION, TAIL_DIRECTION):
                if item not in normalized:
                    normalized.append(item)
            continue
        if direction not in {HEAD_DIRECTION, TAIL_DIRECTION}:
            raise ValueError(f"Unsupported prediction direction: {value}")
        if direction not in normalized:
            normalized.append(direction)

    if not normalized:
        raise ValueError("At least one prediction direction must be configured.")
    return tuple(normalized)


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

        src_ids.append(src_id)
        dst_ids.append(dst_id)
        rel_ids.append(rel_id)

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
    """Build the candidate entity pool for ranking."""
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
    prediction_direction: str = TAIL_DIRECTION,
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

    if task_name in LINK_PREDICTION_TASKS:
        if prediction_direction == TAIL_DIRECTION:
            src_emb = node_embeddings[query_id].unsqueeze(0).repeat(len(candidate_ids), 1)
            dst_emb = node_embeddings[candidate_ids]
            return torch.sum(src_emb * rel_emb * dst_emb, dim=1)
        if prediction_direction == HEAD_DIRECTION:
            src_emb = node_embeddings[candidate_ids]
            dst_emb = node_embeddings[query_id].unsqueeze(0).repeat(len(candidate_ids), 1)
            return torch.sum(src_emb * rel_emb * dst_emb, dim=1)
        raise ValueError(f"Unsupported prediction direction: {prediction_direction}")

    raise ValueError(f"Unsupported task name: {task_name}")


def _link_positive_key(direction: str, src_id: int, rel_id: int, dst_id: int) -> Tuple[Any, ...]:
    if direction == TAIL_DIRECTION:
        return (TAIL_DIRECTION, src_id, rel_id)
    if direction == HEAD_DIRECTION:
        return (HEAD_DIRECTION, rel_id, dst_id)
    raise ValueError(f"Unsupported prediction direction: {direction}")


def evaluate_ranking(
    encoder,
    decoder,
    graph: dgl.DGLGraph,
    edge_types: torch.Tensor,
    eval_triples: List[Dict[str, str]],
    relation_id: int | None,
    task_name: str,
    known_positives: Dict[Any, Set[int]],
    candidate_ids: torch.Tensor,
    device: torch.device,
    hits_k: Tuple[int, ...] = (1, 3, 10),
    prediction_directions: Sequence[str] = (TAIL_DIRECTION,),
) -> Dict[str, float]:
    """Evaluate filtered ranking metrics for a configured task."""
    encoder.eval()
    decoder.eval()

    node_embeddings = encoder(graph, edge_types)
    ranks: List[float] = []
    direction_ranks: Dict[str, List[float]] = {direction: [] for direction in prediction_directions}
    candidate_list = candidate_ids.detach().cpu().tolist()

    for row in eval_triples:
        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        current_relation_id = int(row["rel_id"]) if relation_id is None else int(relation_id)

        if task_name == "disease_to_drug":
            scores = score_candidates(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                relation_id=current_relation_id,
                query_id=dst_id,
                candidate_ids=candidate_ids,
                device=device,
            )
            rank = filtered_rank(
                candidate_ids=candidate_list,
                scores=scores.detach().cpu().tolist(),
                true_candidate_id=src_id,
                known_positive_ids=known_positives.get(dst_id, set()),
            )
            ranks.append(rank)
            continue

        if task_name == "drug_to_disease":
            scores = score_candidates(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                relation_id=current_relation_id,
                query_id=src_id,
                candidate_ids=candidate_ids,
                device=device,
            )
            rank = filtered_rank(
                candidate_ids=candidate_list,
                scores=scores.detach().cpu().tolist(),
                true_candidate_id=dst_id,
                known_positive_ids=known_positives.get(src_id, set()),
            )
            ranks.append(rank)
            continue

        if task_name not in LINK_PREDICTION_TASKS:
            raise ValueError(f"Unsupported task name: {task_name}")

        for direction in prediction_directions:
            if direction == TAIL_DIRECTION:
                query_id = src_id
                true_candidate_id = dst_id
            elif direction == HEAD_DIRECTION:
                query_id = dst_id
                true_candidate_id = src_id
            else:
                raise ValueError(f"Unsupported prediction direction: {direction}")

            scores = score_candidates(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                relation_id=current_relation_id,
                query_id=query_id,
                candidate_ids=candidate_ids,
                device=device,
                prediction_direction=direction,
            )
            query_key = _link_positive_key(direction, src_id, current_relation_id, dst_id)
            rank = filtered_rank(
                candidate_ids=candidate_list,
                scores=scores.detach().cpu().tolist(),
                true_candidate_id=true_candidate_id,
                known_positive_ids=known_positives.get(query_key, set()),
            )
            ranks.append(rank)
            direction_ranks[direction].append(rank)

    if not ranks:
        raise ValueError("No evaluation ranks were computed.")

    results = compute_ranking_metrics(ranks, hits_k)
    for direction, values in direction_ranks.items():
        if not values:
            continue
        direction_metrics = compute_ranking_metrics(values, hits_k)
        for metric_name, metric_value in direction_metrics.items():
            results[f"{direction}_{metric_name}"] = metric_value
    return results

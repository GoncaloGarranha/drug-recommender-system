"""
Negative-set evaluation for R-GCN Disease->Drug link prediction.

This script complements the main filtered-ranking evaluation by evaluating each
positive test/validation triple against a fixed-size set of sampled negatives.
It is useful for comparison with works that report results over 1 positive + N
negative candidates instead of ranking over the full candidate pool.

Example:
    python -m models.rgcn.src.negative_set_evaluation \
      --run_dir results/runs/rgcn/biomedical/disease_to_drug/with_dgidb/seed_42 \
      --split test \
      --num_negatives 50 \
      --negative_mode mixed \
      --hard_negative_ratio 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from models.shared.graph_artifacts import load_prepared_graph, resolve_processed_dir
from models.shared.ranking import build_candidate_pool, build_training_graph
from models.shared.runs import find_latest_run
from models.shared.splits import build_known_positives, get_or_create_splits, load_splits

from .components import DistMultDecoder, RGCNEncoder
from .train import build_drug_ingredient_hard_negative_index, negative_sampling
from .utils.seed import set_seed


def score_batch(
    task_name: str,
    decoder: DistMultDecoder,
    node_embeddings: torch.Tensor,
    queries: torch.Tensor,
    relations: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """Score triples using the same DistMult direction conventions as training."""
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


def triples_to_query_relation_candidate(
    triples: Sequence[Dict[str, str]],
    task_name: str,
    relation_id: int | None,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert split triples into query/relation/candidate tensors."""
    queries: List[int] = []
    relations: List[int] = []
    candidates: List[int] = []

    for row in triples:
        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        rel_id = int(row.get("rel_id", relation_id if relation_id is not None else -1))

        if task_name == "disease_to_drug":
            queries.append(src_id)
            candidates.append(dst_id)
            relations.append(rel_id)
        elif task_name == "drug_to_disease":
            queries.append(dst_id)
            candidates.append(src_id)
            relations.append(rel_id)
        elif task_name == "tail_prediction":
            queries.append(src_id)
            candidates.append(dst_id)
            relations.append(rel_id)
        else:
            raise ValueError(f"Unsupported task name: {task_name}")

    return (
        torch.tensor(queries, dtype=torch.long, device=device),
        torch.tensor(relations, dtype=torch.long, device=device),
        torch.tensor(candidates, dtype=torch.long, device=device),
    )


def average_precision_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute average precision for binary labels without requiring sklearn."""
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    ranks = np.arange(1, len(sorted_labels) + 1)
    precision_at_k = tp / ranks
    return float(precision_at_k[sorted_labels == 1].sum() / n_pos)


def roc_auc_binary(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC-AUC using average ranks, including tie handling."""
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)

    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1

        # Ranks are 1-based. Tied values receive their average rank.
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_pos_ranks = float(ranks[labels == 1].sum())
    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="results/runs/rgcn")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument("--num_negatives", type=int, default=50)
    parser.add_argument("--negative_mode", default="random", choices=["random", "mixed"])
    parser.add_argument("--hard_negative_ratio", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else find_latest_run(args.runs_dir)
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]

    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(cfg["training"].get("device", "cpu"))

    task_cfg = cfg["task"]
    task_name = task_cfg["name"]
    target_relation = task_cfg["target_relation"]
    candidate_entity_type = task_cfg["candidate_entity_type"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    if args.negative_mode == "mixed" and task_name != "disease_to_drug":
        raise ValueError("mixed hard negatives are currently implemented only for disease_to_drug.")

    processed_dir = resolve_processed_dir(cfg["data"]["processed_dir"], task_name=task_name)
    prepared = load_prepared_graph(processed_dir)

    triples = prepared["triples"]
    entity2id = prepared["entity2id"]
    rel2id = prepared["rel2id"]
    entity_types = prepared["entity_types"]

    num_nodes = len(entity2id)
    num_relations = len(rel2id)

    model_cfg = cfg.get("model", {})
    add_inverse_edges = bool(model_cfg.get("add_inverse_edges", False))
    encoder_num_relations = num_relations * 2 if add_inverse_edges else num_relations

    relation_id = rel2id.get(target_relation) if task_name != "tail_prediction" else None
    if task_name != "tail_prediction" and relation_id is None:
        raise ValueError(f"Target relation not found: {target_relation}")

    run_splits_dir = Path(run_dir) / "splits"
    global_splits_dir = processed_dir / "splits"

    if run_splits_dir.exists():
        print(f"Using splits saved in run: {run_splits_dir}")
        train_target, valid_target, test_target = load_splits(run_splits_dir)
        non_target_triples = [
            row for row in triples
            if (row.get("rel") or "").strip() != (target_relation or "")
        ]
    else:
        print(f"Using global splits: {global_splits_dir}")
        split_cfg = cfg["split"]
        train_target, valid_target, test_target, non_target_triples = get_or_create_splits(
            output_dir=global_splits_dir,
            triples=triples,
            target_relation=target_relation,
            seed=seed,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            split_mode=split_cfg.get("mode") if isinstance(split_cfg, dict) else None,
        )

    eval_triples = valid_target if args.split == "valid" else test_target

    graph, edge_types = build_training_graph(
        train_target=train_target,
        non_target_triples=non_target_triples,
        context_relations=context_relations,
        num_nodes=num_nodes,
        device=device,
        num_relations=num_relations,
        add_inverse_edges=add_inverse_edges,
    )

    required_candidate_ids = {
        int(row["src_id"]) if task_name == "drug_to_disease" else int(row["dst_id"])
        for row in train_target + valid_target + test_target
    }

    candidate_pool = build_candidate_pool(
        entity_types=entity_types,
        entity2id=entity2id,
        candidate_entity_type=candidate_entity_type,
        device=device,
        required_entity_ids=required_candidate_ids,
    )

    known_positives = build_known_positives(
        triples=train_target + valid_target + test_target,
        task_name=task_name if task_name != "tail_prediction" else None,
        query_key="src_id" if task_name == "tail_prediction" else None,
        relation_key="rel_id" if task_name == "tail_prediction" else None,
        candidate_key="dst_id" if task_name == "tail_prediction" else None,
    )

    hard_negative_index = None
    if args.negative_mode == "mixed":
        ingredient_triples = [
            row for row in triples
            if (row.get("rel") or "").strip() == "HAS_INGREDIENT"
        ]
        hard_negative_index = build_drug_ingredient_hard_negative_index(ingredient_triples)
        print(
            "Hard negatives enabled | "
            f"source=HAS_INGREDIENT | indexed_drugs={len(hard_negative_index)} | "
            f"ratio={args.hard_negative_ratio:.2f}"
        )

    encoder = RGCNEncoder(
        num_nodes,
        cfg["training"]["emb_dim"],
        encoder_num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
        use_layer_norm=cfg.get("model", {}).get("use_layer_norm", True),
        use_residual=cfg.get("model", {}).get("use_residual", True),
        use_relation_gates=cfg.get("model", {}).get("use_relation_gates", False),
    ).to(device)

    decoder = DistMultDecoder(
        num_relations,
        cfg["training"]["emb_dim"],
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    pos_queries, pos_relations, pos_candidates = triples_to_query_relation_candidate(
        eval_triples,
        task_name=task_name,
        relation_id=relation_id,
        device=device,
    )

    all_scores: List[float] = []
    all_labels: List[int] = []
    reciprocal_ranks: List[float] = []
    hits_at_1: List[float] = []
    hits_at_10: List[float] = []
    hits_at_50: List[float] = []
    detailed_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        node_embeddings = encoder(graph, edge_types)

        for start in range(0, len(eval_triples), args.batch_size):
            end = min(start + args.batch_size, len(eval_triples))

            batch_queries = pos_queries[start:end]
            batch_relations = pos_relations[start:end]
            batch_candidates = pos_candidates[start:end]

            pos_scores = score_batch(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                queries=batch_queries,
                relations=batch_relations,
                candidates=batch_candidates,
            )

            neg_queries, neg_relations, neg_candidates = negative_sampling(
                positive_queries=batch_queries,
                positive_relations=batch_relations,
                positive_candidates=batch_candidates,
                known_positives=known_positives,
                task_name=task_name,
                k=args.num_negatives,
                candidate_pool=candidate_pool,
                device=device,
                strategy=args.negative_mode,
                hard_negative_ratio=args.hard_negative_ratio,
                hard_negative_index=hard_negative_index,
            )

            neg_scores_flat = score_batch(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                queries=neg_queries,
                relations=neg_relations,
                candidates=neg_candidates,
            )
            neg_scores = neg_scores_flat.view(end - start, args.num_negatives)
            neg_candidates_matrix = neg_candidates.view(end - start, args.num_negatives)

            for local_idx in range(end - start):
                global_idx = start + local_idx
                pos_score = float(pos_scores[local_idx].item())
                current_neg_scores = neg_scores[local_idx].detach().cpu().numpy().astype(float)

                # Conservative tie handling: tied negatives are counted above the positive.
                rank = 1 + int(np.sum(current_neg_scores >= pos_score))
                rr = 1.0 / rank

                reciprocal_ranks.append(rr)
                hits_at_1.append(1.0 if rank <= 1 else 0.0)
                hits_at_10.append(1.0 if rank <= 10 else 0.0)
                hits_at_50.append(1.0 if rank <= 50 else 0.0)

                all_scores.append(pos_score)
                all_labels.append(1)
                all_scores.extend(current_neg_scores.tolist())
                all_labels.extend([0] * args.num_negatives)

                row = eval_triples[global_idx]
                detailed_rows.append({
                    "split": args.split,
                    "index": global_idx,
                    "src_id": int(row["src_id"]),
                    "rel": row.get("rel", target_relation),
                    "rel_id": int(row.get("rel_id", relation_id if relation_id is not None else -1)),
                    "dst_id": int(row["dst_id"]),
                    "query_id": int(batch_queries[local_idx].item()),
                    "positive_candidate_id": int(batch_candidates[local_idx].item()),
                    "positive_score": pos_score,
                    "rank_among_1_plus_negatives": rank,
                    "reciprocal_rank": rr,
                    "negative_candidate_ids": "|".join(str(int(x)) for x in neg_candidates_matrix[local_idx].detach().cpu().tolist()),
                    "negative_scores": "|".join(f"{float(x):.6f}" for x in current_neg_scores.tolist()),
                })

    labels_np = np.asarray(all_labels, dtype=np.int64)
    scores_np = np.asarray(all_scores, dtype=np.float64)

    results = {
        "split": args.split,
        "task": task_name,
        "target_relation": target_relation,
        "negative_mode": args.negative_mode,
        "num_negatives": int(args.num_negatives),
        "hard_negative_ratio": float(args.hard_negative_ratio if args.negative_mode == "mixed" else 0.0),
        "num_positives": int(len(eval_triples)),
        "num_scored_pairs": int(len(all_scores)),
        "auc_roc": roc_auc_binary(labels_np, scores_np),
        "average_precision": average_precision_binary(labels_np, scores_np),
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float("nan"),
        "hits@1": float(np.mean(hits_at_1)) if hits_at_1 else float("nan"),
        "hits@10": float(np.mean(hits_at_10)) if hits_at_10 else float("nan"),
        "hits@50": float(np.mean(hits_at_50)) if hits_at_50 else float("nan"),
    }

    output_dir = Path(run_dir) / "negative_set_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.split}_{args.negative_mode}_neg{args.num_negatives}"
    if args.negative_mode == "mixed":
        suffix += f"_hard{args.hard_negative_ratio:.2f}".replace(".", "p")

    metrics_path = output_dir / f"metrics_{suffix}.json"
    details_path = output_dir / f"predictions_{suffix}.csv"

    with open(metrics_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

    with open(details_path, "w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(detailed_rows[0].keys()) if detailed_rows else [])
        if detailed_rows:
            writer.writeheader()
            writer.writerows(detailed_rows)

    print("-" * 60)
    print(f"RUN: {run_dir}")
    print(f"TASK: {task_name}")
    print(f"SPLIT: {args.split}")
    print(f"NEGATIVES: {args.num_negatives} | MODE: {args.negative_mode}")
    if args.negative_mode == "mixed":
        print(f"HARD NEGATIVE RATIO: {args.hard_negative_ratio:.2f}")
    print("-" * 60)
    print(f"AUC-ROC: {results['auc_roc']:.4f}")
    print(f"Average Precision: {results['average_precision']:.4f}")
    print(f"MRR: {results['mrr']:.4f}")
    print(f"Hits@1: {results['hits@1']:.4f}")
    print(f"Hits@10: {results['hits@10']:.4f}")
    print(f"Hits@50: {results['hits@50']:.4f}")
    print("-" * 60)
    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved detailed predictions to: {details_path}")


if __name__ == "__main__":
    main()

"""
Evaluate DistMult on test set and log results to benchmarks_schlichtkrull.csv.

Computes both raw and filtered ranking metrics, consolidates to unified CSV format.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from models.shared.config import load_config
from models.shared.io import ensure_dir, load_json
from models.shared.metrics import compute_ranking_metrics
from models.shared.runs import timestamp
from models.rgcn.src.utils.seed import set_seed

from .data import HEAD, TAIL, known_positives, load_link_prediction_data, triples_to_tensors
from .distmult_model import DistMultStandalone


def evaluate_distmult(
    model: DistMultStandalone,
    triples: list[dict[str, str]],
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: torch.Tensor,
    hits_k: tuple[int, ...] = (1, 3, 10),
    device: torch.device = torch.device("cpu"),
    mode: str = "filtered",
    max_eval_triples: int | None = None,
) -> Dict[str, float]:
    """
    Evaluate DistMult using raw or filtered ranking.
    
    Args:
        model: trained DistMultStandalone
        triples: list of test triples with 'src_id', 'rel_id', 'dst_id'
        known: known positives dict for filtered ranking
        candidate_ids: all entity IDs
        hits_k: tuple of k values for Hits@k
        device: torch device
        mode: 'raw' or 'filtered'
        max_eval_triples: limit for quick evaluation
    
    Returns:
        metrics dict with 'mrr', 'hits@1', 'hits@3', 'hits@10', plus head/tail variants
    """
    model.eval()
    candidate_list = candidate_ids.detach().cpu().tolist()
    all_ranks: list[float] = []
    head_ranks: list[float] = []
    tail_ranks: list[float] = []

    eval_rows = triples[:max_eval_triples] if max_eval_triples else triples

    with torch.no_grad():
        for row in eval_rows:
            h = int(row["src_id"])
            r = int(row["rel_id"])
            t = int(row["dst_id"])

            # Tail prediction: score all candidates for (h, r, ?)
            tail_scores = model.score_all_tails(
                torch.tensor(h, dtype=torch.long, device=device),
                torch.tensor(r, dtype=torch.long, device=device),
                candidate_ids,
            ).detach().cpu().numpy()

            if mode == "filtered":
                # Remove known positives (except true target)
                for idx, candidate in enumerate(candidate_list):
                    if candidate != t and candidate in known.get((TAIL, h, r), set()):
                        tail_scores[idx] = -1e9

            true_score = tail_scores[candidate_list.index(t)]
            tail_rank = 1 + int(np.sum(tail_scores > true_score)) + (int(np.sum(tail_scores == true_score)) - 1) / 2.0
            tail_ranks.append(tail_rank)
            all_ranks.append(tail_rank)

            # Head prediction: score all candidates for (?, r, t)
            head_scores = model.score_all_heads(
                torch.tensor(r, dtype=torch.long, device=device),
                torch.tensor(t, dtype=torch.long, device=device),
                candidate_ids,
            ).detach().cpu().numpy()

            if mode == "filtered":
                # Remove known positives (except true target)
                for idx, candidate in enumerate(candidate_list):
                    if candidate != h and candidate in known.get((HEAD, r, t), set()):
                        head_scores[idx] = -1e9

            true_score = head_scores[candidate_list.index(h)]
            head_rank = 1 + int(np.sum(head_scores > true_score)) + (int(np.sum(head_scores == true_score)) - 1) / 2.0
            head_ranks.append(head_rank)
            all_ranks.append(head_rank)

    metrics = compute_ranking_metrics(all_ranks, hits_k)
    for prefix, ranks in (("head", head_ranks), ("tail", tail_ranks)):
        direction_metrics = compute_ranking_metrics(ranks, hits_k)
        for key, value in direction_metrics.items():
            metrics[f"{prefix}_{key}"] = value

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DistMult and log to CSV")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--output_csv", default="benchmarks_schlichtkrull.csv", 
                        help="Output CSV path")
    parser.add_argument("--dataset", required=True, choices=["wn18", "fb15k", "wn18rr", "fb15k-237"],
                        help="Dataset name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used in training")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(cfg["training"].get("device", "cpu"))

    # Load data
    data = load_link_prediction_data(cfg)
    entity2id = data["entity2id"]
    rel2id = data["rel2id"]
    test = data["test"]

    num_entities = len(entity2id)
    num_relations = len(rel2id)

    known = known_positives(data.get("train", []) + data.get("valid", []) + test)
    candidate_ids = torch.tensor(sorted(entity2id.values()), dtype=torch.long, device=device)

    # Initialize model and load checkpoint
    training_cfg = cfg["training"]
    model = DistMultStandalone(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=int(training_cfg["emb_dim"]),
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    hits_k = tuple(int(k) for k in training_cfg.get("hits_k", [1, 3, 10]))

    # Evaluate both raw and filtered
    raw_metrics = evaluate_distmult(
        model, test, known, candidate_ids, hits_k, device,
        mode="raw", max_eval_triples=None
    )
    filtered_metrics = evaluate_distmult(
        model, test, known, candidate_ids, hits_k, device,
        mode="filtered", max_eval_triples=None
    )

    print(f"\n{'='*60}")
    print(f"DistMult Evaluation | {args.dataset} | seed={args.seed}")
    print(f"{'='*60}")
    print(f"\nRAW Ranking:")
    print(f"  MRR: {raw_metrics['mrr']:.4f}")
    print(f"  Hits@1: {raw_metrics.get('hits@1', 0.0):.4f}")
    print(f"  Hits@3: {raw_metrics.get('hits@3', 0.0):.4f}")
    print(f"  Hits@10: {raw_metrics.get('hits@10', 0.0):.4f}")

    print(f"\nFILTERED Ranking:")
    print(f"  MRR: {filtered_metrics['mrr']:.4f}")
    print(f"  Hits@1: {filtered_metrics.get('hits@1', 0.0):.4f}")
    print(f"  Hits@3: {filtered_metrics.get('hits@3', 0.0):.4f}")
    print(f"  Hits@10: {filtered_metrics.get('hits@10', 0.0):.4f}\n")

    # Prepare CSV row
    csv_row = {
        "dataset": args.dataset,
        "model": "DistMult",
        "seed": args.seed,
        "split": "test",
        "raw_mrr": round(raw_metrics["mrr"], 4),
        "raw_hits@1": round(raw_metrics.get("hits@1", 0.0), 4),
        "raw_hits@3": round(raw_metrics.get("hits@3", 0.0), 4),
        "raw_hits@10": round(raw_metrics.get("hits@10", 0.0), 4),
        "filtered_mrr": round(filtered_metrics["mrr"], 4),
        "filtered_hits@1": round(filtered_metrics.get("hits@1", 0.0), 4),
        "filtered_hits@3": round(filtered_metrics.get("hits@3", 0.0), 4),
        "filtered_hits@10": round(filtered_metrics.get("hits@10", 0.0), 4),
    }

    # Append to CSV
    output_path = Path(args.output_csv)
    file_exists = output_path.exists()

    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_row)

    print(f"✓ Results appended to {output_path}")


if __name__ == "__main__":
    main()

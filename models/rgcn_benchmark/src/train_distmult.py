"""
DistMult standalone training and evaluation.

Implements:
- Negative sampling (head/tail corruption)
- Raw ranking (all entities as candidates)
- Filtered ranking (remove known positives)
- Metrics: MRR, Hits@K for both modes
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.shared.config import load_config, save_config
from models.shared.io import ensure_dir, save_json
from models.shared.metrics import compute_ranking_metrics
from models.shared.runs import organize_legacy_runs, resolve_run_dir
from models.rgcn.src.utils.seed import set_seed

from .data import HEAD, TAIL, known_positives, load_link_prediction_data, triples_to_tensors
from .distmult_model import DistMultStandalone


def sample_negatives(
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: np.ndarray,
    neg_k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample negative examples by corrupting head or tail.
    
    For each positive triple, sample exactly one negative example:
    either a corrupted head or a corrupted tail, chosen uniformly.
    Forbidden negatives: known positives involving the same (head, rel) or (rel, tail).
    """
    rng = np.random.default_rng()
    neg_heads = []
    neg_rels = []
    neg_tails = []

    for h_raw, r_raw, t_raw in zip(heads.detach().cpu().tolist(), rels.detach().cpu().tolist(), tails.detach().cpu().tolist()):
        h = int(h_raw)
        r = int(r_raw)
        t = int(t_raw)

        corrupt_head = bool(rng.integers(0, 2))
        if corrupt_head:
            forbidden_heads = set(known.get((HEAD, r, t), set()))
            forbidden_heads.add(h)
            candidate = int(rng.choice(candidate_ids))
            while candidate in forbidden_heads:
                candidate = int(rng.choice(candidate_ids))
            neg_heads.append(candidate)
            neg_rels.append(r)
            neg_tails.append(t)
        else:
            forbidden_tails = set(known.get((TAIL, h, r), set()))
            forbidden_tails.add(t)
            candidate = int(rng.choice(candidate_ids))
            while candidate in forbidden_tails:
                candidate = int(rng.choice(candidate_ids))
            neg_heads.append(h)
            neg_rels.append(r)
            neg_tails.append(candidate)

    return (
        torch.tensor(neg_heads, dtype=torch.long, device=device),
        torch.tensor(neg_rels, dtype=torch.long, device=device),
        torch.tensor(neg_tails, dtype=torch.long, device=device),
        torch.empty(0, dtype=torch.long, device=device),
        torch.empty(0, dtype=torch.long, device=device),
        torch.empty(0, dtype=torch.long, device=device),
    )


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
    parser = argparse.ArgumentParser(description="Train DistMult standalone model")
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--eval_mode", default="filtered", choices=["raw", "filtered"],
                        help="Evaluation mode: raw (all entities) or filtered (remove known)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(cfg["training"].get("device", "cpu"))

    # Load data
    data = load_link_prediction_data(cfg)
    entity2id = data["entity2id"]
    rel2id = data["rel2id"]
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

    num_entities = len(entity2id)
    num_relations = len(rel2id)

    train_heads, train_rels, train_tails = triples_to_tensors(train, device)
    known = known_positives(train + valid + test)
    candidate_ids = torch.tensor(sorted(entity2id.values()), dtype=torch.long, device=device)
    candidate_array = candidate_ids.detach().cpu().numpy()

    # Initialize model
    training_cfg = cfg["training"]
    model = DistMultStandalone(
        num_entities=num_entities,
        num_relations=num_relations,
        emb_dim=int(training_cfg["emb_dim"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )

    organize_legacy_runs(cfg["runs"]["out_dir"], default_model_name="distmult", seed=seed)
    run_dir = resolve_run_dir(cfg["runs"]["out_dir"], cfg, model_name="distmult", seed=seed)
    checkpoints_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(checkpoints_dir)
    save_config(run_dir / "config.yaml", cfg)
    save_json(
        run_dir / "split_summary.json",
        {
            "train_size": len(train),
            "valid_size": len(valid),
            "test_size": len(test),
            "num_entities": num_entities,
            "num_relations": num_relations,
            "eval_mode": args.eval_mode,
        },
    )

    max_epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 100)))
    eval_every = int(training_cfg.get("eval_every", 10))
    neg_k = int(training_cfg.get("neg_k", 1))
    hits_k = tuple(int(k) for k in training_cfg.get("hits_k", [1, 3, 10]))
    max_eval_triples = training_cfg.get("max_eval_triples")
    max_eval_triples = int(max_eval_triples) if max_eval_triples else None

    best_mrr = -1.0
    best_epoch = -1
    train_log: list[dict[str, float | int | None]] = []

    print(
        f"DistMult Standalone | entities={num_entities} relations={num_relations} "
        f"train={len(train)} valid={len(valid)} eval_mode={args.eval_mode}"
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        positive_scores = model(train_heads, train_rels, train_tails)

        # Sample negatives
        neg_h, neg_r, neg_t, _, _, _ = sample_negatives(
            train_heads, train_rels, train_tails, known, candidate_array, neg_k, device
        )

        negative_scores = model(neg_h, neg_r, neg_t)

        # Combined loss
        scores = torch.cat([positive_scores, negative_scores])
        labels = torch.cat([
            torch.ones_like(positive_scores),
            torch.zeros_like(negative_scores),
        ])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_entry: dict[str, float | int | None] = {"epoch": epoch, "loss": float(loss.item()), "valid_mrr": None}

        if epoch % eval_every == 0 or epoch == max_epochs:
            metrics = evaluate_distmult(
                model, valid, known, candidate_ids, hits_k, device,
                mode=args.eval_mode, max_eval_triples=max_eval_triples
            )
            log_entry["valid_mrr"] = float(metrics["mrr"])
            print(
                f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | "
                f"Valid MRR: {metrics['mrr']:.4f} | "
                f"Head MRR: {metrics.get('head_mrr', 0.0):.4f} | "
                f"Tail MRR: {metrics.get('tail_mrr', 0.0):.4f} | "
                f"Hits@10: {metrics.get('hits@10', 0.0):.4f}"
            )

            if metrics["mrr"] > best_mrr:
                best_mrr = float(metrics["mrr"])
                best_epoch = epoch
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": cfg,
                        "best_epoch": best_epoch,
                        "best_valid_mrr": best_mrr,
                    },
                    checkpoints_dir / "best.pt",
                )
                save_json(run_dir / "best_metrics.json", {"best_epoch": best_epoch, "best_valid_mrr": best_mrr, **metrics})

        train_log.append(log_entry)

    # Final evaluation on test set with best checkpoint
    checkpoint = torch.load(checkpoints_dir / "best.pt")
    model.load_state_dict(checkpoint["model"])

    test_metrics = evaluate_distmult(
        model, test, known, candidate_ids, hits_k, device,
        mode=args.eval_mode, max_eval_triples=None
    )

    print(f"\n[TEST] Best epoch {best_epoch} | {args.eval_mode.upper()} ranking:")
    print(f"  MRR: {test_metrics['mrr']:.4f}")
    print(f"  Hits@1: {test_metrics.get('hits@1', 0.0):.4f}")
    print(f"  Hits@3: {test_metrics.get('hits@3', 0.0):.4f}")
    print(f"  Hits@10: {test_metrics.get('hits@10', 0.0):.4f}")
    print(f"  Head MRR: {test_metrics.get('head_mrr', 0.0):.4f}")
    print(f"  Tail MRR: {test_metrics.get('tail_mrr', 0.0):.4f}\n")

    torch.save(
        {"model": model.state_dict(), "config": cfg, "best_epoch": best_epoch},
        checkpoints_dir / "last.pt",
    )
    save_json(run_dir / "train_log.json", train_log)
    save_json(run_dir / f"test_metrics_{args.eval_mode}.json", test_metrics)

    print(f"Training complete. Best validation MRR: {best_mrr:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()

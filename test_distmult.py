"""
Quick test: train DistMult on WN18 with small epoch count for validation.
"""

import sys
from pathlib import Path

# Add workspace root to path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from models.shared.config import load_config, save_config
from models.shared.io import ensure_dir, save_json
from models.shared.metrics import compute_ranking_metrics
from models.shared.runs import timestamp
from models.rgcn.src.utils.seed import set_seed
from models.rgcn_schlichtkrull.src.data import HEAD, TAIL, known_positives, load_link_prediction_data, triples_to_tensors
from models.rgcn_schlichtkrull.src.distmult_model import DistMultStandalone


def sample_negatives(heads, rels, tails, known, candidate_ids, neg_k, device):
    """Sample exactly one negative example per positive triple."""
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


def evaluate_distmult(model, triples, known, candidate_ids, hits_k=(1, 3, 10), device=torch.device("cpu"), mode="filtered"):
    """Evaluate DistMult using raw or filtered ranking."""
    model.eval()
    candidate_list = candidate_ids.detach().cpu().tolist()
    all_ranks = []
    head_ranks = []
    tail_ranks = []

    with torch.no_grad():
        for row in triples:
            h = int(row["src_id"])
            r = int(row["rel_id"])
            t = int(row["dst_id"])

            # Tail prediction
            tail_scores = model.score_all_tails(
                torch.tensor(h, dtype=torch.long, device=device),
                torch.tensor(r, dtype=torch.long, device=device),
                candidate_ids,
            ).detach().cpu().numpy()

            if mode == "filtered":
                for idx, candidate in enumerate(candidate_list):
                    if candidate != t and candidate in known.get((TAIL, h, r), set()):
                        tail_scores[idx] = -1e9

            true_score = tail_scores[candidate_list.index(t)]
            tail_rank = 1 + int(np.sum(tail_scores > true_score)) + (int(np.sum(tail_scores == true_score)) - 1) / 2.0
            tail_ranks.append(tail_rank)
            all_ranks.append(tail_rank)

            # Head prediction
            head_scores = model.score_all_heads(
                torch.tensor(r, dtype=torch.long, device=device),
                torch.tensor(t, dtype=torch.long, device=device),
                candidate_ids,
            ).detach().cpu().numpy()

            if mode == "filtered":
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="models/rgcn_schlichtkrull/configs/wn18.yaml")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("DistMult Standalone - WN18 Test")
    print(f"{'='*60}\n")

    # Load config
    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(args.device)

    # Load data
    print(f"Loading data...")
    data = load_link_prediction_data(cfg)
    entity2id = data["entity2id"]
    rel2id = data["rel2id"]
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

    num_entities = len(entity2id)
    num_relations = len(rel2id)

    print(f"  Entities: {num_entities}")
    print(f"  Relations: {num_relations}")
    print(f"  Train: {len(train)} | Valid: {len(valid)} | Test: {len(test)}\n")

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

    neg_k = int(training_cfg.get("neg_k", 1))
    hits_k = tuple(int(k) for k in training_cfg.get("hits_k", [1, 3, 10]))
    eval_every = int(training_cfg.get("eval_every", 10))

    best_mrr = -1.0
    best_epoch = -1

    print(f"Training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        positive_scores = model(train_heads, train_rels, train_tails)

        # Sample negatives
        neg_h, neg_r, neg_t, _, _, _ = sample_negatives(
            train_heads, train_rels, train_tails, known, candidate_array, neg_k, device
        )

        negative_scores = model(neg_h, neg_r, neg_t)

        scores = torch.cat([positive_scores, negative_scores])
        labels = torch.cat([
            torch.ones_like(positive_scores),
            torch.zeros_like(negative_scores),
        ])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % eval_every == 0 or epoch == args.epochs:
            metrics = evaluate_distmult(model, valid, known, candidate_ids, hits_k, device, mode="filtered")
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Valid MRR: {metrics['mrr']:.4f} | Hits@10: {metrics.get('hits@10', 0.0):.4f}")

            if metrics["mrr"] > best_mrr:
                best_mrr = float(metrics["mrr"])
                best_epoch = epoch

    print(f"\n✓ Training complete. Best validation MRR: {best_mrr:.4f} (epoch {best_epoch})")

    # Final test evaluation
    test_metrics = evaluate_distmult(model, test, known, candidate_ids, hits_k, device, mode="filtered")
    print(f"\n{'='*60}")
    print(f"Test Results (Filtered Ranking)")
    print(f"{'='*60}")
    print(f"  MRR: {test_metrics['mrr']:.4f}")
    print(f"  Hits@1: {test_metrics.get('hits@1', 0.0):.4f}")
    print(f"  Hits@3: {test_metrics.get('hits@3', 0.0):.4f}")
    print(f"  Hits@10: {test_metrics.get('hits@10', 0.0):.4f}")
    print(f"  Head MRR: {test_metrics.get('head_mrr', 0.0):.4f}")
    print(f"  Tail MRR: {test_metrics.get('tail_mrr', 0.0):.4f}")
    print(f"{'='*60}\n")

    # Target comparison
    target_mrr = 0.813
    diff = test_metrics['mrr'] - target_mrr
    status = "✓ PASS" if diff > -0.01 else "✗ FAIL"
    print(f"Target (article): {target_mrr:.4f}")
    print(f"Achieved: {test_metrics['mrr']:.4f}")
    print(f"Difference: {diff:+.4f} {status}\n")


if __name__ == "__main__":
    main()

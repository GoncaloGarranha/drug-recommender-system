"""
Minimal overfit test for DistMult on a tiny WN18 subset.

Goal: verify that the standalone DistMult baseline can memorize a very small
subset of training triples. If it cannot, there is likely a bug in the pipeline
or optimization setup.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.shared.config import load_config
from models.rgcn.src.utils.seed import set_seed
from models.rgcn_schlichtkrull.src.data import HEAD, TAIL, known_positives, load_link_prediction_data, triples_to_tensors
from models.rgcn_schlichtkrull.src.distmult_model import DistMultStandalone


def sample_negative_batch(
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng()
    neg_heads = []
    neg_rels = []
    neg_tails = []

    for h_raw, r_raw, t_raw in zip(heads.tolist(), rels.tolist(), tails.tolist()):
        h = int(h_raw)
        r = int(r_raw)
        t = int(t_raw)
        if rng.integers(0, 2):
            forbidden = set(known.get((HEAD, r, t), set())) | {h}
            candidate = int(rng.choice(candidate_ids))
            while candidate in forbidden:
                candidate = int(rng.choice(candidate_ids))
            neg_heads.append(candidate)
            neg_rels.append(r)
            neg_tails.append(t)
        else:
            forbidden = set(known.get((TAIL, h, r), set())) | {t}
            candidate = int(rng.choice(candidate_ids))
            while candidate in forbidden:
                candidate = int(rng.choice(candidate_ids))
            neg_heads.append(h)
            neg_rels.append(r)
            neg_tails.append(candidate)

    return (
        torch.tensor(neg_heads, dtype=torch.long, device=device),
        torch.tensor(neg_rels, dtype=torch.long, device=device),
        torch.tensor(neg_tails, dtype=torch.long, device=device),
    )


def ranking_metrics(ranks: list[float], hits_k: tuple[int, ...] = (1, 3, 10)) -> dict[str, float]:
    arr = np.asarray(ranks, dtype=np.float64)
    return {
        "mrr": float(np.mean(1.0 / arr)),
        **{f"hits@{k}": float(np.mean(arr <= k)) for k in hits_k},
    }


def eval_on_subset(
    model: DistMultStandalone,
    triples: list[dict[str, str]],
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: torch.Tensor,
    device: torch.device,
    filtered: bool,
) -> dict[str, float]:
    model.eval()
    candidate_list = candidate_ids.detach().cpu().tolist()
    head_ranks: list[float] = []
    tail_ranks: list[float] = []

    with torch.no_grad():
        for row in triples:
            h = int(row["src_id"])
            r = int(row["rel_id"])
            t = int(row["dst_id"])

            tail_scores = model.score_all_tails(
                torch.tensor(h, dtype=torch.long, device=device),
                torch.tensor(r, dtype=torch.long, device=device),
                candidate_ids,
            ).detach().cpu().numpy()
            if filtered:
                for idx, candidate in enumerate(candidate_list):
                    if candidate != t and candidate in known.get((TAIL, h, r), set()):
                        tail_scores[idx] = -1e9
            true_tail_score = tail_scores[candidate_list.index(t)]
            tail_rank = 1 + int(np.sum(tail_scores > true_tail_score)) + (int(np.sum(tail_scores == true_tail_score)) - 1) / 2.0
            tail_ranks.append(tail_rank)

            head_scores = model.score_all_heads(
                torch.tensor(r, dtype=torch.long, device=device),
                torch.tensor(t, dtype=torch.long, device=device),
                candidate_ids,
            ).detach().cpu().numpy()
            if filtered:
                for idx, candidate in enumerate(candidate_list):
                    if candidate != h and candidate in known.get((HEAD, r, t), set()):
                        head_scores[idx] = -1e9
            true_head_score = head_scores[candidate_list.index(h)]
            head_rank = 1 + int(np.sum(head_scores > true_head_score)) + (int(np.sum(head_scores == true_head_score)) - 1) / 2.0
            head_ranks.append(head_rank)

    head_metrics = ranking_metrics(head_ranks)
    tail_metrics = ranking_metrics(tail_ranks)
    all_metrics = ranking_metrics(head_ranks + tail_ranks)
    return {
        "mrr": all_metrics["mrr"],
        "hits@1": all_metrics["hits@1"],
        "hits@3": all_metrics["hits@3"],
        "hits@10": all_metrics["hits@10"],
        "head_mrr": head_metrics["mrr"],
        "tail_mrr": tail_metrics["mrr"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit DistMult on a tiny WN18 subset")
    parser.add_argument("--config", default="models/rgcn_schlichtkrull/configs/wn18.yaml")
    parser.add_argument("--subset_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = torch.device(args.device)

    data = load_link_prediction_data(cfg)
    train = data["train"]
    entity2id = data["entity2id"]
    rel2id = data["rel2id"]

    subset = train[: args.subset_size]
    heads, rels, tails = triples_to_tensors(subset, device)
    known = known_positives(subset)
    candidate_ids = torch.tensor(sorted(entity2id.values()), dtype=torch.long, device=device)
    candidate_array = candidate_ids.detach().cpu().numpy()

    model = DistMultStandalone(len(entity2id), len(rel2id), int(cfg["training"]["emb_dim"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(f"Subset size: {len(subset)} | Entities: {len(entity2id)} | Relations: {len(rel2id)}")
    print(f"Training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        neg_h, neg_r, neg_t = sample_negative_batch(heads, rels, tails, known, candidate_array, device)

        pos_scores = model(heads, rels, tails)
        neg_scores = model(neg_h, neg_r, neg_t)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            with torch.no_grad():
                pos_mean = float(pos_scores.mean().item())
                neg_mean = float(neg_scores.mean().item())
            train_raw = eval_on_subset(model, subset, known, candidate_ids, device, filtered=False)
            train_filt = eval_on_subset(model, subset, known, candidate_ids, device, filtered=True)
            print(
                f"Epoch {epoch:04d} | loss={loss.item():.4f} | "
                f"pos_mean={pos_mean:.4f} neg_mean={neg_mean:.4f} | "
                f"raw_mrr={train_raw['mrr']:.4f} filt_mrr={train_filt['mrr']:.4f} | "
                f"hits@1={train_filt['hits@1']:.4f} hits@10={train_filt['hits@10']:.4f}"
            )

    final_raw = eval_on_subset(model, subset, known, candidate_ids, device, filtered=False)
    final_filt = eval_on_subset(model, subset, known, candidate_ids, device, filtered=True)

    print("\nFinal subset metrics:")
    print(f"  RAW      MRR: {final_raw['mrr']:.4f} | Hits@1: {final_raw['hits@1']:.4f} | Hits@10: {final_raw['hits@10']:.4f}")
    print(f"  FILTERED MRR: {final_filt['mrr']:.4f} | Hits@1: {final_filt['hits@1']:.4f} | Hits@10: {final_filt['hits@10']:.4f}")


if __name__ == "__main__":
    main()

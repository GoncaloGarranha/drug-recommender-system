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
from models.shared.runs import timestamp
from models.rgcn.src.utils.seed import set_seed

from .data import HEAD, TAIL, build_encoder_graph, known_positives, load_link_prediction_data, triples_to_tensors
from .model import DistMult, SchlichtkrullRGCN


def sample_negatives(
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: np.ndarray,
    neg_k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng()
    neg_tail_heads = []
    neg_tail_rels = []
    neg_tail_tails = []
    neg_head_heads = []
    neg_head_rels = []
    neg_head_tails = []

    for h_raw, r_raw, t_raw in zip(heads.detach().cpu().tolist(), rels.detach().cpu().tolist(), tails.detach().cpu().tolist()):
        h = int(h_raw)
        r = int(r_raw)
        t = int(t_raw)

        forbidden_tails = set(known.get((TAIL, h, r), set()))
        forbidden_tails.add(t)
        collected = 0
        while collected < neg_k:
            candidate = int(rng.choice(candidate_ids))
            if candidate in forbidden_tails:
                continue
            neg_tail_heads.append(h)
            neg_tail_rels.append(r)
            neg_tail_tails.append(candidate)
            collected += 1

        forbidden_heads = set(known.get((HEAD, r, t), set()))
        forbidden_heads.add(h)
        collected = 0
        while collected < neg_k:
            candidate = int(rng.choice(candidate_ids))
            if candidate in forbidden_heads:
                continue
            neg_head_heads.append(candidate)
            neg_head_rels.append(r)
            neg_head_tails.append(t)
            collected += 1

    return (
        torch.tensor(neg_tail_heads, dtype=torch.long, device=device),
        torch.tensor(neg_tail_rels, dtype=torch.long, device=device),
        torch.tensor(neg_tail_tails, dtype=torch.long, device=device),
        torch.tensor(neg_head_heads, dtype=torch.long, device=device),
        torch.tensor(neg_head_rels, dtype=torch.long, device=device),
        torch.tensor(neg_head_tails, dtype=torch.long, device=device),
    )


@torch.no_grad()
def filtered_eval(
    encoder: SchlichtkrullRGCN,
    decoder: DistMult,
    graph,
    edge_types: torch.Tensor,
    triples: list[dict[str, str]],
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: torch.Tensor,
    hits_k: tuple[int, ...],
    device: torch.device,
    max_eval_triples: int | None = None,
) -> Dict[str, float]:
    encoder.eval()
    decoder.eval()
    node_emb = encoder(graph, edge_types)
    candidate_list = candidate_ids.detach().cpu().tolist()
    all_ranks: list[float] = []
    head_ranks: list[float] = []
    tail_ranks: list[float] = []

    eval_rows = triples[:max_eval_triples] if max_eval_triples else triples
    for row in eval_rows:
        h = int(row["src_id"])
        r = int(row["rel_id"])
        t = int(row["dst_id"])
        rel_emb = decoder.rel_emb(torch.tensor([r], device=device)).repeat(len(candidate_ids), 1)

        # Tail prediction: (h, r, ?)
        h_emb = node_emb[h].unsqueeze(0).repeat(len(candidate_ids), 1)
        tail_scores = decoder.score(h_emb, rel_emb, node_emb[candidate_ids]).detach().cpu().numpy().copy()
        for idx, candidate in enumerate(candidate_list):
            if candidate != t and candidate in known.get((TAIL, h, r), set()):
                tail_scores[idx] = -1e9
        true_score = tail_scores[candidate_list.index(t)]
        tail_rank = 1 + int(np.sum(tail_scores > true_score)) + (int(np.sum(tail_scores == true_score)) - 1) / 2.0
        tail_ranks.append(tail_rank)
        all_ranks.append(tail_rank)

        # Head prediction: (?, r, t)
        t_emb = node_emb[t].unsqueeze(0).repeat(len(candidate_ids), 1)
        head_scores = decoder.score(node_emb[candidate_ids], rel_emb, t_emb).detach().cpu().numpy().copy()
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
    parser = argparse.ArgumentParser(description="Train Schlichtkrull-style R-GCN link prediction model")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = torch.device(cfg["training"].get("device", "cpu"))

    data = load_link_prediction_data(cfg)
    entity2id = data["entity2id"]
    rel2id = data["rel2id"]
    train = data["train"]
    valid = data["valid"]
    test = data["test"]

    num_nodes = len(entity2id)
    num_relations = len(rel2id)
    add_inverse = bool(cfg.get("model", {}).get("add_inverse_edges", True))
    encoder_relations = num_relations * 2 if add_inverse else num_relations

    graph, edge_types = build_encoder_graph(train, num_nodes, num_relations, device, add_inverse_edges=add_inverse)
    train_heads, train_rels, train_tails = triples_to_tensors(train, device)
    known = known_positives(train + valid + test)
    candidate_ids = torch.tensor(sorted(entity2id.values()), dtype=torch.long, device=device)
    candidate_array = candidate_ids.detach().cpu().numpy()

    training_cfg = cfg["training"]
    encoder = SchlichtkrullRGCN(
        num_nodes=num_nodes,
        h_dim=int(training_cfg["emb_dim"]),
        num_rels=encoder_relations,
        num_bases=int(training_cfg["num_bases"]),
        num_layers=int(training_cfg["num_layers"]),
        dropout=float(training_cfg["dropout"]),
        regularizer=cfg.get("model", {}).get("regularizer", "basis"),
    ).to(device)
    decoder = DistMult(num_relations, int(training_cfg["emb_dim"])).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    run_dir = Path(cfg["runs"]["out_dir"]) / f"{timestamp()}_schlichtkrull_rgcn"
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
            "num_nodes": num_nodes,
            "num_relations_decoder": num_relations,
            "num_relations_encoder": encoder_relations,
            "add_inverse_edges": add_inverse,
            "graph_edges": int(graph.num_edges()),
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
        f"Schlichtkrull R-GCN | nodes={num_nodes} relations={num_relations} "
        f"encoder_relations={encoder_relations} train={len(train)} valid={len(valid)}"
    )

    for epoch in range(1, max_epochs + 1):
        encoder.train()
        decoder.train()
        node_emb = encoder(graph, edge_types)
        positive_scores = decoder(node_emb, train_heads, train_rels, train_tails)
        neg_th, neg_tr, neg_tt, neg_hh, neg_hr, neg_ht = sample_negatives(
            train_heads, train_rels, train_tails, known, candidate_array, neg_k, device
        )
        negative_tail = decoder(node_emb, neg_th, neg_tr, neg_tt)
        negative_head = decoder(node_emb, neg_hh, neg_hr, neg_ht)
        negative_scores = torch.cat([negative_tail, negative_head])
        loss = F.binary_cross_entropy_with_logits(positive_scores, torch.ones_like(positive_scores)) + F.binary_cross_entropy_with_logits(
            negative_scores, torch.zeros_like(negative_scores)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_entry: dict[str, float | int | None] = {"epoch": epoch, "loss": float(loss.item()), "valid_mrr": None}
        if epoch % eval_every == 0 or epoch == max_epochs:
            metrics = filtered_eval(encoder, decoder, graph, edge_types, valid, known, candidate_ids, hits_k, device, max_eval_triples)
            log_entry["valid_mrr"] = float(metrics["mrr"])
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Valid MRR: {metrics['mrr']:.4f}")
            if metrics["mrr"] > best_mrr:
                best_mrr = float(metrics["mrr"])
                best_epoch = epoch
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "config": cfg,
                        "best_epoch": best_epoch,
                        "best_valid_mrr": best_mrr,
                    },
                    checkpoints_dir / "best.pt",
                )
                save_json(run_dir / "best_metrics.json", {"best_epoch": best_epoch, "best_valid_mrr": best_mrr, **metrics})
        train_log.append(log_entry)

    torch.save(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg, "best_epoch": best_epoch},
        checkpoints_dir / "last.pt",
    )
    save_json(run_dir / "train_log.json", train_log)
    print(f"Training complete. Best validation MRR: {best_mrr:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()

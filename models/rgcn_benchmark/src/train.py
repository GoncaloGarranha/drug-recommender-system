from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from models.shared.config import load_config, save_config
from models.shared.io import ensure_dir, save_json
from models.shared.metrics import compute_ranking_metrics
from models.shared.runs import organize_legacy_runs, resolve_run_dir
from models.rgcn.src.utils.seed import set_seed

from .data import HEAD, TAIL, build_encoder_graph, graph_num_edges, graph_num_nodes, known_positives, load_link_prediction_data, triples_to_tensors
from .model import DistMult, SchlichtkrullRGCN


def sample_negatives(
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: np.ndarray,
    neg_k: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample negatives following Schlichtkrull et al. (2018).

    For each positive triple, randomly decide whether to corrupt the head
    or the tail (50/50). Generates neg_k negatives per positive.
    Known positives are excluded from negative samples (filtered negatives).
    """
    rng = np.random.default_rng()
    neg_heads: List[int] = []
    neg_rels: List[int] = []
    neg_tails: List[int] = []

    for h_raw, r_raw, t_raw in zip(
        heads.detach().cpu().tolist(),
        rels.detach().cpu().tolist(),
        tails.detach().cpu().tolist(),
    ):
        h = int(h_raw)
        r = int(r_raw)
        t = int(t_raw)

        for _ in range(neg_k):
            corrupt_head = bool(rng.integers(0, 2))
            if corrupt_head:
                forbidden = set(known.get((HEAD, r, t), set()))
                forbidden.add(h)
                candidate = int(rng.choice(candidate_ids))
                while candidate in forbidden:
                    candidate = int(rng.choice(candidate_ids))
                neg_heads.append(candidate)
                neg_rels.append(r)
                neg_tails.append(t)
            else:
                forbidden = set(known.get((TAIL, h, r), set()))
                forbidden.add(t)
                candidate = int(rng.choice(candidate_ids))
                while candidate in forbidden:
                    candidate = int(rng.choice(candidate_ids))
                neg_heads.append(h)
                neg_rels.append(r)
                neg_tails.append(candidate)

    return (
        torch.tensor(neg_heads, dtype=torch.long, device=device),
        torch.tensor(neg_rels,  dtype=torch.long, device=device),
        torch.tensor(neg_tails, dtype=torch.long, device=device),
    )


def tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Print basic statistics for a tensor."""
    if tensor is None:
        print(f"[DEBUG] {name}: None")
        return

    detached = tensor.detach()
    if detached.numel() == 0:
        print(f"[DEBUG] {name}: empty tensor")
        return

    print(
        f"[DEBUG] {name}: "
        f"shape={tuple(detached.shape)} | "
        f"mean={detached.float().mean().item():.6f} | "
        f"std={detached.float().std().item():.6f} | "
        f"min={detached.min().item():.6f} | "
        f"max={detached.max().item():.6f}"
    )


def debug_static_data(
    train_heads: torch.Tensor,
    train_rels: torch.Tensor,
    train_tails: torch.Tensor,
    edge_types: torch.Tensor,
    graph,
    num_nodes: int,
    num_relations: int,
    encoder_relations: int,
) -> None:
    """Print static checks before training starts."""
    print("\n================ DEBUG STATIC DATA ================")
    print(f"[DEBUG] num_nodes: {num_nodes}")
    print(f"[DEBUG] decoder relations: {num_relations}")
    print(f"[DEBUG] encoder relations: {encoder_relations}")
    print(f"[DEBUG] graph nodes: {graph_num_nodes(graph)}")
    print(f"[DEBUG] graph edges: {graph_num_edges(graph)}")

    print(
        f"[DEBUG] train rel min/max: "
        f"{train_rels.min().item()} / {train_rels.max().item()}"
    )
    print(
        f"[DEBUG] train head min/max: "
        f"{train_heads.min().item()} / {train_heads.max().item()}"
    )
    print(
        f"[DEBUG] train tail min/max: "
        f"{train_tails.min().item()} / {train_tails.max().item()}"
    )
    print(
        f"[DEBUG] encoder edge_types min/max: "
        f"{edge_types.min().item()} / {edge_types.max().item()}"
    )

    if train_rels.max().item() >= num_relations:
        print("[DEBUG][ERROR] Train relation id exceeds decoder relation count.")

    if edge_types.max().item() >= encoder_relations:
        print("[DEBUG][ERROR] Edge type id exceeds encoder relation count.")

    if train_heads.max().item() >= num_nodes or train_tails.max().item() >= num_nodes:
        print("[DEBUG][ERROR] Entity id exceeds number of nodes.")

    unique_train_rels = torch.unique(train_rels).numel()
    unique_edge_types = torch.unique(edge_types).numel()

    print(f"[DEBUG] unique train relations: {unique_train_rels}")
    print(f"[DEBUG] unique encoder edge types: {unique_edge_types}")
    print("===================================================\n")


def debug_scores_and_gradients(
    epoch: int,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    node_emb: torch.Tensor,
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
) -> None:
    """Print score and gradient diagnostics during training."""
    print(f"\n================ DEBUG EPOCH {epoch} ================")

    tensor_stats("node_emb", node_emb)
    tensor_stats("positive_scores", positive_scores)
    tensor_stats("negative_scores", negative_scores)

    with torch.no_grad():
        pos_mean = positive_scores.mean().item()
        neg_mean = negative_scores.mean().item()
        margin = pos_mean - neg_mean

        print(f"[DEBUG] pos_mean - neg_mean: {margin:.6f}")

        if positive_scores.numel() > 0 and negative_scores.numel() > 0:
            sample_size = min(positive_scores.numel(), negative_scores.numel(), 10000)
            pos_sample = positive_scores[:sample_size]
            neg_sample = negative_scores[:sample_size]
            frac_pos_higher = (pos_sample > neg_sample).float().mean().item()
            print(f"[DEBUG] fraction(pos_score > neg_score) sample: {frac_pos_higher:.6f}")

    print("[DEBUG] gradient norms:")

    total_grad = 0.0
    has_grad = False

    for name, param in list(encoder.named_parameters()) + list(decoder.named_parameters()):
        if param.grad is None:
            print(f"[DEBUG] {name}: grad=None")
            continue

        grad_norm = param.grad.norm().item()
        total_grad += grad_norm
        has_grad = True
        print(f"[DEBUG] {name}: grad_norm={grad_norm:.6f}")

    if not has_grad:
        print("[DEBUG][ERROR] No gradients found.")

    print(f"[DEBUG] total_grad_norm_sum: {total_grad:.6f}")
    print("====================================================\n")


def debug_negative_samples(
    epoch: int,
    neg_hh: torch.Tensor,
    neg_hr: torch.Tensor,
    neg_ht: torch.Tensor,
    num_relations: int,
    num_nodes: int,
) -> None:
    """Validate negative samples."""
    print(f"\n============= DEBUG NEGATIVE SAMPLES EPOCH {epoch} =============")
    print(f"[DEBUG] neg triples: {len(neg_hh)}")

    if len(neg_hr) > 0:
        print(f"[DEBUG] neg rel min/max: {neg_hr.min().item()} / {neg_hr.max().item()}")
    if len(neg_hh) > 0:
        print(f"[DEBUG] neg head entity min/max: {neg_hh.min().item()} / {neg_hh.max().item()}")
    if len(neg_ht) > 0:
        print(f"[DEBUG] neg tail entity min/max: {neg_ht.min().item()} / {neg_ht.max().item()}")

    if len(neg_hr) > 0 and neg_hr.max().item() >= num_relations:
        print("[DEBUG][ERROR] Negative relation id exceeds decoder relation count.")
    if len(neg_hh) > 0 and neg_hh.max().item() >= num_nodes:
        print("[DEBUG][ERROR] Negative head entity id exceeds number of nodes.")
    if len(neg_ht) > 0 and neg_ht.max().item() >= num_nodes:
        print("[DEBUG][ERROR] Negative tail entity id exceeds number of nodes.")

    print("===============================================================\n")


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
    debug = bool(training_cfg.get("debug", False))
    debug_epochs = set(int(e) for e in training_cfg.get("debug_epochs", [1, 2, 5, 10, 20]))
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

    organize_legacy_runs(cfg["runs"]["out_dir"], default_model_name="rgcn_benchmark", seed=seed)
    run_dir = resolve_run_dir(cfg["runs"]["out_dir"], cfg, model_name="rgcn_benchmark", seed=seed)
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
            "graph_edges": graph_num_edges(graph),
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

    if debug:
        debug_static_data(
            train_heads=train_heads,
            train_rels=train_rels,
            train_tails=train_tails,
            edge_types=edge_types,
            graph=graph,
            num_nodes=num_nodes,
            num_relations=num_relations,
            encoder_relations=encoder_relations,
        )

    for epoch in range(1, max_epochs + 1):
        encoder.train()
        decoder.train()
        node_emb = encoder(graph, edge_types)
        positive_scores = decoder(node_emb, train_heads, train_rels, train_tails)
        neg_hh, neg_hr, neg_ht = sample_negatives(
            train_heads, train_rels, train_tails, known, candidate_array, neg_k, device
        )
        negative_scores = decoder(node_emb, neg_hh, neg_hr, neg_ht)

        scores = torch.cat([positive_scores, negative_scores])
        labels = torch.cat([
            torch.ones_like(positive_scores),
            torch.zeros_like(negative_scores),
        ])
        loss = F.binary_cross_entropy_with_logits(scores, labels)

        optimizer.zero_grad()
        loss.backward()

        if debug and epoch in debug_epochs:
            debug_negative_samples(
                epoch=epoch,
                neg_hh=neg_hh,
                neg_hr=neg_hr,
                neg_ht=neg_ht,
                num_relations=num_relations,
                num_nodes=num_nodes,
            )

            debug_scores_and_gradients(
                epoch=epoch,
                encoder=encoder,
                decoder=decoder,
                node_emb=node_emb,
                positive_scores=positive_scores,
                negative_scores=negative_scores,
            )

        optimizer.step()

        log_entry: dict[str, float | int | None] = {"epoch": epoch, "loss": float(loss.item()), "valid_mrr": None}
        if epoch % eval_every == 0 or epoch == max_epochs:
            metrics = filtered_eval(encoder, decoder, graph, edge_types, valid, known, candidate_ids, hits_k, device, max_eval_triples)
            log_entry["valid_mrr"] = float(metrics["mrr"])
            print(
                f"Epoch {epoch:04d} | "
                f"Loss: {loss.item():.4f} | "
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
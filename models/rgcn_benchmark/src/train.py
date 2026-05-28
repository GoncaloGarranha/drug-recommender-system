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

from .data import HEAD, TAIL, build_encoder_graph, build_encoder_graph_from_tensors, graph_num_edges, graph_num_nodes, known_positives, load_link_prediction_data, triples_to_tensors
from .model import DistMult, SchlichtkrullRGCN


def sample_negatives(
    heads: torch.Tensor,
    rels: torch.Tensor,
    tails: torch.Tensor,
    neg_k: int,
    device: torch.device,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample negatives without filtering (matching Schlichtkrull et al.).

    For each positive triple, randomly decide whether to corrupt the head
    or the tail (50/50). Generates `neg_k` negatives per positive.
    This implementation intentionally does NOT filter known positives.
    """
    batch_size = heads.numel()
    # repeat positives neg_k times
    pos_heads = heads.repeat(neg_k)
    pos_rels = rels.repeat(neg_k)
    pos_tails = tails.repeat(neg_k)

    corrupt_head = torch.randint(0, 2, (batch_size * neg_k,), device=device).bool()
    # Use a reasonable upper bound for entity ids; prefer global node count if available
    # Fall back to max id from provided tensors
    upper = max(1, int(num_nodes))
    random_entities = torch.randint(0, upper, (batch_size * neg_k,), device=device)

    neg_heads = pos_heads.clone()
    neg_tails = pos_tails.clone()

    neg_heads[~corrupt_head] = random_entities[~corrupt_head]
    neg_tails[corrupt_head] = random_entities[corrupt_head]

    return neg_heads, pos_rels, neg_tails


def build_edge_neighborhood_index(
    triples: list[dict[str, str]],
    num_nodes: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Build adjacency index used by sample_edge_neighborhood."""
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for edge_id, row in enumerate(triples):
        src = int(row["src_id"])
        dst = int(row["dst_id"])
        adjacency[src].append((edge_id, dst))
        adjacency[dst].append((edge_id, src))

    adj_list = [
        np.asarray(entries, dtype=np.int64) if entries else np.empty((0, 2), dtype=np.int64)
        for entries in adjacency
    ]
    degrees = np.asarray([arr.shape[0] for arr in adj_list], dtype=np.int64)
    return adj_list, degrees


def sample_edge_neighborhood(
    adj_list: list[np.ndarray],
    degrees: np.ndarray,
    sample_size: int,
    num_edges: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample edge ids with neighborhood bias, following Schlichtkrull et al.

    Vectorized implementation: instead of a Python loop selecting one edge at
    a time, we sample vertices proportionally to their degree and pick a random
    unpicked edge from each vertex. This is O(sample_size) numpy operations
    instead of O(sample_size) Python iterations.
    """
    sample_size = min(sample_size, num_edges)
    if sample_size <= 0:
        return np.empty((0,), dtype=np.int64)

    # Degree-proportional vertex sampling (with replacement for speed)
    total_degree = degrees.sum()
    if total_degree == 0:
        return rng.choice(num_edges, size=sample_size, replace=False).astype(np.int64)

    probs = degrees / total_degree
    # Sample more vertices than needed to account for collisions
    oversample = min(int(sample_size * 3), num_edges)
    chosen_vertices = rng.choice(len(degrees), size=oversample, p=probs, replace=True)

    picked = set()
    edges = []
    for v in chosen_vertices:
        if len(edges) >= sample_size:
            break
        candidates = adj_list[v]
        if candidates.shape[0] == 0:
            continue
        # Pick random edge from this vertex that hasn't been picked yet
        available_mask = np.array([c[0] not in picked for c in candidates])
        available = candidates[available_mask]
        if available.shape[0] == 0:
            continue
        chosen = available[int(rng.integers(0, available.shape[0]))]
        edge_id = int(chosen[0])
        picked.add(edge_id)
        edges.append(edge_id)

    # Fill remaining with random edges if needed
    if len(edges) < sample_size:
        remaining = np.array([i for i in range(num_edges) if i not in picked], dtype=np.int64)
        if remaining.size > 0:
            extra = rng.choice(remaining, size=min(sample_size - len(edges), remaining.size), replace=False)
            edges.extend(extra.tolist())

    return np.array(edges[:sample_size], dtype=np.int64)


def sample_batch_indices(
    total_train: int,
    batch_size: int,
    device: torch.device,
    sampler_mode: str,
    rng: np.random.Generator,
    adj_list: list[np.ndarray] | None,
    degrees: np.ndarray | None,
) -> torch.Tensor:
    """Sample training edge ids using neighborhood or uniform strategy."""
    bsize = min(batch_size, total_train)
    if sampler_mode == "neighborhood" and adj_list is not None and degrees is not None:
        ids = sample_edge_neighborhood(adj_list, degrees, bsize, total_train, rng)
        return torch.tensor(ids, dtype=torch.long, device=device)
    return torch.randperm(total_train, device=device)[:bsize]


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
    edge_norm: torch.Tensor | None = None,
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
    edge_norm: torch.Tensor | None,
    triples: list[dict[str, str]],
    known: Dict[Tuple[Any, ...], set[int]],
    candidate_ids: torch.Tensor,
    hits_k: tuple[int, ...],
    device: torch.device,
    max_eval_triples: int | None = None,
) -> Dict[str, float]:
    encoder.eval()
    decoder.eval()
    node_emb = encoder(graph, edge_types, edge_norm)

    eval_rows = triples[:max_eval_triples] if max_eval_triples else triples
    num_candidates = len(candidate_ids)
    candidate_list = candidate_ids.detach().cpu().tolist()
    candidate_index = {c: i for i, c in enumerate(candidate_list)}

    # Precompute candidate embeddings once — avoids repeating for every triple
    cand_emb = node_emb[candidate_ids]  # [C, D]
    rel_emb_table = decoder.rel_emb.weight

    all_ranks: list[float] = []
    head_ranks: list[float] = []
    tail_ranks: list[float] = []

    for row in eval_rows:
        h = int(row["src_id"])
        r = int(row["rel_id"])
        t = int(row["dst_id"])
        rel_emb = rel_emb_table[r].unsqueeze(0)  # [1, D]

        # Tail prediction: score(h, r, c) for all candidates c
        h_emb = node_emb[h].unsqueeze(0)  # [1, D]
        tail_scores = (h_emb * rel_emb * cand_emb).sum(dim=1).cpu().numpy()  # [C]
        tail_known_positions = [candidate_index[c] for c in known.get((TAIL, h, r), set()) if c != t and c in candidate_index]
        if tail_known_positions:
            tail_scores[np.asarray(tail_known_positions, dtype=np.int64)] = -1e9
        if t in candidate_index:
            true_score = tail_scores[candidate_index[t]]
            tail_rank = 1.0 + float(np.sum(tail_scores > true_score)) + (float(np.sum(tail_scores == true_score)) - 1.0) / 2.0
        else:
            tail_rank = float(num_candidates)
        tail_ranks.append(tail_rank)
        all_ranks.append(tail_rank)

        # Head prediction: score(c, r, t) for all candidates c
        t_emb = node_emb[t].unsqueeze(0)  # [1, D]
        head_scores = (cand_emb * rel_emb * t_emb).sum(dim=1).cpu().numpy()  # [C]
        head_known_positions = [candidate_index[c] for c in known.get((HEAD, r, t), set()) if c != h and c in candidate_index]
        if head_known_positions:
            head_scores[np.asarray(head_known_positions, dtype=np.int64)] = -1e9
        if h in candidate_index:
            true_score = head_scores[candidate_index[h]]
            head_rank = 1.0 + float(np.sum(head_scores > true_score)) + (float(np.sum(head_scores == true_score)) - 1.0) / 2.0
        else:
            head_rank = float(num_candidates)
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

    eval_graph, eval_edge_types, eval_edge_norm = build_encoder_graph(
        train,
        num_nodes,
        num_relations,
        device,
        add_inverse_edges=add_inverse,
    )
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
        weight_decay=0.0,
    )

    # decoder L2 regularization applied manually to embeddings (matches paper)
    decoder_l2 = float(training_cfg.get("decoder_l2", 0.01))

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
            "graph_edges": graph_num_edges(eval_graph),
        },
    )

    max_epochs = int(training_cfg.get("max_epochs", training_cfg.get("epochs", 100)))
    max_iterations = int(training_cfg.get("max_iterations", 0))
    burnin_iterations = int(training_cfg.get("burnin_iterations", 0))
    eval_every = int(training_cfg.get("eval_every", 10))
    progress_every = int(training_cfg.get("progress_every", 50))
    neg_k = int(training_cfg.get("neg_k", 1))
    graph_batch_size = int(training_cfg.get("graph_batch_size", 30000))
    graph_split_size = float(training_cfg.get("graph_split_size", 0.5))
    grad_clip_norm = float(training_cfg.get("grad_clip_norm", 0.0))
    graph_sampler = str(training_cfg.get("graph_sampler", "neighborhood")).strip().lower()
    sampler_rng = np.random.default_rng(seed)
    if graph_sampler in {"neighborhood", "edge_neighborhood"}:
        neighborhood_adj_list, neighborhood_degrees = build_edge_neighborhood_index(train, num_nodes)
    else:
        neighborhood_adj_list, neighborhood_degrees = None, None
    hits_k = tuple(int(k) for k in training_cfg.get("hits_k", [1, 3, 10]))
    early_stopping_patience = int(training_cfg.get("early_stopping_patience", 0))
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
            edge_types=eval_edge_types,
            graph=eval_graph,
            num_nodes=num_nodes,
            num_relations=num_relations,
            encoder_relations=encoder_relations,
            edge_norm=eval_edge_norm,
        )

    # Support iterations-based training (preferred to mirror the article).
    if max_iterations > 0:
        best_step = -1
        bad_checks = 0
        for step in range(1, max_iterations + 1):
            encoder.train()
            decoder.train()

            # Sample minibatch
            total_train = train_heads.numel()
            perm = sample_batch_indices(
                total_train=total_train,
                batch_size=graph_batch_size,
                device=device,
                sampler_mode=graph_sampler,
                rng=sampler_rng,
                adj_list=neighborhood_adj_list,
                degrees=neighborhood_degrees,
            )
            bsize = int(perm.numel())

            batch_heads = train_heads[perm]
            batch_rels = train_rels[perm]
            batch_tails = train_tails[perm]

            split_n = max(1, int(graph_split_size * bsize))
            graph_perm = perm[torch.randperm(bsize, device=device)[:split_n]]

            batch_graph, batch_edge_types, batch_edge_norm = build_encoder_graph_from_tensors(
                train_heads[graph_perm],
                train_rels[graph_perm],
                train_tails[graph_perm],
                num_nodes,
                num_relations,
                device,
                add_inverse_edges=add_inverse,
            )

            node_emb = encoder(batch_graph, batch_edge_types, batch_edge_norm)
            positive_scores = decoder(node_emb, batch_heads, batch_rels, batch_tails)

            neg_hh, neg_hr, neg_ht = sample_negatives(batch_heads, batch_rels, batch_tails, neg_k, device, num_nodes)
            negative_scores = decoder(node_emb, neg_hh, neg_hr, neg_ht)

            scores = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([
                torch.ones_like(positive_scores),
                torch.zeros_like(negative_scores),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            reg = (
                node_emb[batch_heads].pow(2).mean()
                + decoder.rel_emb(batch_rels).pow(2).mean()
                + node_emb[batch_tails].pow(2).mean()
            )
            loss = loss + decoder_l2 * reg

            optimizer.zero_grad()
            loss.backward()

            if grad_clip_norm and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip_norm)

            if debug and step in debug_epochs:
                debug_negative_samples(
                    epoch=step,
                    neg_hh=neg_hh,
                    neg_hr=neg_hr,
                    neg_ht=neg_ht,
                    num_relations=num_relations,
                    num_nodes=num_nodes,
                )

                debug_scores_and_gradients(
                    epoch=step,
                    encoder=encoder,
                    decoder=decoder,
                    node_emb=node_emb,
                    positive_scores=positive_scores,
                    negative_scores=negative_scores,
                )

            optimizer.step()

            if progress_every > 0 and step % progress_every == 0:
                print(f"Progress Step {step:06d} | Loss: {loss.item():.4f}", flush=True)

            log_entry: dict[str, float | int | None] = {"step": step, "loss": float(loss.item()), "valid_mrr": None}
            if (step % eval_every == 0 and step >= burnin_iterations) or step == max_iterations:
                metrics = filtered_eval(
                    encoder,
                    decoder,
                    eval_graph,
                    eval_edge_types,
                    eval_edge_norm,
                    valid,
                    known,
                    candidate_ids,
                    hits_k,
                    device,
                    max_eval_triples,
                )
                log_entry["valid_mrr"] = float(metrics["mrr"])
                print(
                    f"Step {step:06d} | "
                    f"Loss: {loss.item():.4f} | "
                    f"Valid MRR: {metrics['mrr']:.4f} | "
                    f"Head MRR: {metrics.get('head_mrr', 0.0):.4f} | "
                    f"Tail MRR: {metrics.get('tail_mrr', 0.0):.4f} | "
                    f"Hits@10: {metrics.get('hits@10', 0.0):.4f}"
                )
                if metrics["mrr"] > best_mrr:
                    best_mrr = float(metrics["mrr"])
                    best_step = step
                    bad_checks = 0
                    torch.save(
                        {
                            "encoder": encoder.state_dict(),
                            "decoder": decoder.state_dict(),
                            "config": cfg,
                            "best_step": best_step,
                            "best_valid_mrr": best_mrr,
                        },
                        checkpoints_dir / "best.pt",
                    )
                    save_json(run_dir / "best_metrics.json", {"best_step": best_step, "best_valid_mrr": best_mrr, **metrics})
                elif step >= burnin_iterations:
                    bad_checks += 1
                    if early_stopping_patience > 0 and bad_checks >= early_stopping_patience:
                        print(f"Early stopping at step {step:06d} (patience={early_stopping_patience}).")
                        train_log.append(log_entry)
                        break
            train_log.append(log_entry)
        best_epoch = best_step
    else:
        bad_checks = 0
        for epoch in range(1, max_epochs + 1):
            encoder.train()
            decoder.train()

            # Sample a minibatch of training triples (graph_batch_size) and a
            # smaller subset of those triples to build the encoder graph
            total_train = train_heads.numel()
            perm = sample_batch_indices(
                total_train=total_train,
                batch_size=graph_batch_size,
                device=device,
                sampler_mode=graph_sampler,
                rng=sampler_rng,
                adj_list=neighborhood_adj_list,
                degrees=neighborhood_degrees,
            )
            bsize = int(perm.numel())

            batch_heads = train_heads[perm]
            batch_rels = train_rels[perm]
            batch_tails = train_tails[perm]

            split_n = max(1, int(graph_split_size * bsize))
            graph_perm = perm[torch.randperm(bsize, device=device)[:split_n]]

            batch_graph, batch_edge_types, batch_edge_norm = build_encoder_graph_from_tensors(
                train_heads[graph_perm],
                train_rels[graph_perm],
                train_tails[graph_perm],
                num_nodes,
                num_relations,
                device,
                add_inverse_edges=add_inverse,
            )

            node_emb = encoder(batch_graph, batch_edge_types, batch_edge_norm)
            positive_scores = decoder(node_emb, batch_heads, batch_rels, batch_tails)

            neg_hh, neg_hr, neg_ht = sample_negatives(batch_heads, batch_rels, batch_tails, neg_k, device, num_nodes)
            negative_scores = decoder(node_emb, neg_hh, neg_hr, neg_ht)

            scores = torch.cat([positive_scores, negative_scores])
            labels = torch.cat([
                torch.ones_like(positive_scores),
                torch.zeros_like(negative_scores),
            ])
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            # manual decoder L2 regularization over embeddings used in the batch
            reg = (
                node_emb[batch_heads].pow(2).mean()
                + decoder.rel_emb(batch_rels).pow(2).mean()
                + node_emb[batch_tails].pow(2).mean()
            )
            loss = loss + decoder_l2 * reg

            optimizer.zero_grad()
            loss.backward()

            if grad_clip_norm and grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), grad_clip_norm)

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
                metrics = filtered_eval(
                    encoder,
                    decoder,
                    eval_graph,
                    eval_edge_types,
                    eval_edge_norm,
                    valid,
                    known,
                    candidate_ids,
                    hits_k,
                    device,
                    max_eval_triples,
                )
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
                    bad_checks = 0
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
                elif epoch >= burnin_iterations:
                    bad_checks += 1
                    if early_stopping_patience > 0 and bad_checks >= early_stopping_patience:
                        print(f"Early stopping at epoch {epoch:04d} (patience={early_stopping_patience}).")
                        train_log.append(log_entry)
                        break
            train_log.append(log_entry)

    torch.save(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict(), "config": cfg, "best_epoch": best_epoch},
        checkpoints_dir / "last.pt",
    )
    save_json(run_dir / "train_log.json", train_log)
    print(f"Training complete. Best validation MRR: {best_mrr:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
"""
R-GCN training script.

Train an R-GCN encoder with a DistMult decoder for configurable link prediction
tasks over a prepared knowledge graph.

Supported task modes:
    - disease_to_drug
    - drug_to_disease
    - tail_prediction

Usage:
    python -m models.rgcn.src.train --config models/rgcn/configs/disease_to_drug.yaml
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import numpy as np
import torch
import torch.nn.functional as F


from models.shared.config import load_config
from models.shared.experiment_integrity import validate_experiment_integrity
from models.shared.graph_artifacts import load_prepared_graph
from models.shared.io import ensure_dir, save_json
from models.shared.ranking import build_candidate_pool, build_training_graph, evaluate_ranking
from models.shared.runs import (
    capture_environment,
    capture_git_commit,
    organize_legacy_runs,
    resolve_run_dir,
    save_metrics,
    write_run_artifacts,
)
from models.shared.splits import build_known_positives, get_or_create_splits

from .components import DistMultDecoder, RGCNEncoder
from .utils.seed import set_seed


def build_training_pairs(
    train_triples: List[Dict[str, str]],
    task_name: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build raw indexed triples for training.

    Returns:
        src_ids, rel_ids, dst_ids as torch tensors on the selected device.
    """
    srcs: List[int] = []
    rels: List[int] = []
    dsts: List[int] = []

    for row in train_triples:
        srcs.append(int(row["src_id"]))
        rels.append(int(row["rel_id"]))
        dsts.append(int(row["dst_id"]))

    return (
        torch.tensor(srcs, dtype=torch.long, device=device),
        torch.tensor(rels, dtype=torch.long, device=device),
        torch.tensor(dsts, dtype=torch.long, device=device),
    )


def negative_sampling(
    positive_queries: torch.Tensor,
    positive_relations: torch.Tensor,
    positive_candidates: torch.Tensor,
    known_positives: dict,
    task_name: str,
    k: int,
    candidate_pool: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate negative samples by corrupting the candidate entity while excluding
    known positives for the same query or for the same (query, relation) pair
    in tail_prediction mode.
    """
    batch_size = positive_queries.shape[0]
    candidate_array = candidate_pool.detach().cpu().numpy()

    neg_queries: List[int] = []
    neg_relations: List[int] = []
    neg_candidates: List[int] = []

    rng = np.random.default_rng()

    for i in range(batch_size):
        q = int(positive_queries[i].item())
        rel = int(positive_relations[i].item())
        true_cand = int(positive_candidates[i].item())

        if task_name == "tail_prediction":
            key = (q, rel)
        else:
            key = q

        forbidden = set(known_positives.get(key, set()))
        forbidden.add(true_cand)

        collected = 0
        attempts = 0

        while collected < k:
            pick_count = max(8, (k - collected) * 4)
            picks = rng.choice(candidate_array, size=pick_count, replace=True)

            for p in picks:
                p_int = int(p)

                if p_int in forbidden:
                    continue

                neg_queries.append(q)
                neg_relations.append(rel)
                neg_candidates.append(p_int)
                collected += 1

                if collected >= k:
                    break

            attempts += 1

            if attempts > 200 and collected == 0:
                raise ValueError(
                    f"Unable to sample negatives for query {key}: "
                    "candidate pool may be exhausted or too small."
                )

    negative_queries = torch.tensor(neg_queries, dtype=torch.long, device=device)
    negative_relations = torch.tensor(neg_relations, dtype=torch.long, device=device)
    negative_candidates = torch.tensor(neg_candidates, dtype=torch.long, device=device)

    return negative_queries, negative_relations, negative_candidates


def score_training_batch(
    task_name: str,
    decoder: DistMultDecoder,
    node_embeddings: torch.Tensor,
    queries: torch.Tensor,
    relations: torch.Tensor,
    candidates: torch.Tensor,
) -> torch.Tensor:
    """
    Score training triples according to the configured task direction.
    """
    rel_emb = decoder.rel_emb(relations)

    if task_name == "disease_to_drug":
        drug_emb = node_embeddings[candidates]
        disease_emb = node_embeddings[queries]
        return torch.sum(drug_emb * rel_emb * disease_emb, dim=1)

    if task_name == "drug_to_disease":
        drug_emb = node_embeddings[queries]
        disease_emb = node_embeddings[candidates]
        return torch.sum(drug_emb * rel_emb * disease_emb, dim=1)

    if task_name == "tail_prediction":
        src_emb = node_embeddings[queries]
        dst_emb = node_embeddings[candidates]
        return torch.sum(src_emb * rel_emb * dst_emb, dim=1)

    raise ValueError(f"Unsupported task name: {task_name}")


def main() -> None:
    """
    Run the training pipeline and save the best checkpoint according to
    validation MRR.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device(cfg["training"].get("device", "cpu"))

    prepared = load_prepared_graph(cfg["data"]["processed_dir"])
    triples = prepared["triples"]
    entity2id = prepared["entity2id"]
    rel2id = prepared["rel2id"]
    entity_types = prepared["entity_types"]

    num_nodes = len(entity2id)
    num_relations = len(rel2id)

    model_cfg = cfg.get("model", {})
    add_inverse_edges = bool(model_cfg.get("add_inverse_edges", False))
    encoder_num_relations = num_relations * 2 if add_inverse_edges else num_relations

    task_cfg = cfg["task"]
    split_cfg = cfg["split"]

    task_name = task_cfg["name"]
    target_relation = task_cfg["target_relation"]
    candidate_entity_type = task_cfg["candidate_entity_type"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    organize_legacy_runs(cfg["runs"]["out_dir"], default_model_name="rgcn", seed=seed)

    if task_name != "tail_prediction":
        relation_id = rel2id.get(target_relation)

        if relation_id is None:
            raise ValueError(f"Target relation not found: {target_relation}")
    else:
        relation_id = None

    splits_dir = Path(cfg["data"]["processed_dir"]) / "splits"
    train_target, valid_target, test_target, non_target_triples = get_or_create_splits(
        output_dir=splits_dir,
        triples=triples,
        target_relation=target_relation,
        seed=seed,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        split_mode=split_cfg.get("mode") if isinstance(split_cfg, dict) else None,
    )

    split_integrity_report = validate_experiment_integrity(
        cfg,
        config_path=args.config,
        output_path=Path(splits_dir) / "split_integrity_report.json",
    )

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
        int(row["src_id"]) if task_name == "disease_to_drug" else int(row["dst_id"])
        for row in train_target + valid_target + test_target
    }

    candidate_pool = build_candidate_pool(
        entity_types=entity_types,
        entity2id=entity2id,
        candidate_entity_type=candidate_entity_type,
        device=device,
        required_entity_ids=required_candidate_ids,
    )

    corrupt_mode = str(cfg["training"].get("corrupt_mode", "tail"))

    if task_name == "tail_prediction" and corrupt_mode == "both":
        known_positives_tail = build_known_positives(
            triples=train_target + valid_target + test_target,
            task_name=None,
            query_key="src_id",
            relation_key="rel_id",
            candidate_key="dst_id",
        )

        known_positives_head = build_known_positives(
            triples=train_target + valid_target + test_target,
            task_name=None,
            query_key="dst_id",
            relation_key="rel_id",
            candidate_key="src_id",
        )

        known_positives = {**known_positives_tail, **known_positives_head}

        for key in set(known_positives_tail) & set(known_positives_head):
            known_positives[key] = known_positives_tail[key] | known_positives_head[key]

        candidate_pool = torch.arange(num_nodes, device=device)
    else:
        known_positives = build_known_positives(
            triples=train_target + valid_target + test_target,
            task_name=task_name if task_name != "tail_prediction" else None,
            query_key="src_id" if task_name == "tail_prediction" else None,
            relation_key="rel_id" if task_name == "tail_prediction" else None,
            candidate_key="dst_id" if task_name == "tail_prediction" else None,
        )

    train_src, train_relations, train_dst = build_training_pairs(
        train_triples=train_target,
        task_name=task_name,
        device=device,
    )

    encoder = RGCNEncoder(
        num_nodes,
        cfg["training"]["emb_dim"],
        encoder_num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        cfg["training"]["dropout"],
    ).to(device)

    decoder = DistMultDecoder(
        num_relations,
        cfg["training"]["emb_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    run_dir = resolve_run_dir(cfg["runs"]["out_dir"], cfg, model_name="rgcn", seed=seed)
    checkpoints_dir = run_dir / "checkpoints"

    ensure_dir(run_dir)
    ensure_dir(checkpoints_dir)

    run_splits_dir = ensure_dir(run_dir / "splits")

    for filename in [
        "train_triples.json",
        "valid_triples.json",
        "test_triples.json",
        "split_metadata.json",
    ]:
        source_path = splits_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, run_splits_dir / filename)

    model_summary = {
        "model_name": "rgcn",
        "task_name": task_name,
        "seed": seed,
        "architecture": {
            "embedding_dim": int(cfg["training"]["emb_dim"]),
            "num_layers": int(cfg["training"]["num_layers"]),
            "num_bases": int(cfg["training"]["num_bases"]),
            "dropout": float(cfg["training"]["dropout"]),
            "decoder": "DistMult",
            "add_inverse_edges": add_inverse_edges,
        },
        "relation_counts": {
            "decoder": num_relations,
            "encoder": encoder_num_relations,
        },
        "parameter_counts": {
            "encoder": sum(parameter.numel() for parameter in encoder.parameters()),
            "decoder": sum(parameter.numel() for parameter in decoder.parameters()),
        },
    }

    write_run_artifacts(
        run_dir=run_dir,
        cfg=cfg,
        split_summary={
            "task_name": task_name,
            "target_relation": target_relation,
            "train_size": len(train_target),
            "valid_size": len(valid_target),
            "test_size": len(test_target),
            "num_nodes": num_nodes,
            "num_relations_decoder": num_relations,
            "num_relations_encoder": encoder_num_relations,
            "add_inverse_edges": add_inverse_edges,
            "num_graph_edges": int(graph.num_edges()),
            "candidate_pool_size": int(len(candidate_pool)),
            "context_relations": context_relations,
            "split_integrity_report": split_integrity_report["report_path"],
        },
        model_summary=model_summary,
        environment=capture_environment(),
        git_commit=capture_git_commit(),
    )

    print(
        f"Task: {task_name} | Target relation: {target_relation} | Device: {device} | "
        f"Inverse edges: {add_inverse_edges}"
    )
    print(
        f"Relations | decoder={num_relations} | encoder={encoder_num_relations} | "
        f"graph_edges={graph.num_edges()}"
    )
    print(
        f"Split sizes | train={len(train_target)} | valid={len(valid_target)} | "
        f"test={len(test_target)}"
    )

    best_mrr = -1.0
    best_epoch = -1

    early_cfg = cfg["training"].get("early_stopping")

    if early_cfg:
        monitor = early_cfg.get("monitor", "valid_mrr")
        patience = int(early_cfg.get("patience", 20))
        mode = early_cfg.get("mode", "max")

        if mode == "max":
            best_monitor_value = float("-inf")
        else:
            best_monitor_value = float("inf")

        no_improve = 0
    else:
        monitor = None
        patience = None
        mode = None
        best_monitor_value = None
        no_improve = None

    train_log: List[Dict[str, Any]] = []

    max_epochs = int(cfg["training"].get("max_epochs", cfg["training"].get("epochs", 100)))

    for epoch in range(1, max_epochs + 1):
        encoder.train()
        decoder.train()

        node_embeddings = encoder(graph, edge_types)

        if task_name == "tail_prediction":
            if corrupt_mode == "tail":
                queries = train_src
                relations = train_relations
                candidates = train_dst
            elif corrupt_mode == "head":
                queries = train_dst
                relations = train_relations
                candidates = train_src
            else:
                mask = torch.rand(train_src.size(0), device=device) < 0.5
                queries = torch.where(mask, train_src, train_dst)
                relations = train_relations
                candidates = torch.where(mask, train_dst, train_src)
        else:
            if task_name == "disease_to_drug":
                queries = train_dst
                relations = train_relations
                candidates = train_src
            elif task_name == "drug_to_disease":
                queries = train_src
                relations = train_relations
                candidates = train_dst
            else:
                raise ValueError(f"Unsupported task name: {task_name}")

        positive_scores = score_training_batch(
            task_name=task_name,
            decoder=decoder,
            node_embeddings=node_embeddings,
            queries=queries,
            relations=relations,
            candidates=candidates,
        )

        negative_queries, negative_relations, negative_candidates = negative_sampling(
            positive_queries=queries,
            positive_relations=relations,
            positive_candidates=candidates,
            known_positives=known_positives,
            task_name=task_name,
            k=cfg["training"]["neg_k"],
            candidate_pool=candidate_pool,
            device=device,
        )

        negative_scores = score_training_batch(
            task_name=task_name,
            decoder=decoder,
            node_embeddings=node_embeddings,
            queries=negative_queries,
            relations=negative_relations,
            candidates=negative_candidates,
        )

        loss = (
            F.binary_cross_entropy_with_logits(
                positive_scores,
                torch.ones_like(positive_scores),
            )
            + F.binary_cross_entropy_with_logits(
                negative_scores,
                torch.zeros_like(negative_scores),
            )
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        log_entry: Dict[str, Any] = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "valid_mrr": None,
        }

        if epoch % cfg["training"]["eval_every"] == 0 or epoch == max_epochs:
            metrics = evaluate_ranking(
                encoder=encoder,
                decoder=decoder,
                graph=graph,
                edge_types=edge_types,
                eval_triples=valid_target,
                relation_id=relation_id,
                task_name=task_name,
                known_positives=known_positives,
                candidate_ids=candidate_pool,
                device=device,
                hits_k=tuple(cfg["training"]["hits_k"]),
            )

            log_entry["valid_mrr"] = float(metrics["mrr"])

            for k in cfg["training"]["hits_k"]:
                log_entry[f"valid_hits@{k}"] = float(metrics[f"hits@{k}"])

            print(
                f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | "
                f"Valid MRR: {metrics['mrr']:.4f}"
            )

            if metrics["mrr"] > best_mrr:
                best_mrr = float(metrics["mrr"])
                best_epoch = epoch

                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                        "decoder": decoder.state_dict(),
                        "config": cfg,
                        "task_name": task_name,
                        "target_relation": target_relation,
                        "best_epoch": best_epoch,
                        "best_valid_mrr": best_mrr,
                    },
                    checkpoints_dir / "best.pt",
                )

                best_metrics = {
                    "best_epoch": best_epoch,
                    "best_valid_mrr": best_mrr,
                    **{
                        f"best_valid_hits@{k}": float(metrics[f"hits@{k}"])
                        for k in cfg["training"]["hits_k"]
                    },
                }

                save_json(run_dir / "best_metrics.json", best_metrics)
                save_metrics(run_dir, "valid", best_metrics)

            if monitor is not None:
                if monitor == "valid_mrr":
                    mon_val = float(metrics.get("mrr", float("nan")))
                else:
                    mon_val = float(metrics.get(monitor)) if monitor in metrics else float("nan")

                improved = False

                if mode == "max" and mon_val > best_monitor_value:
                    improved = True

                if mode == "min" and mon_val < best_monitor_value:
                    improved = True

                if improved:
                    best_monitor_value = mon_val
                    no_improve = 0
                else:
                    no_improve += 1

                if no_improve is not None and no_improve >= patience:
                    print(
                        f"Early stopping: no improvement in {monitor} "
                        f"for {patience} evaluations (epoch {epoch})."
                    )
                    break

        train_log.append(log_entry)

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "config": cfg,
            "task_name": task_name,
            "target_relation": target_relation,
            "final_epoch": epoch,
            "best_epoch": best_epoch,
            "best_valid_mrr": best_mrr,
        },
        checkpoints_dir / "last.pt",
    )

    write_run_artifacts(run_dir=run_dir, train_log=train_log)

    print(f"Training complete. Best validation MRR: {best_mrr:.4f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
"""
R-GCN Model Evaluation Script

Evaluate a trained R-GCN model on the validation or test split using filtered
ranking metrics for a configurable task.

Supported task modes:
    - disease_to_drug
    - drug_to_disease
    - tail_prediction

Usage:
    python -m rgcn.src.03_evaluate --runs_dir rgcn/runs
    python -m rgcn.src.03_evaluate --run_dir rgcn/runs/<run_name> --split test
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import numpy as np
import torch

from models.shared.graph_artifacts import load_prepared_graph, resolve_processed_dir
from models.shared.io import read_csv, read_json
from models.shared.runs import find_latest_run
from models.shared.ranking import build_candidate_pool, build_training_graph, evaluate_ranking
from models.shared.splits import build_known_positives, get_or_create_splits

from .models.decoders import DistMultDecoder
from .models.rgcn_encoder import RGCNEncoder
from .utils.seed import set_seed


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json

from models.shared.benchmark_tracker import log_experiment


def main() -> None:
    """
    Load a trained checkpoint and evaluate it on the validation or test split.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="rgcn/runs")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else find_latest_run(args.runs_dir)
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    device = torch.device(cfg["training"].get("device", "cpu"))

    task_cfg = cfg["task"]
    task_name = task_cfg["name"]
    processed_dir = resolve_processed_dir(cfg["data"]["processed_dir"], task_name=task_name)

    prepared = load_prepared_graph(processed_dir)
    triples = prepared["triples"]
    entity2id = prepared["entity2id"]
    rel2id = prepared["rel2id"]
    entity_types = prepared["entity_types"]

    num_nodes = len(entity2id)
    num_relations = len(rel2id)

    split_cfg = cfg["split"]

    target_relation = task_cfg["target_relation"]
    candidate_entity_type = task_cfg["candidate_entity_type"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    if task_name == "tail_prediction":
        relation_id = 0
    else:
        relation_id = rel2id.get(target_relation)
        if relation_id is None:
            raise ValueError(f"Target relation not found: {target_relation}")

    splits_dir = processed_dir / "splits"
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

    graph, edge_types = build_training_graph(
        train_target=train_target,
        non_target_triples=non_target_triples,
        context_relations=context_relations,
        num_nodes=num_nodes,
        device=device,
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

    known_positives = build_known_positives(
        triples=train_target + valid_target + test_target,
        task_name=task_name if task_name != "tail_prediction" else None,
        query_key="src_id" if task_name == "tail_prediction" else None,
        relation_key="rel_id" if task_name == "tail_prediction" else None,
        candidate_key="dst_id" if task_name == "tail_prediction" else None,
    )

    eval_triples = valid_target if args.split == "valid" else test_target

    encoder = RGCNEncoder(
        num_nodes,
        cfg["training"]["emb_dim"],
        num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
    ).to(device)

    decoder = DistMultDecoder(
        num_relations,
        cfg["training"]["emb_dim"],
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    results = evaluate_ranking(
        encoder=encoder,
        decoder=decoder,
        graph=graph,
        edge_types=edge_types,
        eval_triples=eval_triples,
        relation_id=relation_id,
        task_name=task_name,
        known_positives=known_positives,
        candidate_ids=candidate_pool,
        device=device,
        hits_k=tuple(cfg["training"]["hits_k"]),
    )

    print("-" * 60)
    print(f"RUN: {run_dir}")
    print(f"TASK: {task_name}")
    print(f"SPLIT: {args.split}")
    print("-" * 60)
    print(f"MRR: {results['mrr']:.4f}")
    for k in cfg["training"]["hits_k"]:
        print(f"Hits@{k}: {results[f'hits@{k}']:.4f}")
    print("-" * 60)

    log_experiment(
        model_name=f"R-GCN ({task_name})",
        hits1=results["hits@1"],
        hits3=results["hits@3"],
        hits10=results["hits@10"],
        mrr=results["mrr"],
        notes=f"Run: {Path(run_dir).name} | Split: {args.split}",
    )

    output_path = os.path.join(run_dir, f"final_metrics_{task_name}_{args.split}.json")
    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

    print(f"Saved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
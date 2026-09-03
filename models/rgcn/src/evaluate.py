"""
R-GCN model evaluation script.

Evaluate a trained R-GCN model on the validation or test split using filtered
or unfiltered ranking metrics for a configurable task.

Supported task modes:
    - disease_to_drug
    - drug_to_disease
    - tail_prediction

Usage:
    python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn
    python -m models.rgcn.src.evaluate --run_dir results/runs/rgcn/<run_name> --split test
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from models.shared.benchmark_tracker import log_experiment
from models.shared.graph_artifacts import load_prepared_graph, resolve_processed_dir
from models.shared.io import read_json
from models.shared.ranking import build_candidate_pool, build_training_graph, evaluate_ranking
from models.shared.runs import find_latest_run, save_metrics
from models.shared.splits import build_known_positives, get_or_create_splits, load_splits

from .components import DistMultDecoder, RGCNEncoder
from .utils.seed import set_seed


def main() -> None:
    """
    Load a trained checkpoint and evaluate it on the validation or test split.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="results/runs/rgcn")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--split", default="test", choices=["valid", "test"])
    parser.add_argument(
        "--ranking_mode",
        default="filtered",
        choices=["filtered", "unfiltered"],
        help="Use filtered ranking (default) or raw/unfiltered ranking.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else find_latest_run(args.runs_dir)
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    run_config_path = Path(run_dir) / "config.yaml"
    if run_config_path.exists():
        import yaml
        with run_config_path.open("r", encoding="utf-8") as file_obj:
            cfg = yaml.safe_load(file_obj)
        print(f"Using run config: {run_config_path}")
    else:
        cfg = checkpoint["config"]
        print("Using config embedded in checkpoint.")

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

    model_cfg = cfg.get("model", {})
    add_inverse_edges = bool(model_cfg.get("add_inverse_edges", False))
    encoder_num_relations = num_relations * 2 if add_inverse_edges else num_relations

    split_cfg = cfg["split"]

    target_relation = task_cfg["target_relation"]
    candidate_entity_type = task_cfg["candidate_entity_type"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    if task_name == "tail_prediction":
        relation_id = None
    else:
        relation_id = rel2id.get(target_relation)

        if relation_id is None:
            raise ValueError(f"Target relation not found: {target_relation}")

    run_splits_dir = Path(run_dir) / "splits"
    global_splits_dir = processed_dir / "splits"

    if run_splits_dir.exists():
        print(f"Using splits saved in run: {run_splits_dir}")
        train_target, valid_target, test_target = load_splits(run_splits_dir)

        # In generic link prediction (target_relation=None), the saved
        # train/valid/test splits partition the complete graph. Only the
        # training split may be used to construct the encoder graph;
        # including the remaining triples here would leak validation/test
        # edges into evaluation.
        if target_relation is None:
            non_target_triples = []
        else:
            non_target_triples = [
                row for row in triples
                if (row.get("rel") or "").strip() != target_relation
            ]
    else:
        print(f"Using global splits: {global_splits_dir}")
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

    graph, edge_types = build_training_graph(
        train_target=train_target,
        non_target_triples=non_target_triples,
        context_relations=context_relations,
        num_nodes=num_nodes,
        device=device,
        num_relations=num_relations,
        add_inverse_edges=add_inverse_edges,
    )

    explicit_candidate_path = task_cfg.get(
        "candidate_entities_path"
    )

    explicit_candidate_entities = (
        read_json(explicit_candidate_path)
        if explicit_candidate_path
        else None
    )

    # Em disease_to_drug, os triples estão representados como:
    # DRUG --INDICATION--> DISEASE.
    # A entidade candidata é, portanto, o src.
    # Determinar o lado candidato através do tipo real das entidades.
    # Isto evita depender da orientação assumida pelo nome da tarefa.
    required_candidate_ids = set()

    if task_name == "tail_prediction":
        # Generic link-prediction benchmarks evaluate both head and tail
        # prediction. Candidate entities are therefore not inferred from
        # semantic entity types.
        for row in valid_target + test_target:
            required_candidate_ids.add(int(row["src_id"]))
            required_candidate_ids.add(int(row["dst_id"]))
    else:
        for row in valid_target + test_target:
            src_entity = str(row.get("src", ""))
            dst_entity = str(row.get("dst", ""))

            matching_ids = []

            if entity_types.get(src_entity) == candidate_entity_type:
                matching_ids.append(int(row["src_id"]))

            if entity_types.get(dst_entity) == candidate_entity_type:
                matching_ids.append(int(row["dst_id"]))

            if len(matching_ids) != 1:
                raise ValueError(
                    "Could not determine exactly one candidate entity for "
                    f"triple {src_entity} --{row.get('rel')}--> {dst_entity}. "
                    f"candidate_entity_type={candidate_entity_type!r}, "
                    f"src_type={entity_types.get(src_entity)!r}, "
                    f"dst_type={entity_types.get(dst_entity)!r}"
                )

            required_candidate_ids.add(matching_ids[0])

    candidate_pool = build_candidate_pool(
        entity_types=entity_types,
        entity2id=entity2id,
        candidate_entity_type=candidate_entity_type,
        device=device,
        required_entity_ids=required_candidate_ids,
        explicit_entity_ids=explicit_candidate_entities,
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
        encoder_num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        cfg["training"]["dropout"],
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

    with torch.no_grad():
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
            direction="both" if task_name == "tail_prediction" else "tail",
            filtered=args.ranking_mode == "filtered",
        )

    print("-" * 60)
    print(f"RUN: {run_dir}")
    print(f"TASK: {task_name}")
    print(f"SPLIT: {args.split}")
    print(f"RANKING MODE: {args.ranking_mode}")
    print("-" * 60)
    print(f"MRR: {results['mrr']:.4f}")

    for k in cfg["training"]["hits_k"]:
        print(f"Hits@{k}: {results[f'hits@{k}']:.4f}")

    print("-" * 60)

    log_experiment(
        dataset=Path(cfg["data"]["processed_dir"]).name,
        model_name=f"R-GCN ({task_name})",
        hits1=results["hits@1"],
        hits3=results["hits@3"],
        hits10=results["hits@10"],
        mrr=results["mrr"],
        notes=(
            f"Run: {Path(run_dir).name} | Split: {args.split} | "
            f"Ranking: {args.ranking_mode}"
        ),
    )

    metrics_split = (
        args.split
        if args.ranking_mode == "filtered"
        else f"{args.split}_unfiltered"
    )
    save_metrics(run_dir, metrics_split, results)

    output_suffix = (
        args.split
        if args.ranking_mode == "filtered"
        else f"{args.split}_unfiltered"
    )
    output_path = os.path.join(
        run_dir,
        f"final_metrics_{task_name}_{output_suffix}.json",
    )

    with open(output_path, "w", encoding="utf-8") as file_obj:
        json.dump(results, file_obj, indent=2)

    print(f"Saved evaluation metrics to: {output_path}")


if __name__ == "__main__":
    main()
"""Evaluate an R-GCN checkpoint on a shared fixed negative set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.shared.evaluate_negative_set import (
    build_metrics_payload,
    evaluate_score_sets,
    load_evaluation_set,
)
from models.shared.graph_artifacts import (
    load_prepared_graph,
    resolve_processed_dir,
)
from models.shared.ranking import build_training_graph
from models.shared.runs import find_latest_run
from models.shared.splits import get_or_create_splits, load_splits

from .components import DistMultDecoder, RGCNEncoder
from .negative_set_evaluation import score_batch
from .utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--runs_dir",
        default="results/runs/rgcn",
    )
    parser.add_argument(
        "--run_dir",
        default=None,
    )
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="test",
    )
    parser.add_argument(
        "--evaluation_set",
        required=True,
        help="Shared fixed-candidate evaluation-set JSON.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )

    return parser.parse_args()


def validate_examples(
    examples: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    num_negatives: int,
) -> None:
    if len(examples) != len(split_rows):
        raise ValueError(
            "Evaluation-set size differs from split size: "
            f"{len(examples)} != {len(split_rows)}"
        )

    for index, (example, split_row) in enumerate(
        zip(examples, split_rows)
    ):
        # disease_to_drug:
        # split src = Drug, split dst = Disease
        expected_query = int(split_row["dst_id"])
        expected_positive = int(split_row["src_id"])

        query = int(example["query_id"])
        positive = int(example["positive_candidate_id"])
        negatives = [
            int(value)
            for value in example["negative_candidate_ids"]
        ]

        if query != expected_query:
            raise ValueError(
                f"Query mismatch at example {index}: "
                f"{query} != {expected_query}"
            )

        if positive != expected_positive:
            raise ValueError(
                f"Positive mismatch at example {index}: "
                f"{positive} != {expected_positive}"
            )

        if len(negatives) != num_negatives:
            raise ValueError(
                f"Example {index} has {len(negatives)} negatives; "
                f"expected {num_negatives}."
            )

        if len(set(negatives)) != num_negatives:
            raise ValueError(
                f"Example {index} contains duplicate negatives."
            )

        if positive in negatives:
            raise ValueError(
                f"Example {index} contains its positive as a negative."
            )


def main() -> None:
    args = parse_args()

    run_dir = Path(
        args.run_dir
        if args.run_dir
        else find_latest_run(args.runs_dir)
    )

    checkpoint_path = run_dir / "checkpoints" / "best.pt"

    if not checkpoint_path.is_file():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}"
        )

    evaluation_set_path = Path(args.evaluation_set)

    try:
        evaluation_payload = load_evaluation_set(
            evaluation_set_path
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error

    metadata = evaluation_payload["metadata"]
    examples = evaluation_payload["examples"]

    if metadata.get("split") != args.split:
        raise SystemExit(
            "Evaluation-set split differs from --split: "
            f"{metadata.get('split')} != {args.split}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    cfg = checkpoint["config"]
    seed = int(cfg.get("seed", 42))

    set_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    task_cfg = cfg["task"]
    task_name = str(task_cfg["name"])
    target_relation = str(task_cfg["target_relation"])

    if task_name != "disease_to_drug":
        raise SystemExit(
            "This evaluator currently supports only "
            "task.name=disease_to_drug."
        )

    if metadata.get("target_relation") != target_relation:
        raise SystemExit(
            "Evaluation-set relation differs from the checkpoint."
        )

    device = torch.device(
        cfg["training"].get("device", "cpu")
    )

    processed_dir = resolve_processed_dir(
        cfg["data"]["processed_dir"],
        task_name=task_name,
    )

    prepared = load_prepared_graph(processed_dir)

    triples = prepared["triples"]
    entity2id = prepared["entity2id"]
    rel2id = prepared["rel2id"]

    num_nodes = len(entity2id)
    num_relations = len(rel2id)

    relation_id = rel2id.get(target_relation)

    if relation_id is None:
        raise SystemExit(
            f"Target relation not found: {target_relation}"
        )

    run_splits_dir = run_dir / "splits"
    global_splits_dir = processed_dir / "splits"

    if run_splits_dir.is_dir():
        print(f"Using splits saved in run: {run_splits_dir}")

        train_rows, valid_rows, test_rows = load_splits(
            run_splits_dir
        )

        non_target_rows = [
            row
            for row in triples
            if str(row.get("rel", "")).strip()
            != target_relation
        ]
    else:
        print(f"Using global splits: {global_splits_dir}")

        split_cfg = cfg["split"]

        (
            train_rows,
            valid_rows,
            test_rows,
            non_target_rows,
        ) = get_or_create_splits(
            output_dir=global_splits_dir,
            triples=triples,
            target_relation=target_relation,
            seed=seed,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            split_mode=split_cfg.get("mode"),
        )

    split_rows = (
        valid_rows
        if args.split == "valid"
        else test_rows
    )

    num_negatives = int(metadata["num_negatives"])

    validate_examples(
        examples=examples,
        split_rows=split_rows,
        num_negatives=num_negatives,
    )

    model_cfg = cfg.get("model", {})
    add_inverse_edges = bool(
        model_cfg.get("add_inverse_edges", False)
    )

    encoder_num_relations = (
        num_relations * 2
        if add_inverse_edges
        else num_relations
    )

    graph, edge_types = build_training_graph(
        train_target=train_rows,
        non_target_triples=non_target_rows,
        context_relations=task_cfg.get(
            "context_relations",
            [target_relation],
        ),
        num_nodes=num_nodes,
        device=device,
        num_relations=num_relations,
        add_inverse_edges=add_inverse_edges,
    )

    encoder = RGCNEncoder(
        num_nodes,
        cfg["training"]["emb_dim"],
        encoder_num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
        use_layer_norm=model_cfg.get(
            "use_layer_norm",
            True,
        ),
        use_residual=model_cfg.get(
            "use_residual",
            True,
        ),
        use_relation_gates=model_cfg.get(
            "use_relation_gates",
            False,
        ),
    ).to(device)

    decoder = DistMultDecoder(
        num_relations,
        cfg["training"]["emb_dim"],
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    decoder.eval()

    positive_score_parts: list[np.ndarray] = []
    negative_score_parts: list[np.ndarray] = []

    with torch.no_grad():
        node_embeddings = encoder(graph, edge_types)

        for start in range(
            0,
            len(examples),
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                len(examples),
            )

            batch = examples[start:end]
            current_batch_size = len(batch)

            queries = torch.tensor(
                [
                    int(example["query_id"])
                    for example in batch
                ],
                dtype=torch.long,
                device=device,
            )

            relations = torch.full(
                (current_batch_size,),
                int(relation_id),
                dtype=torch.long,
                device=device,
            )

            positive_candidates = torch.tensor(
                [
                    int(example["positive_candidate_id"])
                    for example in batch
                ],
                dtype=torch.long,
                device=device,
            )

            negative_candidates = torch.tensor(
                [
                    [
                        int(value)
                        for value in example[
                            "negative_candidate_ids"
                        ]
                    ]
                    for example in batch
                ],
                dtype=torch.long,
                device=device,
            )

            positive_scores = score_batch(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                queries=queries,
                relations=relations,
                candidates=positive_candidates,
            )

            flat_negative_candidates = (
                negative_candidates.reshape(-1)
            )

            flat_negative_scores = score_batch(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                queries=queries.repeat_interleave(
                    num_negatives
                ),
                relations=relations.repeat_interleave(
                    num_negatives
                ),
                candidates=flat_negative_candidates,
            )

            negative_scores = flat_negative_scores.view(
                current_batch_size,
                num_negatives,
            )

            positive_score_parts.append(
                positive_scores.detach().cpu().numpy()
            )

            negative_score_parts.append(
                negative_scores.detach().cpu().numpy()
            )

    positive_scores_array = np.concatenate(
        positive_score_parts
    ).astype(np.float64)

    negative_scores_array = np.concatenate(
        negative_score_parts,
        axis=0,
    ).astype(np.float64)

    metrics, ranks = evaluate_score_sets(
        positive_scores=positive_scores_array,
        negative_scores=negative_scores_array,
    )

    metrics_payload = build_metrics_payload(
        metrics=metrics,
        metadata=metadata,
        split=args.split,
        evaluation_set_path=evaluation_set_path,
        example_count=len(examples),
    )

    metrics_payload.update(
        {
            "model": "R-GCN",
            "task": task_name,
            "target_relation": target_relation,
            "checkpoint": str(checkpoint_path.resolve()),
        }
    )

    rows: list[dict[str, object]] = []

    for index, example in enumerate(examples):
        negative_scores = negative_scores_array[index]

        rows.append(
            {
                "split": args.split,
                "example_index": int(
                    example.get("example_index", index)
                ),
                "query_id": int(example["query_id"]),
                "relation": target_relation,
                "relation_id": int(relation_id),
                "positive_candidate_id": int(
                    example["positive_candidate_id"]
                ),
                "positive_score": float(
                    positive_scores_array[index]
                ),
                "rank": float(ranks[index]),
                "reciprocal_rank": float(
                    1.0 / ranks[index]
                ),
                "negative_candidate_ids": "|".join(
                    str(value)
                    for value in example[
                        "negative_candidate_ids"
                    ]
                ),
                "negative_scores": "|".join(
                    f"{float(value):.8f}"
                    for value in negative_scores
                ),
            }
        )

    output_dir = run_dir / "negative_set_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    negative_mode = str(
        metadata.get("negative_mode", "random")
    )

    suffix = (
        f"{args.split}_shared_"
        f"{negative_mode}_neg{num_negatives}_"
        f"seed{metadata['seed']}"
    )

    metrics_path = (
        output_dir / f"metrics_{suffix}.json"
    )

    predictions_path = (
        output_dir / f"predictions_{suffix}.csv"
    )

    metrics_path.write_text(
        json.dumps(
            metrics_payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    with predictions_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]) if rows else [],
        )

        if rows:
            writer.writeheader()
            writer.writerows(rows)

    print("-" * 72)
    print(f"Run:        {run_dir}")
    print(f"Split:      {args.split}")
    print(f"Protocol:   1 positive vs {num_negatives} negatives")
    print(f"Candidates: {num_negatives + 1}")
    print(f"Seed:       {metadata['seed']}")
    print("-" * 72)
    print(f"MRR:        {metrics['mrr']:.4f}")
    print(f"Hits@1:     {metrics['hits@1']:.4f}")
    print(f"Hits@3:     {metrics['hits@3']:.4f}")
    print(f"Hits@10:    {metrics['hits@10']:.4f}")
    print(f"AUC-ROC:    {metrics['auc_roc']:.4f}")
    print(
        f"Avg. Prec.: {metrics['average_precision']:.4f}"
    )
    print("-" * 72)
    print(f"Metrics:     {metrics_path}")
    print(f"Predictions: {predictions_path}")


if __name__ == "__main__":
    main()

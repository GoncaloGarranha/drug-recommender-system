"""Evaluate the content-based baseline on fixed sampled candidate sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from models.shared.benchmark_tracker import log_experiment
from models.shared.config import load_config
from models.shared.io import read_json
from models.shared.metrics import compute_ranking_metrics
from models.shared.runs import resolve_run_dir

from .baseline import (
    HybridContentBasedBaseline,
    load_fixed_weights,
    resolve_existing_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        default="test",
    )
    parser.add_argument(
        "--evaluation-set",
        help=(
            "Path to a shared evaluation-set JSON file. "
            "When omitted, the default random-neg50 seed file "
            "inside processed_dir/evaluation_sets is used."
        ),
    )
    parser.add_argument(
        "--weights-file",
        help=(
            "Optional selected_weights.json file. "
            "When omitted, weights are read from the YAML config."
        ),
    )

    return parser.parse_args()


def load_weights(
    baseline_config: Mapping[str, object],
    weights_file: str | None,
) -> dict[str, float]:
    if not weights_file:
        return load_fixed_weights(baseline_config)

    weights_path = resolve_existing_path(weights_file)

    payload = json.loads(
        weights_path.read_text(encoding="utf-8")
    )

    selected = payload.get("selected_weights")

    if not isinstance(selected, dict):
        raise ValueError(
            f"Missing selected_weights in {weights_path}"
        )

    return {
        str(name): float(value)
        for name, value in selected.items()
    }


def main() -> None:
    args = parse_args()

    try:
        config_path = resolve_existing_path(args.config)
        config = load_config(config_path)

        triples_path = resolve_existing_path(
            config["data"]["triples_path"]
        )
        processed_dir = resolve_existing_path(
            config["data"]["processed_dir"]
        )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    seed = int(config.get("seed", 42))
    task_config = config["task"]

    task_name = str(
        task_config.get("name", "disease_to_drug")
    )

    if task_name != "disease_to_drug":
        raise SystemExit(
            "This evaluator supports only task.name=disease_to_drug."
        )

    target_relation = str(
        task_config["target_relation"]
    )

    entity2id_path = processed_dir / "entity2id.json"
    entity_types_path = processed_dir / "entity_types.json"
    train_split_path = (
        processed_dir
        / "splits"
        / "train_triples.json"
    )

    for required_path in (
        entity2id_path,
        entity_types_path,
        train_split_path,
    ):
        if not required_path.is_file():
            raise SystemExit(
                f"Missing required file: {required_path}"
            )

    if args.evaluation_set:
        evaluation_set_path = resolve_existing_path(
            args.evaluation_set
        )
    else:
        evaluation_set_path = (
            processed_dir
            / "evaluation_sets"
            / (
                f"{args.split}_random_neg50_"
                f"seed{seed}.json"
            )
        )

    if not evaluation_set_path.is_file():
        raise SystemExit(
            f"Evaluation set not found: {evaluation_set_path}"
        )

    evaluation_payload = read_json(
        evaluation_set_path
    )

    metadata = evaluation_payload.get("metadata", {})

    if metadata.get("split") != args.split:
        raise SystemExit(
            "Evaluation-set split does not match --split: "
            f"{metadata.get('split')} != {args.split}"
        )

    if metadata.get("target_relation") != target_relation:
        raise SystemExit(
            "Evaluation-set target relation does not match config."
        )

    entity2id = read_json(entity2id_path)
    entity_types = read_json(entity_types_path)

    candidate_type = str(
        task_config.get(
            "candidate_entity_type",
            "DRUG",
        )
    )

    candidate_entities = sorted(
        [
            str(entity)
            for entity, entity_type in entity_types.items()
            if (
                candidate_type == "ALL"
                or str(entity_type) == candidate_type
            )
        ],
        key=lambda entity: int(entity2id[entity]),
    )

    train_df = pd.DataFrame(
        read_json(train_split_path)
    )

    baseline_config = dict(
        config.get(
            "content_based",
            config.get("baseline", {}),
        )
    )

    weights = load_weights(
        baseline_config,
        args.weights_file,
    )

    model = HybridContentBasedBaseline(
        triples_path=triples_path,
        candidate_entities=candidate_entities,
        train_triples=train_df,
        config=baseline_config,
        target_relation=target_relation,
    )

    print(
        f"--- Evaluating {args.split}: "
        f"1 positive vs {metadata['num_negatives']} negatives ---"
    )

    ranks, diagnostics, detailed_rows = (
        model.evaluate_negative_set(
            evaluation_payload=evaluation_payload,
            weights=weights,
        )
    )

    if not ranks:
        raise SystemExit(
            "No sampled-evaluation ranks were computed."
        )

    metrics = compute_ranking_metrics(ranks)

    run_dir = resolve_run_dir(
        config["runs"]["out_dir"],
        config,
        model_name="content_based",
        seed=seed,
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    negative_mode = str(
        metadata.get("negative_mode", "random")
    )
    num_negatives = int(
        metadata["num_negatives"]
    )

    suffix = (
        f"{args.split}_{negative_mode}_"
        f"neg{num_negatives}"
    )

    metrics_payload = {
        **metrics,
        "protocol": (
            f"1_positive_vs_{num_negatives}_negatives"
        ),
        "negative_mode": negative_mode,
        "num_negatives": num_negatives,
        "candidate_count_per_example": (
            num_negatives + 1
        ),
        "seed": int(metadata["seed"]),
        "evaluation_set": str(
            evaluation_set_path
        ),
        "selected_weights": weights,
    }

    write_json(
        run_dir / f"metrics_{suffix}.json",
        metrics_payload,
    )

    write_json(
        run_dir / f"diagnostics_{suffix}.json",
        diagnostics,
    )

    pd.DataFrame(detailed_rows).to_csv(
        run_dir / f"predictions_{suffix}.csv",
        index=False,
    )

    print("--- Final sampled-ranking metrics ---")
    print(f"Hits@1:  {metrics['hits@1']:.4f}")
    print(f"Hits@3:  {metrics['hits@3']:.4f}")
    print(f"Hits@10: {metrics['hits@10']:.4f}")
    print(f"MRR:     {metrics['mrr']:.4f}")
    print(f"Saved to: {run_dir}")

    log_experiment(
        dataset=Path(config_path).stem,
        model_name="Hybrid Content-Based",
        hits1=metrics["hits@1"],
        hits3=metrics["hits@3"],
        hits10=metrics["hits@10"],
        mrr=metrics["mrr"],
        notes=(
            f"Shared sampled evaluation | "
            f"split={args.split} | "
            f"negative_mode={negative_mode} | "
            f"num_negatives={num_negatives} | "
            f"seed={metadata['seed']}"
        ),
    )


if __name__ == "__main__":
    main()

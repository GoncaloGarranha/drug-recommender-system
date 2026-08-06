"""Generate deterministic sampled evaluation sets shared by all models.

For each positive INDICATION triple, the output contains the positive drug and
a fixed number of negative drugs sampled from the common candidate universe.

Known positives for the same disease are never sampled as negatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_split(path: Path) -> list[dict[str, Any]]:
    rows = read_json(path)

    if not isinstance(rows, list):
        raise ValueError(f"Split must contain a JSON list: {path}")

    required = {"src", "rel", "dst"}

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Invalid row {index} in {path}")

        missing = required - set(row)

        if missing:
            raise ValueError(
                f"Row {index} in {path} is missing fields: "
                f"{sorted(missing)}"
            )

    return rows


def build_known_positives(
    split_rows: list[dict[str, Any]],
    target_relation: str,
) -> dict[str, set[str]]:
    known: dict[str, set[str]] = {}

    for row in split_rows:
        if str(row["rel"]) != target_relation:
            continue

        drug = str(row["src"])
        disease = str(row["dst"])

        known.setdefault(disease, set()).add(drug)

    return known


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Model configuration containing data.processed_dir.",
    )
    parser.add_argument(
        "--split",
        choices=("valid", "test"),
        required=True,
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--force",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)

    if not config_path.is_file():
        raise SystemExit(f"Configuration not found: {config_path}")

    config = yaml.safe_load(
        config_path.read_text(encoding="utf-8")
    )

    processed_dir = Path(config["data"]["processed_dir"])
    splits_dir = processed_dir / "splits"

    split_paths = {
        name: splits_dir / f"{name}_triples.json"
        for name in ("train", "valid", "test")
    }

    for path in split_paths.values():
        if not path.is_file():
            raise SystemExit(f"Split not found: {path}")

    entity2id_path = processed_dir / "entity2id.json"
    entity_types_path = processed_dir / "entity_types.json"

    if not entity2id_path.is_file():
        raise SystemExit(f"Missing entity mapping: {entity2id_path}")

    if not entity_types_path.is_file():
        raise SystemExit(f"Missing entity types: {entity_types_path}")

    entity2id = read_json(entity2id_path)
    entity_types = read_json(entity_types_path)

    task = config["task"]
    target_relation = str(task["target_relation"])
    candidate_type = str(task["candidate_entity_type"])

    seed = int(
        args.seed
        if args.seed is not None
        else config.get("seed", 42)
    )

    if args.num_negatives < 1:
        raise SystemExit("--num-negatives must be at least 1.")

    candidate_entities = sorted(
        (
            str(entity)
            for entity, entity_type in entity_types.items()
            if candidate_type == "ALL"
            or str(entity_type) == candidate_type
        ),
        key=lambda entity: int(entity2id[entity]),
    )

    if not candidate_entities:
        raise SystemExit("Candidate pool is empty.")

    all_rows: list[dict[str, Any]] = []

    for split_name in ("train", "valid", "test"):
        all_rows.extend(load_split(split_paths[split_name]))

    evaluation_rows = load_split(split_paths[args.split])

    known_positives = build_known_positives(
        all_rows,
        target_relation=target_relation,
    )

    output_dir = processed_dir / "evaluation_sets"

    output_path = output_dir / (
        f"{args.split}_random_neg{args.num_negatives}_"
        f"seed{seed}.json"
    )

    if output_path.exists() and not args.force:
        raise SystemExit(
            f"Output already exists: {output_path}\n"
            "Use --force to recreate it."
        )

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for index, row in enumerate(evaluation_rows):
        relation = str(row["rel"])

        if relation != target_relation:
            continue

        positive_drug = str(row["src"])
        disease = str(row["dst"])

        forbidden = set(known_positives.get(disease, set()))
        forbidden.add(positive_drug)

        eligible_negatives = [
            candidate
            for candidate in candidate_entities
            if candidate not in forbidden
        ]

        if len(eligible_negatives) < args.num_negatives:
            raise SystemExit(
                f"Not enough valid negatives for row {index}: "
                f"required={args.num_negatives}, "
                f"available={len(eligible_negatives)}"
            )

        negative_drugs = rng.sample(
            eligible_negatives,
            args.num_negatives,
        )

        records.append(
            {
                "example_index": index,
                "query": disease,
                "query_id": int(entity2id[disease]),
                "relation": relation,
                "positive_candidate": positive_drug,
                "positive_candidate_id": int(
                    entity2id[positive_drug]
                ),
                "negative_candidates": negative_drugs,
                "negative_candidate_ids": [
                    int(entity2id[candidate])
                    for candidate in negative_drugs
                ],
            }
        )

    payload = {
        "metadata": {
            "format_version": 1,
            "task_name": str(task["name"]),
            "target_relation": target_relation,
            "query_entity_type": str(
                task["query_entity_type"]
            ),
            "candidate_entity_type": candidate_type,
            "split": args.split,
            "protocol": (
                f"1_positive_vs_{args.num_negatives}_random_negatives"
            ),
            "negative_mode": "random",
            "num_negatives": args.num_negatives,
            "candidate_pool_size": len(candidate_entities),
            "example_count": len(records),
            "seed": seed,
            "filtering": (
                "All train, validation and test positives for the "
                "same query are excluded from negative sampling."
            ),
            "source_splits": {
                name: {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
                for name, path in split_paths.items()
            },
        },
        "examples": records,
    }

    write_json(output_path, payload)

    print(f"Saved: {output_path}")
    print(f"Examples: {len(records)}")
    print(f"Candidate pool: {len(candidate_entities)}")
    print(f"Negatives per example: {args.num_negatives}")
    print(f"Seed: {seed}")


if __name__ == "__main__":
    main()

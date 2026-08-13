from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from pykeen.triples import TriplesFactory


REPO_ROOT = Path(__file__).resolve().parents[3]

TARGET_RELATION = "INDICATION"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_publication_factories(processed_dir=None):
    if processed_dir is None:
        processed_dir = (
            REPO_ROOT
            / "models/rgcn/data/processed/disease_to_drug/"
            "six_months_publication_random_split"
        )
    else:
        processed_dir = Path(processed_dir)
        if not processed_dir.is_absolute():
            processed_dir = REPO_ROOT / processed_dir

    entity2id = read_json(processed_dir / "entity2id.json")
    rel2id = read_json(processed_dir / "rel2id.json")
    entity_types = read_json(processed_dir / "entity_types.json")

    splits_dir = processed_dir / "splits"

    train = read_json(splits_dir / "train_triples.json")
    valid = read_json(splits_dir / "valid_triples.json")
    test = read_json(splits_dir / "test_triples.json")

    indexed_graph = pd.read_csv(
        processed_dir / "triples_indexed.csv",
        dtype=str,
    )

    non_target = indexed_graph[
        indexed_graph["rel"] != TARGET_RELATION
    ]

    train_rows = [
        [
            int(row["src_id"]),
            int(row["rel_id"]),
            int(row["dst_id"]),
        ]
        for row in train
    ]

    train_rows.extend(
        non_target[
            ["src_id", "rel_id", "dst_id"]
        ]
        .astype(int)
        .values
        .tolist()
    )

    entity_to_id = {
        str(entity): int(index)
        for entity, index in entity2id.items()
    }

    relation_to_id = {
        str(relation): int(index)
        for relation, index in rel2id.items()
    }

    training = TriplesFactory(
        mapped_triples=torch.as_tensor(
            train_rows,
            dtype=torch.long,
        ),
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id,
        create_inverse_triples=False,
    )

    def split_factory(rows):
        return TriplesFactory(
            mapped_triples=torch.as_tensor(
                [
                    [
                        int(row["src_id"]),
                        int(row["rel_id"]),
                        int(row["dst_id"]),
                    ]
                    for row in rows
                ],
                dtype=torch.long,
            ),
            entity_to_id=training.entity_to_id,
            relation_to_id=training.relation_to_id,
            create_inverse_triples=False,
        )

    validation = split_factory(valid)
    testing = split_factory(test)

    candidate_drugs = sorted(
        entity
        for entity, entity_type in entity_types.items()
        if entity_type == "DRUG"
    )

    candidate_ids = torch.as_tensor(
        [
            entity_to_id[entity]
            for entity in candidate_drugs
        ],
        dtype=torch.long,
    )

    return {
        "training": training,
        "validation": validation,
        "testing": testing,
        "candidate_drugs": candidate_drugs,
        "candidate_ids": candidate_ids,
        "entity_types": entity_types,
        "train_rows": train,
        "valid_rows": valid,
        "test_rows": test,
        "processed_dir": processed_dir,
    }

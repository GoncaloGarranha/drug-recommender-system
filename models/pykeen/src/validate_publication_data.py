from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch
from pykeen.triples import TriplesFactory


REPO_ROOT = Path(__file__).resolve().parents[3]

PROCESSED_DIR = (
    REPO_ROOT
    / "models/rgcn/data/processed/disease_to_drug/"
    "six_months_random_split"
)

TRIPLES_PATH = (
    REPO_ROOT
    / "kg_constructor/data/builds/"
    "2025-10_to_2026-03/"
    "triples/triples_with_dgidb.csv"
)

TARGET_RELATION = "INDICATION"

EXPECTED = {
    "graph_triples": 51759,
    "entities": 10380,
    "relations": 5,
    "target_triples": 6143,
    "train_target": 4914,
    "valid_target": 614,
    "test_target": 615,
    "candidate_drugs": 2347,
    "train_graph_triples": 50530,
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def triple_key(row: dict[str, str]) -> tuple[str, str, str]:
    return str(row["src"]), str(row["rel"]), str(row["dst"])


def mapped_tensor(rows: list[dict[str, str]]) -> torch.LongTensor:
    return torch.as_tensor(
        [
            [
                int(row["src_id"]),
                int(row["rel_id"]),
                int(row["dst_id"]),
            ]
            for row in rows
        ],
        dtype=torch.long,
    )


def main() -> None:
    print("=== PyKEEN publication data validation ===")

    graph = pd.read_csv(TRIPLES_PATH, dtype=str)

    entity2id = read_json(PROCESSED_DIR / "entity2id.json")
    rel2id = read_json(PROCESSED_DIR / "rel2id.json")
    entity_types = read_json(PROCESSED_DIR / "entity_types.json")

    splits_dir = PROCESSED_DIR / "splits"

    train = read_json(splits_dir / "train_triples.json")
    valid = read_json(splits_dir / "valid_triples.json")
    test = read_json(splits_dir / "test_triples.json")

    evaluation_set = read_json(
        PROCESSED_DIR
        / "evaluation_sets/test_random_neg50_seed42.json"
    )

    # ---------------------------------------------------------
    # Global graph
    # ---------------------------------------------------------

    assert len(graph) == EXPECTED["graph_triples"]
    assert len(entity2id) == EXPECTED["entities"]
    assert len(rel2id) == EXPECTED["relations"]

    target_df = graph[graph["rel"] == TARGET_RELATION]
    non_target_df = graph[graph["rel"] != TARGET_RELATION]

    assert len(target_df) == EXPECTED["target_triples"]

    # ---------------------------------------------------------
    # Frozen target splits
    # ---------------------------------------------------------

    assert len(train) == EXPECTED["train_target"]
    assert len(valid) == EXPECTED["valid_target"]
    assert len(test) == EXPECTED["test_target"]

    train_keys = {triple_key(row) for row in train}
    valid_keys = {triple_key(row) for row in valid}
    test_keys = {triple_key(row) for row in test}

    assert train_keys.isdisjoint(valid_keys)
    assert train_keys.isdisjoint(test_keys)
    assert valid_keys.isdisjoint(test_keys)

    target_keys = {
        (str(row.src), str(row.rel), str(row.dst))
        for row in target_df.itertuples(index=False)
    }

    assert train_keys | valid_keys | test_keys == target_keys

    for rows in (train, valid, test):
        assert all(row["rel"] == TARGET_RELATION for row in rows)

    # ---------------------------------------------------------
    # Candidate pool
    # ---------------------------------------------------------

    candidate_drugs = sorted(
        entity
        for entity, entity_type in entity_types.items()
        if entity_type == "DRUG"
    )

    assert len(candidate_drugs) == EXPECTED["candidate_drugs"]

    candidate_ids = {
        int(entity2id[entity])
        for entity in candidate_drugs
    }

    # Every positive target candidate must be a Drug.
    for row in train + valid + test:
        assert int(row["src_id"]) in candidate_ids
        assert entity_types[row["src"]] == "DRUG"
        assert entity_types[row["dst"]] == "DISEASE"

    # ---------------------------------------------------------
    # Frozen 1-vs-50 evaluation set
    # ---------------------------------------------------------

    metadata = evaluation_set["metadata"]
    examples = evaluation_set["examples"]

    assert metadata["task_name"] == "disease_to_drug"
    assert metadata["target_relation"] == TARGET_RELATION
    assert metadata["candidate_entity_type"] == "DRUG"
    assert metadata["query_entity_type"] == "DISEASE"
    assert metadata["num_negatives"] == 50
    assert metadata["candidate_pool_size"] == EXPECTED["candidate_drugs"]
    assert len(examples) == EXPECTED["test_target"]

    for example in examples:
        negatives = example["negative_candidates"]
        negative_ids = example["negative_candidate_ids"]

        assert len(negatives) == 50
        assert len(negative_ids) == 50
        assert len(set(negatives)) == 50

        assert example["positive_candidate"] not in negatives
        assert example["relation"] == TARGET_RELATION

        assert entity_types[example["query"]] == "DISEASE"
        assert entity_types[example["positive_candidate"]] == "DRUG"

        assert all(
            entity_types[candidate] == "DRUG"
            for candidate in negatives
        )

    # ---------------------------------------------------------
    # PyKEEN training graph
    #
    # Same information available during R-GCN training:
    #   INDICATION train
    #   + all non-target context relations
    #
    # Validation/test INDICATION triples remain excluded.
    # ---------------------------------------------------------

    indexed_graph = pd.read_csv(
        PROCESSED_DIR / "triples_indexed.csv",
        dtype=str,
    )

    assert len(indexed_graph) == EXPECTED["graph_triples"]

    non_target_indexed = indexed_graph[
        indexed_graph["rel"] != TARGET_RELATION
    ]

    assert len(non_target_indexed) == len(non_target_df)

    non_target_rows = non_target_indexed[
        ["src_id", "rel_id", "dst_id"]
    ].astype(int).values.tolist()

    train_graph_rows = [
        [
            int(row["src_id"]),
            int(row["rel_id"]),
            int(row["dst_id"]),
        ]
        for row in train
    ] + non_target_rows

    assert len(train_graph_rows) == EXPECTED["train_graph_triples"]

    train_factory = TriplesFactory(
        mapped_triples=torch.as_tensor(
            train_graph_rows,
            dtype=torch.long,
        ),
        entity_to_id={
            str(entity): int(index)
            for entity, index in entity2id.items()
        },
        relation_to_id={
            str(relation): int(index)
            for relation, index in rel2id.items()
        },
        create_inverse_triples=False,
    )

    valid_factory = TriplesFactory(
        mapped_triples=mapped_tensor(valid),
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
        create_inverse_triples=False,
    )

    test_factory = TriplesFactory(
        mapped_triples=mapped_tensor(test),
        entity_to_id=train_factory.entity_to_id,
        relation_to_id=train_factory.relation_to_id,
        create_inverse_triples=False,
    )

    # Mapping preservation is mandatory.
    assert dict(train_factory.entity_to_id) == {
        str(k): int(v) for k, v in entity2id.items()
    }
    assert dict(train_factory.relation_to_id) == {
        str(k): int(v) for k, v in rel2id.items()
    }

    assert train_factory.num_entities == EXPECTED["entities"]
    assert train_factory.num_relations == EXPECTED["relations"]

    assert train_factory.num_triples == EXPECTED["train_graph_triples"]
    assert valid_factory.num_triples == EXPECTED["valid_target"]
    assert test_factory.num_triples == EXPECTED["test_target"]

    print()
    print("Graph triples:       ", len(graph))
    print("Entities:            ", len(entity2id))
    print("Relations:           ", len(rel2id))
    print("INDICATION total:    ", len(target_df))
    print()
    print("Target train:        ", len(train))
    print("Target valid:        ", len(valid))
    print("Target test:         ", len(test))
    print()
    print("Training KG PyKEEN:  ", train_factory.num_triples)
    print("Validation PyKEEN:   ", valid_factory.num_triples)
    print("Test PyKEEN:         ", test_factory.num_triples)
    print("Drug candidates:     ", len(candidate_drugs))
    print("1-vs-50 examples:    ", len(examples))
    print()
    print("Entity mapping:       PRESERVED")
    print("Relation mapping:     PRESERVED")
    print("Split overlap:        NONE")
    print("Validation leakage:   NONE")
    print("Test leakage:         NONE")
    print()
    print("STATUS: CLEAN")


if __name__ == "__main__":
    main()

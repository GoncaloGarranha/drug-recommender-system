from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import torch

from models.shared.evaluation import filtered_rank
from models.shared.evaluate_negative_set import (
    evaluate_score_sets,
    load_evaluation_set,
)
from models.shared.metrics import compute_ranking_metrics

from .data import TARGET_RELATION


def build_known_indications(
    train_rows: list[dict[str, Any]],
    valid_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> dict[int, set[int]]:
    """
    disease_id -> set of all known positive drug_ids.

    Used only for filtered evaluation.
    """
    known: dict[int, set[int]] = defaultdict(set)

    for row in train_rows + valid_rows + test_rows:
        known[int(row["dst_id"])].add(int(row["src_id"]))

    return known


def evaluate_full_ranking(
    model,
    data: dict[str, Any],
    split: str = "test",
) -> dict[str, float]:
    """
    Disease -> Drug filtered ranking over the full Drug candidate pool.
    """
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be 'train', 'valid' or 'test'")

    rows = data[f"{split}_rows"]

    relation_id = int(
        data["training"].relation_to_id[TARGET_RELATION]
    )

    candidate_ids = data["candidate_ids"].to(model.device)
    candidate_list = candidate_ids.detach().cpu().tolist()

    known_positives = build_known_indications(
        data["train_rows"],
        data["valid_rows"],
        data["test_rows"],
    )

    # For head prediction:
    #
    #   (?, INDICATION, Disease)
    #
    # PyKEEN rt_batch = [relation_id, tail_id]
    rt_batch = torch.tensor(
        [
            [relation_id, int(row["dst_id"])]
            for row in rows
        ],
        dtype=torch.long,
        device=model.device,
    )

    model.eval()

    with torch.no_grad():
        scores = model.predict_h(
            rt_batch,
            heads=candidate_ids,
        )

    scores_np = scores.detach().cpu().numpy()

    assert scores_np.shape == (
        len(rows),
        len(candidate_list),
    )

    ranks: list[float] = []

    for index, row in enumerate(rows):
        disease_id = int(row["dst_id"])
        true_drug_id = int(row["src_id"])

        rank = filtered_rank(
            candidate_ids=candidate_list,
            scores=scores_np[index],
            true_candidate_id=true_drug_id,
            known_positive_ids=known_positives.get(
                disease_id,
                set(),
            ),
        )

        ranks.append(rank)

    metrics = compute_ranking_metrics(
        ranks,
        hits_k=(1, 3, 10),
    )

    return {
        **metrics,
        "num_examples": len(rows),
        "candidate_pool_size": len(candidate_list),
    }


def evaluate_fixed_1v50(
    model,
    data: dict[str, Any],
    split: str = "test",
) -> dict[str, float]:
    """
    Evaluate on the exact frozen 1-positive-vs-50-negative set.
    """
    if split not in {"valid", "test"}:
        raise ValueError("split must be 'valid' or 'test'")

    processed_dir = data["processed_dir"]

    evaluation_path = (
        processed_dir
        / "evaluation_sets"
        / f"{split}_random_neg50_seed42.json"
    )

    payload = load_evaluation_set(evaluation_path)

    metadata = payload["metadata"]
    examples = payload["examples"]

    rows = data[f"{split}_rows"]

    assert metadata["task_name"] == "disease_to_drug"
    assert metadata["target_relation"] == TARGET_RELATION
    assert int(metadata["num_negatives"]) == 50
    assert len(examples) == len(rows)

    relation_id = int(
        data["training"].relation_to_id[TARGET_RELATION]
    )

    positive_triples: list[list[int]] = []
    negative_triples: list[list[int]] = []

    for index, (example, row) in enumerate(
        zip(examples, rows)
    ):
        query_id = int(example["query_id"])
        positive_id = int(
            example["positive_candidate_id"]
        )
        negative_ids = [
            int(value)
            for value in example["negative_candidate_ids"]
        ]

        # Frozen set must correspond exactly to the split.
        assert query_id == int(row["dst_id"]), index
        assert positive_id == int(row["src_id"]), index
        assert len(negative_ids) == 50
        assert len(set(negative_ids)) == 50
        assert positive_id not in negative_ids

        # Original graph orientation:
        # Drug --INDICATION--> Disease
        positive_triples.append(
            [positive_id, relation_id, query_id]
        )

        negative_triples.extend(
            [
                [negative_id, relation_id, query_id]
                for negative_id in negative_ids
            ]
        )

    positive_tensor = torch.tensor(
        positive_triples,
        dtype=torch.long,
        device=model.device,
    )

    negative_tensor = torch.tensor(
        negative_triples,
        dtype=torch.long,
        device=model.device,
    )

    model.eval()

    with torch.no_grad():
        positive_scores = (
            model.predict_hrt(positive_tensor)
            .view(-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        negative_scores = (
            model.predict_hrt(negative_tensor)
            .view(len(examples), 50)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    metrics, _ = evaluate_score_sets(
        positive_scores=positive_scores,
        negative_scores=negative_scores,
    )

    return {
        **metrics,
        "num_examples": len(examples),
        "num_negatives": 50,
        "candidate_count_per_example": 51,
    }

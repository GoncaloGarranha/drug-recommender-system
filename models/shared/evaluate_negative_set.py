"""Shared utilities for fixed negative-set ranking evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from models.shared.metrics import compute_ranking_metrics


def load_evaluation_set(path: str | Path) -> dict[str, Any]:
    evaluation_path = Path(path)

    if not evaluation_path.is_file():
        raise FileNotFoundError(
            f"Evaluation set not found: {evaluation_path}"
        )

    payload = json.loads(
        evaluation_path.read_text(encoding="utf-8")
    )

    metadata = payload.get("metadata")
    examples = payload.get("examples")

    if not isinstance(metadata, dict):
        raise ValueError(
            "Evaluation set must contain a metadata object."
        )

    if not isinstance(examples, list):
        raise ValueError(
            "Evaluation set must contain an examples list."
        )

    num_negatives = int(metadata.get("num_negatives", 0))

    if num_negatives < 1:
        raise ValueError(
            "Evaluation metadata contains an invalid num_negatives."
        )

    for position, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(
                f"Invalid example at position {position}."
            )

        required = {
            "query_id",
            "positive_candidate_id",
            "negative_candidate_ids",
        }

        missing = required - set(example)

        if missing:
            raise ValueError(
                f"Example {position} is missing fields: "
                f"{sorted(missing)}"
            )

        negatives = list(
            map(int, example["negative_candidate_ids"])
        )

        if len(negatives) != num_negatives:
            raise ValueError(
                f"Example {position} has {len(negatives)} negatives; "
                f"expected {num_negatives}."
            )

        if len(set(negatives)) != num_negatives:
            raise ValueError(
                f"Example {position} contains duplicate negatives."
            )

        positive = int(example["positive_candidate_id"])

        if positive in negatives:
            raise ValueError(
                f"Example {position} contains the positive candidate "
                "among its negatives."
            )

    return payload


def tie_aware_rank(
    positive_score: float,
    negative_scores: Sequence[float] | np.ndarray,
) -> float:
    """Calculate the positive candidate's average rank under score ties."""

    negatives = np.asarray(
        negative_scores,
        dtype=np.float64,
    )

    better = int(np.sum(negatives > positive_score))
    equal = int(np.sum(negatives == positive_score))

    return 1.0 + better + equal / 2.0


def average_precision_binary(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    positive_count = int(labels.sum())

    if positive_count == 0:
        return float("nan")

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_labels = labels[order]
    cumulative_positives = np.cumsum(sorted_labels)
    ranks = np.arange(
        1,
        len(sorted_labels) + 1,
    )

    precision = cumulative_positives / ranks

    return float(
        precision[sorted_labels == 1].sum()
        / positive_count
    )


def roc_auc_binary(
    labels: np.ndarray,
    scores: np.ndarray,
) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    positive_count = int(labels.sum())
    negative_count = int(len(labels) - positive_count)

    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(
        scores,
        kind="mergesort",
    )

    sorted_scores = scores[order]
    ranks = np.empty(
        len(scores),
        dtype=np.float64,
    )

    start = 0

    while start < len(scores):
        end = start + 1

        while (
            end < len(scores)
            and sorted_scores[end] == sorted_scores[start]
        ):
            end += 1

        average_rank = (
            start + 1 + end
        ) / 2.0

        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(
        ranks[labels == 1].sum()
    )

    auc = (
        positive_rank_sum
        - positive_count * (positive_count + 1) / 2.0
    ) / (
        positive_count * negative_count
    )

    return float(auc)


def evaluate_score_sets(
    positive_scores: Sequence[float] | np.ndarray,
    negative_scores: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[dict[str, float], list[float]]:
    """Evaluate aligned positive scores and fixed negative-score matrices."""

    positives = np.asarray(
        positive_scores,
        dtype=np.float64,
    )

    negatives = np.asarray(
        negative_scores,
        dtype=np.float64,
    )

    if positives.ndim != 1:
        raise ValueError(
            "positive_scores must be one-dimensional."
        )

    if negatives.ndim != 2:
        raise ValueError(
            "negative_scores must be two-dimensional."
        )

    if len(positives) != negatives.shape[0]:
        raise ValueError(
            "Positive and negative score counts do not match."
        )

    ranks = [
        tie_aware_rank(
            positive_score=float(positives[index]),
            negative_scores=negatives[index],
        )
        for index in range(len(positives))
    ]

    ranking_metrics = compute_ranking_metrics(ranks)

    labels = np.concatenate(
        [
            np.ones(
                len(positives),
                dtype=np.int64,
            ),
            np.zeros(
                negatives.size,
                dtype=np.int64,
            ),
        ]
    )

    scores = np.concatenate(
        [
            positives,
            negatives.reshape(-1),
        ]
    )

    metrics = {
        **ranking_metrics,
        "auc_roc": roc_auc_binary(
            labels,
            scores,
        ),
        "average_precision": average_precision_binary(
            labels,
            scores,
        ),
    }

    return metrics, ranks


def build_metrics_payload(
    *,
    metrics: dict[str, float],
    metadata: dict[str, Any],
    split: str,
    evaluation_set_path: str | Path,
    example_count: int,
) -> dict[str, Any]:
    num_negatives = int(
        metadata["num_negatives"]
    )

    return {
        **metrics,
        "split": split,
        "protocol": (
            f"1_positive_vs_{num_negatives}_negatives"
        ),
        "negative_mode": str(
            metadata.get("negative_mode", "random")
        ),
        "num_negatives": num_negatives,
        "candidate_count_per_example": (
            num_negatives + 1
        ),
        "num_positives": int(example_count),
        "num_scored_pairs": int(
            example_count * (num_negatives + 1)
        ),
        "seed": int(metadata["seed"]),
        "evaluation_set": str(
            Path(evaluation_set_path).resolve()
        ),
        "tie_handling": "average_rank",
    }

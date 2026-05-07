from __future__ import annotations

from typing import Iterable, List, Sequence, Set

import numpy as np


def filtered_rank(
    candidate_ids: Sequence[int],
    scores: Sequence[float],
    true_candidate_id: int,
    known_positive_ids: Set[int],
) -> float:
    """Compute a filtered rank for one query using tie-aware ranking."""
    score_array = np.array(scores, dtype=np.float64).copy()
    candidate_array = np.array(candidate_ids, dtype=np.int64)

    for idx, candidate_id in enumerate(candidate_array):
        if candidate_id == true_candidate_id:
            continue
        if int(candidate_id) in known_positive_ids:
            score_array[idx] = -1e9

    true_mask = candidate_array == true_candidate_id
    if not np.any(true_mask):
        raise ValueError("True candidate id not present in candidate pool.")

    true_score = float(score_array[true_mask][0])
    num_better = int(np.sum(score_array > true_score))
    num_equal = int(np.sum(score_array == true_score))
    return 1 + num_better + (num_equal - 1) / 2.0


def filtered_ranks_from_batches(
    candidate_ids: Sequence[int],
    batch_scores: Iterable[Sequence[float]],
    true_candidate_ids: Sequence[int],
    known_positive_ids_per_query: Sequence[Set[int]],
) -> List[float]:
    """Compute filtered ranks for a batch of independent queries."""
    if len(true_candidate_ids) != len(known_positive_ids_per_query):
        raise ValueError("Mismatched lengths between true candidates and known positives.")

    ranks: List[float] = []
    for scores, true_id, known_ids in zip(batch_scores, true_candidate_ids, known_positive_ids_per_query):
        ranks.append(
            filtered_rank(
                candidate_ids=candidate_ids,
                scores=scores,
                true_candidate_id=int(true_id),
                known_positive_ids=known_ids,
            )
        )
    return ranks

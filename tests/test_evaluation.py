from __future__ import annotations

import unittest

from models.shared.evaluation import filtered_rank, filtered_ranks_from_batches


class FilteredRankingTests(unittest.TestCase):
    def test_known_positive_with_higher_score_is_filtered(self) -> None:
        rank = filtered_rank(
            candidate_ids=[10, 20, 30],
            scores=[0.9, 0.8, 0.7],
            true_candidate_id=20,
            known_positive_ids={10, 20},
        )

        self.assertEqual(rank, 1.0)

    def test_ties_use_average_rank(self) -> None:
        rank = filtered_rank(
            candidate_ids=[10, 20, 30],
            scores=[0.8, 0.8, 0.2],
            true_candidate_id=20,
            known_positive_ids={20},
        )

        self.assertEqual(rank, 1.5)

    def test_missing_true_candidate_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not present"):
            filtered_rank(
                candidate_ids=[10, 20],
                scores=[0.9, 0.8],
                true_candidate_id=30,
                known_positive_ids=set(),
            )

    def test_batch_length_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Mismatched lengths"):
            filtered_ranks_from_batches(
                candidate_ids=[10, 20],
                batch_scores=[[0.9, 0.8]],
                true_candidate_ids=[10, 20],
                known_positive_ids_per_query=[{10}, {20}],
            )


if __name__ == "__main__":
    unittest.main()

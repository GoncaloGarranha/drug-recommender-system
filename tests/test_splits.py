from __future__ import annotations

import unittest

from models.shared.splits import build_known_positives, split_target_triples


def _row(index: int, relation: str = "INDICATION") -> dict[str, str]:
    return {
        "src": f"DRUG:{index}",
        "rel": relation,
        "dst": f"DOID:{index % 3}",
        "src_id": str(index),
        "rel_id": "0" if relation == "INDICATION" else "1",
        "dst_id": str(100 + index % 3),
    }


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.target_rows = [_row(index) for index in range(10)]
        self.context_row = _row(99, relation="HAS_INGREDIENT")
        self.triples = self.target_rows + [self.context_row]

    def test_random_split_is_reproducible_and_disjoint(self) -> None:
        first = split_target_triples(
            self.triples, "INDICATION", 42, 0.8, 0.1, 0.1
        )
        second = split_target_triples(
            self.triples, "INDICATION", 42, 0.8, 0.1, 0.1
        )

        self.assertEqual(first, second)
        train, valid, test, context = first
        train_keys = {(r["src"], r["rel"], r["dst"]) for r in train}
        valid_keys = {(r["src"], r["rel"], r["dst"]) for r in valid}
        test_keys = {(r["src"], r["rel"], r["dst"]) for r in test}

        self.assertTrue(train_keys.isdisjoint(valid_keys))
        self.assertTrue(train_keys.isdisjoint(test_keys))
        self.assertTrue(valid_keys.isdisjoint(test_keys))
        self.assertEqual(len(train) + len(valid) + len(test), 10)
        self.assertEqual(context, [self.context_row])

    def test_invalid_ratios_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum to 1.0"):
            split_target_triples(
                self.triples, "INDICATION", 42, 0.8, 0.2, 0.2
            )

    def test_known_positives_follow_task_direction(self) -> None:
        disease_to_drug = build_known_positives(
            self.target_rows[:2], task_name="disease_to_drug"
        )
        drug_to_disease = build_known_positives(
            self.target_rows[:2], task_name="drug_to_disease"
        )

        self.assertIn(0, disease_to_drug[100])
        self.assertIn(100, drug_to_disease[0])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from models.shared.ranking import build_candidate_pool, build_training_graph


class CandidatePoolTests(unittest.TestCase):
    def test_pool_contains_only_configured_entity_type(self) -> None:
        pool = build_candidate_pool(
            entity_types={"DRUG:1": "DRUG", "DRUG:2": "DRUG", "DOID:1": "DISEASE"},
            entity2id={"DRUG:1": 1, "DRUG:2": 2, "DOID:1": 3},
            candidate_entity_type="DRUG",
            device=torch.device("cpu"),
        )

        self.assertEqual(pool.tolist(), [1, 2])

    def test_explicit_pool_must_contain_ground_truth(self) -> None:
        with self.assertRaisesRegex(ValueError, "Ground-truth"):
            build_candidate_pool(
                entity_types={"DRUG:1": "DRUG", "DRUG:2": "DRUG"},
                entity2id={"DRUG:1": 1, "DRUG:2": 2},
                candidate_entity_type="DRUG",
                device=torch.device("cpu"),
                required_entity_ids={2},
                explicit_entity_ids={"DRUG:1"},
            )


class TrainingGraphTests(unittest.TestCase):
    def test_inverse_edges_reverse_endpoints_and_offset_relation(self) -> None:
        row = {"src_id": "1", "rel_id": "2", "dst_id": "3", "rel": "INDICATION"}
        graph, edge_types = build_training_graph(
            train_target=[row],
            non_target_triples=[],
            context_relations="ALL",
            num_nodes=4,
            device=torch.device("cpu"),
            num_relations=5,
            add_inverse_edges=True,
        )

        src, dst = graph.edges()
        self.assertEqual(src.tolist(), [1, 3])
        self.assertEqual(dst.tolist(), [3, 1])
        self.assertEqual(edge_types.tolist(), [2, 7])


if __name__ == "__main__":
    unittest.main()

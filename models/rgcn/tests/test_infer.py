from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.rgcn.src.infer import (
    build_split_positive_maps,
    eligible_candidate_positions,
    filtered_candidate_rank,
    load_entity_aliases,
    load_entity_labels,
    original_entity_names,
    preferred_entity_label,
    relation_candidates_for_query,
    resolve_query_entity,
    statuses_for_query,
)
from models.rgcn.src.inference.query_resolver import (
    QueryNotFoundError,
    QuerySelectionCancelled,
)


class EntityMetadataTests(unittest.TestCase):
    def test_three_column_entity_table_uses_label_header(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entity_table.csv"
            path.write_text(
                "entity_id,entity_type,label\n"
                "DOID:9352,DISEASE,type 2 diabetes\n",
                encoding="utf-8",
            )

            self.assertEqual(
                load_entity_labels(path),
                {"DOID:9352": "type 2 diabetes"},
            )

    def test_drug_original_name_is_preferred_over_numeric_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aliases.csv"
            path.write_text(
                "alias,entity_id,alias_type\n"
                "Synjardy,CHEBI:90875,DRUG_NAME_ORIGINAL\n"
                "SYNJARDY,CHEBI:90875,DRUG_NAME_NORM\n",
                encoding="utf-8",
            )
            aliases = load_entity_aliases(path)

            self.assertEqual(
                preferred_entity_label(
                    "CHEBI:90875",
                    "DRUG",
                    {"CHEBI:90875": "90875"},
                    aliases,
                ),
                "Synjardy",
            )

    def test_distinct_original_names_are_deduplicated(self) -> None:
        aliases = {
            "DRUGBANK:1": [
                ("Product A", "DRUG_NAME_ORIGINAL"),
                ("PRODUCT A", "DRUG_NAME_NORM"),
                ("Product B", "DRUG_NAME_ORIGINAL"),
                ("product a", "DRUG_NAME_ORIGINAL"),
            ]
        }

        self.assertEqual(
            original_entity_names("DRUGBANK:1", "DRUG", aliases),
            ["Product A", "Product B"],
        )


class QueryResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.entity2id = {"DOID:9744": 1, "DOID:9352": 2}
        self.entity_types = {"DOID:9744": "DISEASE", "DOID:9352": "DISEASE"}
        self.entity_labels = {
            "DOID:9744": "type 1 diabetes mellitus",
            "DOID:9352": "type 2 diabetes",
        }

    def test_exact_label_match_is_case_insensitive(self) -> None:
        self.assertEqual(
            resolve_query_entity(
                query="TYPE 2 DIABETES",
                entity_id=None,
                expected_entity_type="DISEASE",
                entity2id=self.entity2id,
                entity_types=self.entity_types,
                entity_labels=self.entity_labels,
            ),
            ("DOID:9352", 2),
        )

    def test_ambiguous_partial_match_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Possible matches"):
            resolve_query_entity(
                query="type",
                entity_id=None,
                expected_entity_type="DISEASE",
                entity2id=self.entity2id,
                entity_types=self.entity_types,
                entity_labels=self.entity_labels,
            )

    def test_interactive_typo_can_select_a_numbered_match(self) -> None:
        messages: list[str] = []
        entity2id = {
            "DOID:2841": 1,
            "DOID:0080810": 2,
            "DOID:1398": 3,
        }
        entity_types = {entity_id: "DISEASE" for entity_id in entity2id}
        entity_labels = {
            "DOID:2841": "asthma",
            "DOID:0080810": "acute asthma",
            "DOID:1398": "parasitemia",
        }

        resolved = resolve_query_entity(
            query="astma",
            entity_id=None,
            expected_entity_type="DISEASE",
            entity2id=entity2id,
            entity_types=entity_types,
            entity_labels=entity_labels,
            interactive=True,
            input_fn=lambda _: "2",
            output_fn=messages.append,
        )

        self.assertEqual(resolved, ("DOID:0080810", 2))
        self.assertTrue(any("No exact match" in message for message in messages))
        self.assertTrue(any("Selected disease" in message for message in messages))
        self.assertFalse(any("parasitemia" in message for message in messages))

    def test_interactive_selection_rejects_invalid_number_then_accepts(self) -> None:
        selections = iter(["9", "1"])
        messages: list[str] = []

        resolved = resolve_query_entity(
            query="type",
            entity_id=None,
            expected_entity_type="DISEASE",
            entity2id=self.entity2id,
            entity_types=self.entity_types,
            entity_labels=self.entity_labels,
            interactive=True,
            input_fn=lambda _: next(selections),
            output_fn=messages.append,
        )

        self.assertIn(resolved[0], self.entity2id)
        self.assertIn(
            "Invalid selection. Please enter one of the listed options.",
            messages,
        )

    def test_interactive_selection_can_be_cancelled(self) -> None:
        with self.assertRaises(QuerySelectionCancelled):
            resolve_query_entity(
                query="type",
                entity_id=None,
                expected_entity_type="DISEASE",
                entity2id=self.entity2id,
                entity_types=self.entity_types,
                entity_labels=self.entity_labels,
                interactive=True,
                input_fn=lambda _: "0",
                output_fn=lambda _: None,
            )

    def test_interactive_selection_can_refine_the_query(self) -> None:
        selections = iter(["r", "type 2 diabetes"])

        self.assertEqual(
            resolve_query_entity(
                query="type",
                entity_id=None,
                expected_entity_type="DISEASE",
                entity2id=self.entity2id,
                entity_types=self.entity_types,
                entity_labels=self.entity_labels,
                interactive=True,
                input_fn=lambda _: next(selections),
                output_fn=lambda _: None,
            ),
            ("DOID:9352", 2),
        )

    def test_interactive_selection_shows_at_most_five_matches(self) -> None:
        entity2id = {f"DOID:{index}": index for index in range(1, 8)}
        entity_types = {entity_id: "DISEASE" for entity_id in entity2id}
        entity_labels = {
            entity_id: f"asthma variant {index}"
            for index, entity_id in enumerate(entity2id, start=1)
        }
        messages: list[str] = []

        resolved_entity_id, _ = resolve_query_entity(
            query="asthm",
            entity_id=None,
            expected_entity_type="DISEASE",
            entity2id=entity2id,
            entity_types=entity_types,
            entity_labels=entity_labels,
            interactive=True,
            input_fn=lambda _: "1",
            output_fn=messages.append,
        )

        self.assertIn(resolved_entity_id, entity2id)
        formatted_matches = next(message for message in messages if "[1]" in message)
        self.assertEqual(len(formatted_matches.splitlines()), 5)
        self.assertNotIn("[6]", formatted_matches)
        self.assertTrue(any("closest of 7" in message for message in messages))

    def test_unknown_query_has_a_concise_graph_scope_message(self) -> None:
        with self.assertRaisesRegex(
            QueryNotFoundError,
            "Only diseases represented in the current graph can be queried",
        ):
            resolve_query_entity(
                query="headache",
                entity_id=None,
                expected_entity_type="DISEASE",
                entity2id=self.entity2id,
                entity_types=self.entity_types,
                entity_labels=self.entity_labels,
            )


class SplitStatusTests(unittest.TestCase):
    @staticmethod
    def _row(candidate_id: int, query_id: int) -> dict[str, str]:
        return {
            "src_id": str(candidate_id),
            "rel_id": "3",
            "dst_id": str(query_id),
            "src": f"DRUG:{candidate_id}",
            "rel": "INDICATION",
            "dst": f"DOID:{query_id}",
        }

    def test_statuses_preserve_split_provenance(self) -> None:
        split_maps = build_split_positive_maps(
            task_name="disease_to_drug",
            train_target=[self._row(1, 10)],
            valid_target=[self._row(2, 10)],
            test_target=[self._row(3, 10)],
        )

        self.assertEqual(
            statuses_for_query(split_maps, 10),
            {1: "TRAIN", 2: "VALIDATION", 3: "TEST"},
        )

    def test_duplicate_association_across_splits_is_rejected(self) -> None:
        split_maps = build_split_positive_maps(
            task_name="disease_to_drug",
            train_target=[self._row(1, 10)],
            valid_target=[self._row(1, 10)],
            test_target=[],
        )

        with self.assertRaisesRegex(ValueError, "more than one split"):
            statuses_for_query(split_maps, 10)

    def test_contraindications_are_annotated_without_being_removed(self) -> None:
        contraindication = self._row(4, 10)
        contraindication["rel"] = "CONTRAINDICATION"

        self.assertEqual(
            relation_candidates_for_query(
                triples=[contraindication],
                task_name="disease_to_drug",
                query_node_id=10,
                relation="CONTRAINDICATION",
            ),
            {4},
        )

    def test_exploration_excludes_known_pairs(self) -> None:
        positions, contraindication_exclusions = eligible_candidate_positions(
            candidate_pool_ids=[1, 2, 3, 4],
            candidate_statuses={1: "TRAIN", 2: "TEST"},
            contraindicated_candidates={3},
            mode="exploration",
            exclude_contraindications=False,
        )

        self.assertEqual(positions, [2, 3])
        self.assertEqual(contraindication_exclusions, 0)

    def test_contraindication_filter_is_optional(self) -> None:
        positions, contraindication_exclusions = eligible_candidate_positions(
            candidate_pool_ids=[1, 2, 3, 4],
            candidate_statuses={1: "TRAIN"},
            contraindicated_candidates={2, 4},
            mode="exploration",
            exclude_contraindications=True,
        )

        self.assertEqual(positions, [2])
        self.assertEqual(contraindication_exclusions, 2)

    def test_filtered_candidate_rank_matches_evaluation_definition(self) -> None:
        self.assertEqual(
            filtered_candidate_rank(
                candidate_pool_ids=[10, 20, 30],
                scores=[9.0, 8.0, 7.0],
                candidate_id=20,
                known_candidate_ids={10, 20},
            ),
            1.0,
        )


if __name__ == "__main__":
    unittest.main()

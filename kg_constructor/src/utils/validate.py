from __future__ import annotations

from collections import Counter
from typing import Dict, List, Mapping, Sequence


# -----------------------------------------------------------------------------
# Generic validation helpers
# -----------------------------------------------------------------------------

def _normalize_columns(columns: Sequence[str]) -> list[str]:
    return [str(column).strip() for column in columns]


def validate_required_columns(
    rows: list[Mapping[str, object]],
    required_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Ensure that a tabular dataset contains the required columns."""
    expected = _normalize_columns(required_columns)

    if not rows:
        raise ValueError(f"{dataset_name} is empty or contains no data rows.")

    available_columns = {str(column).strip() for column in rows[0].keys()}
    missing_columns = [column for column in expected if column not in available_columns]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing_columns}. "
            f"Available columns: {sorted(available_columns)}"
        )


def validate_no_empty_required_values(
    rows: list[Mapping[str, object]],
    required_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Ensure that required columns do not contain empty values."""
    normalized_columns = _normalize_columns(required_columns)

    for index, row in enumerate(rows, start=1):
        for column in normalized_columns:
            value = row.get(column)
            if value is None or str(value).strip() == "":
                raise ValueError(
                    f"{dataset_name} has an empty value in column '{column}' at row {index}."
                )


def count_duplicate_rows(
    rows: list[Mapping[str, object]],
    key_columns: Sequence[str],
) -> int:
    """Count duplicate rows according to a subset of key columns."""
    normalized_columns = _normalize_columns(key_columns)
    keys = [
        tuple(str(row.get(column, "")).strip() for column in normalized_columns)
        for row in rows
    ]

    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def validate_no_duplicate_keys(
    rows: list[Mapping[str, object]],
    key_columns: Sequence[str],
    dataset_name: str,
) -> None:
    """Raise an error when duplicate rows exist for the selected key columns."""
    duplicate_count = count_duplicate_rows(rows, key_columns)

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} contains {duplicate_count} duplicate rows based on {list(key_columns)}."
        )


# -----------------------------------------------------------------------------
# Existing validators used by benchmark, triples and filtering steps
# -----------------------------------------------------------------------------

def validate_triples_columns(
    rows: list[Mapping[str, object]],
    expected_columns: Sequence[str],
) -> None:
    """Ensure every triple row exposes the expected columns."""
    validate_required_columns(rows, expected_columns, "triples dataset")


def validate_no_empty_values(
    rows: list[Mapping[str, object]],
    columns: Sequence[str],
) -> None:
    """Ensure the selected columns contain non-empty values."""
    validate_no_empty_required_values(rows, columns, "dataset")


def validate_split_values(
    rows: list[Mapping[str, object]],
    split_column: str = "split",
    allowed_values: Sequence[str] = ("train", "valid", "test"),
) -> None:
    """Ensure benchmark split labels are valid."""
    allowed = {str(value).strip() for value in allowed_values}

    for index, row in enumerate(rows):
        value = row.get(split_column)
        if value is None:
            raise ValueError(f"Missing split value at row {index}.")

        split_value = str(value).strip()
        if split_value not in allowed:
            raise ValueError(
                f"Invalid split value at row {index}: '{split_value}'. "
                f"Allowed values: {sorted(allowed)}"
            )


def validate_relation_counts(
    rows: list[Mapping[str, object]],
    relation_column: str = "rel",
    min_unique_relations: int = 1,
) -> Counter:
    """Return relation frequencies and ensure the dataset is not degenerate."""
    counts: Counter = Counter()

    for row in rows:
        value = row.get(relation_column)
        if value is None:
            continue
        relation = str(value).strip()
        if relation:
            counts[relation] += 1

    if len(counts) < min_unique_relations:
        raise ValueError(
            f"Expected at least {min_unique_relations} unique relations in column "
            f"'{relation_column}', got {len(counts)}."
        )

    return counts


# -----------------------------------------------------------------------------
# KG Constructor interim output validators
# -----------------------------------------------------------------------------

def validate_drugs_raw(rows: list[Mapping[str, object]]) -> Dict[str, object]:
    """Validate drugs_raw.csv and return audit metrics."""
    validate_required_columns(
        rows,
        ["source", "name_original", "name_norm", "drugbank_id", "chebi_id", "file"],
        "drugs_raw.csv",
    )
    validate_no_empty_required_values(rows, ["source", "name_norm", "file"], "drugs_raw.csv")

    without_drugbank = sum(1 for row in rows if not str(row.get("drugbank_id", "")).strip())
    without_chebi = sum(1 for row in rows if not str(row.get("chebi_id", "")).strip())
    without_external_id = sum(
        1
        for row in rows
        if not str(row.get("drugbank_id", "")).strip()
        and not str(row.get("chebi_id", "")).strip()
    )

    return {
        "total_rows": len(rows),
        "rows_without_drugbank_id": without_drugbank,
        "rows_without_chebi_id": without_chebi,
        "rows_without_external_id": without_external_id,
        "duplicate_rows_by_name_and_ids": count_duplicate_rows(
            rows,
            ["name_norm", "drugbank_id", "chebi_id"],
        ),
    }


def validate_ingredients_raw(rows: list[Mapping[str, object]]) -> Dict[str, object]:
    """Validate ingredients_raw.csv and return audit metrics."""
    validate_required_columns(
        rows,
        ["source", "name_original", "name_norm", "unii_code", "file"],
        "ingredients_raw.csv",
    )
    validate_no_empty_required_values(rows, ["source", "file"], "ingredients_raw.csv")

    without_unii = sum(1 for row in rows if not str(row.get("unii_code", "")).strip())

    return {
        "total_rows": len(rows),
        "rows_without_unii_code": without_unii,
        "duplicate_rows_by_name_and_unii": count_duplicate_rows(rows, ["name_norm", "unii_code"]),
    }


def validate_diseases_raw(rows: list[Mapping[str, object]]) -> Dict[str, object]:
    """Validate diseases_raw.csv and return audit metrics."""
    validate_required_columns(rows, ["doid"], "diseases_raw.csv")
    validate_no_empty_required_values(rows, ["doid"], "diseases_raw.csv")

    invalid_doids = sum(
        1
        for row in rows
        if not str(row.get("doid", "")).strip().upper().startswith("DOID:")
    )

    if invalid_doids > 0:
        raise ValueError(f"diseases_raw.csv contains {invalid_doids} invalid DOID identifiers.")

    return {
        "total_rows": len(rows),
        "invalid_doids": invalid_doids,
        "duplicate_doids": count_duplicate_rows(rows, ["doid"]),
    }


def validate_edges_raw(rows: list[Mapping[str, object]]) -> Dict[str, object]:
    """Validate edges_raw.csv and return audit metrics."""
    validate_required_columns(
        rows,
        ["src", "rel", "dst", "source", "file"],
        "edges_raw.csv",
    )
    validate_no_empty_required_values(
        rows,
        ["src", "rel", "dst", "source", "file"],
        "edges_raw.csv",
    )

    relation_counts = validate_relation_counts(rows, relation_column="rel")

    return {
        "total_rows": len(rows),
        "duplicate_edges_by_src_rel_dst": count_duplicate_rows(rows, ["src", "rel", "dst"]),
        "relation_counts": dict(relation_counts),
    }


# -----------------------------------------------------------------------------
# KG Constructor mapping validators
# -----------------------------------------------------------------------------

def validate_entity_table(rows: list[Mapping[str, object]]) -> Dict[str, object]:
    """Validate entity_table.csv and return entity statistics."""
    validate_required_columns(rows, ["entity_id", "entity_type", "label"], "entity_table.csv")
    validate_no_empty_required_values(rows, ["entity_id", "entity_type"], "entity_table.csv")
    validate_no_duplicate_keys(rows, ["entity_id"], "entity_table.csv")

    type_counts = Counter(str(row.get("entity_type", "")).strip() for row in rows)

    return {
        "total_entities": len(rows),
        "entity_type_counts": dict(type_counts),
    }


def validate_aliases(rows: list[Mapping[str, object]]) -> Dict[str, object]:
    """Validate aliases.csv and return alias statistics."""
    validate_required_columns(rows, ["alias", "entity_id", "alias_type"], "aliases.csv")
    validate_no_empty_required_values(rows, ["alias", "entity_id", "alias_type"], "aliases.csv")

    alias_type_counts = Counter(str(row.get("alias_type", "")).strip() for row in rows)

    return {
        "total_aliases": len(rows),
        "duplicate_alias_rows": count_duplicate_rows(rows, ["alias", "entity_id", "alias_type"]),
        "alias_type_counts": dict(alias_type_counts),
    }


def validate_alias_entities_exist(
    aliases: list[Mapping[str, object]],
    entity_table: list[Mapping[str, object]],
) -> Dict[str, object]:
    """Check whether all aliases point to entities present in entity_table.csv."""
    entity_ids = {str(row.get("entity_id", "")).strip() for row in entity_table}
    missing = sum(
        1
        for row in aliases
        if str(row.get("entity_id", "")).strip() not in entity_ids
    )

    if missing > 0:
        raise ValueError(f"aliases.csv contains {missing} aliases pointing to missing entities.")

    return {
        "aliases_with_missing_entity": missing,
    }


# -----------------------------------------------------------------------------
# Cross-step consistency validators
# -----------------------------------------------------------------------------

def validate_triple_entities_exist(
    triples: list[Mapping[str, object]],
    entity_table: list[Mapping[str, object]],
) -> Dict[str, object]:
    """Check whether triple source and destination entities exist in entity_table.csv."""
    entity_ids = {str(row.get("entity_id", "")).strip() for row in entity_table}

    missing_src = sum(
        1
        for row in triples
        if str(row.get("src", "")).strip() not in entity_ids
    )
    missing_dst = sum(
        1
        for row in triples
        if str(row.get("dst", "")).strip() not in entity_ids
    )

    if missing_src > 0 or missing_dst > 0:
        raise ValueError(
            "Triples reference entities missing from entity_table.csv: "
            f"missing_src={missing_src}, missing_dst={missing_dst}."
        )

    return {
        "triples_with_missing_src_entity": missing_src,
        "triples_with_missing_dst_entity": missing_dst,
    }

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Sequence


def _normalize_columns(columns: Sequence[str]) -> list[str]:
	return [str(column).strip() for column in columns]


def validate_triples_columns(rows: list[Mapping[str, object]], expected_columns: Sequence[str]) -> None:
	"""Ensure every row exposes the expected columns."""
	expected = _normalize_columns(expected_columns)
	missing_rows = []

	for index, row in enumerate(rows):
		row_keys = {str(key).strip() for key in row.keys()}
		missing = [column for column in expected if column not in row_keys]
		if missing:
			missing_rows.append((index, missing))

	if missing_rows:
		first_index, missing = missing_rows[0]
		raise ValueError(
			f"Invalid triples schema at row {first_index}: missing columns {missing}. Expected columns: {expected}"
		)


def validate_no_empty_values(rows: list[Mapping[str, object]], columns: Sequence[str]) -> None:
	"""Ensure the selected columns contain non-empty values."""
	normalized_columns = _normalize_columns(columns)

	for index, row in enumerate(rows):
		for column in normalized_columns:
			value = row.get(column)
			if value is None or str(value).strip() == "":
				raise ValueError(f"Empty value found at row {index}, column '{column}'.")


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
				f"Invalid split value at row {index}: '{split_value}'. Allowed values: {sorted(allowed)}"
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
			f"Expected at least {min_unique_relations} unique relations in column '{relation_column}', got {len(counts)}."
		)

	return counts

from __future__ import annotations

from typing import Any, Dict, List, Set


def infer_entity_type(entity_id: str) -> str:
	"""
	Infer the entity type from its canonical prefix.
	
	Supports:
		- DRUGBANK:*, DRUG:* → "DRUG"
		- DOID:* → "DISEASE"
		- UNII:*, ING:* → "INGREDIENT"
		- Otherwise → "UNKNOWN"
	"""
	if entity_id.startswith("DRUGBANK:") or entity_id.startswith("DRUG:"):
		return "DRUG"
	if entity_id.startswith("DOID:"):
		return "DISEASE"
	if entity_id.startswith("UNII:") or entity_id.startswith("ING:"):
		return "INGREDIENT"
	return "UNKNOWN"


def validate_triples_schema(rows: List[Dict[str, str]]) -> None:
	"""
	Validate that triples have required columns (src, rel, dst).
	
	Raises:
		ValueError: If columns are missing or rows are empty.
	"""
	if not rows:
		raise ValueError("Input triple file is empty.")

	required_columns = {"src", "rel", "dst"}
	first_row_keys = set(rows[0].keys())

	missing_columns = required_columns - first_row_keys
	if missing_columns:
		raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


def validate_no_empty_values(rows: List[Dict[str, str]]) -> None:
	"""
	Validate that no triple has empty src, rel, or dst values.
	
	Raises:
		ValueError: If any triple has empty values.
	"""
	for row in rows:
		src = (row.get("src") or "").strip()
		rel = (row.get("rel") or "").strip()
		dst = (row.get("dst") or "").strip()

		if not src or not rel or not dst:
			raise ValueError("Found triple row with empty src, rel, or dst.")


def validate_allowed_relations(
	rows: List[Dict[str, str]],
	allowed_relations: Set[str],
) -> None:
	"""
	Validate that all triples use only allowed relations.
	
	Args:
		rows: List of triple dictionaries.
		allowed_relations: Set of relation names to allow.
	
	Raises:
		ValueError: If any triple uses a disallowed relation.
	"""
	disallowed = set()
	for row in rows:
		rel = (row.get("rel") or "").strip()
		if rel not in allowed_relations:
			disallowed.add(rel)
	
	if disallowed:
		raise ValueError(
			f"Found triples with disallowed relations: {sorted(disallowed)}"
		)


def validate_required_relation(
	rows: List[Dict[str, str]],
	required_relation: str,
) -> None:
	"""
	Validate that at least one triple uses the required relation.
	
	Args:
		rows: List of triple dictionaries.
		required_relation: The relation that must be present.
	
	Raises:
		ValueError: If no triples use the required relation.
	"""
	count = sum(1 for row in rows if (row.get("rel") or "").strip() == required_relation)
	
	if count == 0:
		raise ValueError(
			f"No triples found for required relation: {required_relation}"
		)


def validate_entity_types_coverage(
	entity_types: Dict[str, str],
	expected_types: Set[str],
) -> None:
	"""
	Validate that entity types contain expected entity type categories.
	
	Args:
		entity_types: Dictionary mapping entity IDs to their types.
		expected_types: Set of entity types that should be present.
	
	Raises:
		ValueError: If any expected type is missing.
	"""
	found_types = set(entity_types.values())
	missing = expected_types - found_types
	
	if missing:
		raise ValueError(
			f"Entity type coverage incomplete. Missing: {sorted(missing)}"
		)

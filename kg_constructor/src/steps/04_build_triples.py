"""
Step 04: Convert raw edges into canonical triples (src, rel, dst).

This step uses the canonical drug mapping generated in Step 03 to ensure
that all triples refer only to normalized canonical entities.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from kg_constructor.src.utils.log import get_logger
from kg_constructor.src.utils.validate import (
    validate_no_empty_values,
    validate_relation_counts,
    validate_triples_columns,
)

logger = get_logger(__name__)


def read_csv(path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file into a list of dictionaries.

    Args:
        path: Path to the CSV file.

    Returns:
        List of rows as dictionaries.
    """
    with path.open("r", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """
    Write rows to a CSV file.

    Args:
        path: Output CSV path.
        header: CSV header row.
        rows: Data rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)


def build_triples(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 04: Convert raw edges into canonical triples.

    This step reads the raw edge table and replaces each source drug name
    with its canonical entity identifier using the mapping generated in
    Step 03.

    Inputs:
        - data/output/interim/edges_raw.csv
        - data/output/mappings/drug_name_to_entity.json

    Output:
        - data/output/triples/triples.csv

    Args:
        cfg: Pipeline configuration dictionary. Kept for interface consistency.
        paths: Object containing pipeline paths. Must provide `output_dir`.
        force: If True, overwrite existing outputs.
    """
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    mappings_dir = output_dir / "mappings"
    triples_dir = output_dir / "triples"
    triples_dir.mkdir(parents=True, exist_ok=True)

    edges_csv = interim_dir / "edges_raw.csv"
    drug_map_json = mappings_dir / "drug_name_to_entity.json"
    triples_csv = triples_dir / "triples.csv"

    if not force and triples_csv.exists():
        logger.info("Triples output already exists. Skipping step (use --force to re-run).")
        return

    if not edges_csv.exists():
        raise FileNotFoundError("Missing edges_raw.csv. Run step 02_parse_sources first.")

    if not drug_map_json.exists():
        raise FileNotFoundError("Missing drug_name_to_entity.json. Run step 03_resolve_entities first.")

    edge_rows = read_csv(edges_csv)

    # Load the mapping from normalized drug names to canonical entity IDs
    drug_name_to_entity = json.loads(drug_map_json.read_text(encoding="utf-8"))

    triple_rows: List[List[str]] = []
    skipped_missing_mapping = 0
    skipped_invalid_rows = 0

    for row in edge_rows:
        src_name_norm = (row.get("src") or "").strip()
        relation = (row.get("rel") or "").strip()
        dst_entity = (row.get("dst") or "").strip()

        if not src_name_norm or not relation or not dst_entity:
            skipped_invalid_rows += 1
            continue

        src_entity = drug_name_to_entity.get(src_name_norm)
        if not src_entity:
            skipped_missing_mapping += 1
            continue

        triple_rows.append([src_entity, relation, dst_entity])

    # Deduplicate canonical triples: different drug name variants may resolve
    # to the same canonical entity_id, producing duplicate (src, rel, dst) rows.
    seen_triples: set = set()
    unique_triple_rows: List[List[str]] = []
    for row in triple_rows:
        key = tuple(row)
        if key not in seen_triples:
            seen_triples.add(key)
            unique_triple_rows.append(row)

    skipped_canonical_duplicates = len(triple_rows) - len(unique_triple_rows)
    triple_rows = unique_triple_rows

    triple_dict_rows = [
        {"src": src, "rel": rel, "dst": dst}
        for src, rel, dst in triple_rows
    ]
    validate_triples_columns(triple_dict_rows, ["src", "rel", "dst"])
    validate_no_empty_values(triple_dict_rows, ["src", "rel", "dst"])
    relation_counts = validate_relation_counts(triple_dict_rows)

    write_csv(triples_csv, ["src", "rel", "dst"], triple_rows)

    logger.info("Saved: %s", triples_csv)
    logger.info(
        "Triples: %d | Skipped (missing drug mapping): %d | Skipped (invalid row): %d | Skipped (canonical duplicates): %d",
        len(triple_rows),
        skipped_missing_mapping,
        skipped_invalid_rows,
        skipped_canonical_duplicates,
    )
    logger.info("Relation distribution: %s", dict(relation_counts))


# Alias used by the pipeline runner
run = build_triples


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"

    build_triples(cfg={}, paths=Paths(), force=True)
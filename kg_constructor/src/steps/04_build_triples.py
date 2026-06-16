"""
Step 04: Convert raw edges into canonical triples (src, rel, dst).

This step uses the canonical drug mapping generated in Step 03 to ensure
that all triples refer only to normalized canonical entities.

In addition to the training-oriented triples.csv file, this step also
generates triples_with_provenance.csv for auditability and traceability.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from kg_constructor.src.utils.io import read_csv, write_csv
from kg_constructor.src.utils.log import get_logger
from kg_constructor.src.utils.validate import (
    validate_no_empty_values,
    validate_relation_counts,
    validate_triples_columns,
)

logger = get_logger(__name__)


def build_triples(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 04: Convert raw edges into canonical triples.

    Inputs:
        - data/output/interim/edges_raw.csv
        - data/output/mappings/drug_name_to_entity.json

    Outputs:
        - data/output/triples/triples.csv
        - data/output/triples/triples_with_provenance.csv
        - data/output/reports/build_triples_report.json
    """
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    mappings_dir = output_dir / "mappings"
    triples_dir = output_dir / "triples"
    reports_dir = output_dir / "reports"

    triples_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    edges_csv = interim_dir / "edges_raw.csv"
    drug_map_json = mappings_dir / "drug_name_to_entity.json"
    triples_csv = triples_dir / "triples.csv"
    triples_with_provenance_csv = triples_dir / "triples_with_provenance.csv"
    build_report_json = reports_dir / "build_triples_report.json"

    output_files = [triples_csv, triples_with_provenance_csv, build_report_json]
    if not force and all(path.exists() for path in output_files):
        logger.info("Triples outputs already exist. Skipping step (use --force to re-run).")
        return

    if not edges_csv.exists():
        raise FileNotFoundError("Missing edges_raw.csv. Run step 02_parse_sources first.")

    if not drug_map_json.exists():
        raise FileNotFoundError("Missing drug_name_to_entity.json. Run step 03_resolve_entities first.")

    edge_rows = read_csv(edges_csv)
    drug_name_to_entity = json.loads(drug_map_json.read_text(encoding="utf-8"))

    triple_rows: List[List[str]] = []
    triple_provenance_rows: List[List[str]] = []
    skipped_missing_mapping = 0
    skipped_invalid_rows = 0

    for row in edge_rows:
        src_name_norm = (row.get("src") or "").strip()
        relation = (row.get("rel") or "").strip()
        dst_entity = (row.get("dst") or "").strip()
        source = (row.get("source") or "").strip()
        file_name = (row.get("file") or "").strip()

        if not src_name_norm or not relation or not dst_entity:
            skipped_invalid_rows += 1
            continue

        src_entity = drug_name_to_entity.get(src_name_norm)
        if not src_entity:
            skipped_missing_mapping += 1
            continue

        triple_rows.append([src_entity, relation, dst_entity])
        triple_provenance_rows.append([
            src_entity,
            relation,
            dst_entity,
            source,
            file_name,
        ])

    seen_triples: set = set()
    unique_triple_rows: List[List[str]] = []

    for row in triple_rows:
        key = tuple(row)
        if key not in seen_triples:
            seen_triples.add(key)
            unique_triple_rows.append(row)

    skipped_canonical_duplicates = len(triple_rows) - len(unique_triple_rows)
    triple_rows = unique_triple_rows

    seen_provenance_rows: set = set()
    unique_provenance_rows: List[List[str]] = []

    for row in triple_provenance_rows:
        key = tuple(row)
        if key not in seen_provenance_rows:
            seen_provenance_rows.add(key)
            unique_provenance_rows.append(row)

    triple_provenance_rows = unique_provenance_rows

    triple_dict_rows = [
        {"src": src, "rel": rel, "dst": dst}
        for src, rel, dst in triple_rows
    ]

    validate_triples_columns(triple_dict_rows, ["src", "rel", "dst"])
    validate_no_empty_values(triple_dict_rows, ["src", "rel", "dst"])
    relation_counts = validate_relation_counts(triple_dict_rows)

    report = {
        "input_edges": len(edge_rows),
        "output_triples": len(triple_rows),
        "output_triples_with_provenance": len(triple_provenance_rows),
        "skipped": {
            "missing_drug_mapping": skipped_missing_mapping,
            "invalid_rows": skipped_invalid_rows,
            "canonical_duplicates": skipped_canonical_duplicates,
        },
        "relation_distribution": dict(relation_counts),
    }

    write_csv(triples_csv, ["src", "rel", "dst"], triple_rows)
    write_csv(
        triples_with_provenance_csv,
        ["src", "rel", "dst", "source", "file"],
        triple_provenance_rows,
    )

    build_report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved: %s", triples_csv)
    logger.info("Saved: %s", triples_with_provenance_csv)
    logger.info("Saved: %s", build_report_json)
    logger.info(
        "Triples: %d | Provenance rows: %d | Skipped (missing drug mapping): %d | Skipped (invalid row): %d | Skipped (canonical duplicates): %d",
        len(triple_rows),
        len(triple_provenance_rows),
        skipped_missing_mapping,
        skipped_invalid_rows,
        skipped_canonical_duplicates,
    )
    logger.info("Relation distribution: %s", dict(relation_counts))


run = build_triples


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"

    build_triples(cfg={}, paths=Paths(), force=True)
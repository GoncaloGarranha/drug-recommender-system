"""Generate a reproducible audit of drug entities with multiple source names.

The report is derived from one Knowledge Graph build. It contains only facts
extracted from the final triples and alias mapping. Human review decisions must
be stored separately so that regenerating this file never overwrites them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


REPORT_FIELDS = [
    "entity_id",
    "id_namespace",
    "current_first_name",
    "original_name_count",
    "case_insensitive_name_count",
    "only_case_variation",
    "original_names",
    "chebi_ids",
    "drugbank_ids",
    "source_record_count",
    "source_files",
    "identifier_bearing_ingredients",
    "identifier_origin",
    "direct_name_match_records",
    "ingredient_inheritance_records",
    "external_id_without_matching_ingredient_records",
    "missing_source_file_records",
    "audit_flags",
    "total_graph_edges",
    "indication_edges",
    "contraindication_edges",
    "has_ingredient_edges",
    "interacts_with_gene_edges",
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    """Read a CSV file and verify that the required columns are present."""
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = required_fields - fields
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Missing columns in {path}: {missing_text}")
        return [dict(row) for row in reader]


def distinct_in_order(values: Iterable[str]) -> list[str]:
    """Remove exact duplicates while preserving their first occurrence."""
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def compact_identifier(value: str) -> str:
    """Convert known identifier URLs to the compact form used in the graph."""
    value = value.strip()
    if "CHEBI_" in value:
        return "CHEBI:" + value.rsplit("CHEBI_", 1)[1]
    if "drugbank" in value.casefold() and "/drugs/" in value:
        return "DRUGBANK:" + value.rstrip("/").rsplit("/", 1)[1]
    return value


def normalized_name(value: str) -> str:
    """Normalize whitespace and letter case for provenance comparisons."""
    return " ".join(value.split()).casefold()


def selected_external_identifier(
    drug_row: dict[str, str],
    entity_id: str,
) -> str:
    """Return the row identifier that corresponds to the canonical entity."""
    if entity_id.startswith("DRUGBANK:"):
        candidate = compact_identifier(drug_row.get("drugbank_id", ""))
        return candidate if candidate == entity_id else ""
    if entity_id.startswith("CHEBI:"):
        candidate = compact_identifier(drug_row.get("chebi_id", ""))
        return candidate if candidate == entity_id else ""
    return ""


def ingredient_identifiers(ingredient: dict[str, Any]) -> set[str]:
    """Return compact DrugBank and ChEBI identifiers from an ingredient."""
    identifiers = set()
    for field in ("drugbank_id", "chebi_id"):
        value = ingredient.get(field)
        if isinstance(value, str) and value.strip():
            identifiers.add(compact_identifier(value))
    return identifiers


def summarize_identifier_origin(counts: Counter[str]) -> str:
    """Summarize how an external identifier is connected to source products."""
    populated = [key for key, value in counts.items() if value]
    if not populated:
        return "unavailable"
    if len(populated) == 1:
        return populated[0]
    if set(populated) <= {"direct_name_match", "inherited_from_ingredient"}:
        return "mixed_direct_and_inherited"
    return "mixed_identifier_origins"


def build_provenance_by_entity(
    drug_rows: list[dict[str, str]],
    drug_name_to_entity: dict[str, str],
    source_dir: Path,
) -> dict[str, dict[str, str | int]]:
    """Trace canonical identifiers back to MedJsonify product records."""
    source_cache: dict[Path, dict[str, Any] | None] = {}
    records_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for drug_row in drug_rows:
        name_norm = drug_row.get("name_norm", "").strip()
        entity_id = drug_name_to_entity.get(name_norm, "")
        if not entity_id:
            continue

        source_folder = drug_row.get("source", "").strip()
        source_file = drug_row.get("file", "").strip()
        source_reference = "/".join(
            value for value in (source_folder, source_file) if value
        )
        source_path = source_dir / source_folder / source_file
        source_missing = not source_path.is_file()

        source_data: dict[str, Any] | None = None
        if not source_missing:
            if source_path not in source_cache:
                try:
                    loaded = json.loads(source_path.read_text(encoding="utf-8"))
                    source_cache[source_path] = loaded if isinstance(loaded, dict) else None
                except (json.JSONDecodeError, UnicodeDecodeError):
                    source_cache[source_path] = None
            source_data = source_cache[source_path]

        matching_ingredient_names: list[str] = []
        external_identifier = selected_external_identifier(drug_row, entity_id)
        if source_data is not None and external_identifier:
            ingredients = source_data.get("ingredients", [])
            if isinstance(ingredients, list):
                for ingredient in ingredients:
                    if not isinstance(ingredient, dict):
                        continue
                    if external_identifier not in ingredient_identifiers(ingredient):
                        continue
                    ingredient_name = ingredient.get("name")
                    if isinstance(ingredient_name, str) and ingredient_name.strip():
                        matching_ingredient_names.append(ingredient_name.strip())

        product_name = drug_row.get("name_original", "").strip()
        if source_missing or source_data is None:
            origin = "missing_source_file"
        elif entity_id.startswith("DRUG:"):
            origin = "normalized_name_fallback"
        elif matching_ingredient_names:
            product_key = normalized_name(product_name)
            ingredient_keys = {
                normalized_name(name) for name in matching_ingredient_names
            }
            origin = (
                "direct_name_match"
                if product_key and product_key in ingredient_keys
                else "inherited_from_ingredient"
            )
        else:
            origin = "external_id_without_matching_ingredient"

        records_by_entity[entity_id].append(
            {
                "source_reference": source_reference,
                "ingredient_names": matching_ingredient_names,
                "origin": origin,
            }
        )

    provenance: dict[str, dict[str, str | int]] = {}
    for entity_id, records in records_by_entity.items():
        origin_counts = Counter(str(record["origin"]) for record in records)
        source_files = distinct_in_order(
            str(record["source_reference"]) for record in records
        )
        ingredient_names = distinct_in_order(
            name
            for record in records
            for name in record["ingredient_names"]
            if isinstance(name, str)
        )
        provenance[entity_id] = {
            "source_record_count": len(records),
            "source_files": " | ".join(source_files),
            "identifier_bearing_ingredients": " | ".join(ingredient_names),
            "identifier_origin": summarize_identifier_origin(origin_counts),
            "direct_name_match_records": origin_counts["direct_name_match"],
            "ingredient_inheritance_records": origin_counts["inherited_from_ingredient"],
            "external_id_without_matching_ingredient_records": origin_counts[
                "external_id_without_matching_ingredient"
            ],
            "missing_source_file_records": origin_counts["missing_source_file"],
        }
    return provenance


def build_report_rows(
    triples: list[dict[str, str]],
    aliases: list[dict[str, str]],
    provenance_by_entity: dict[str, dict[str, str | int]],
) -> list[dict[str, str | int]]:
    """Build deterministic report rows from graph triples and entity aliases."""
    graph_entities: set[str] = set()
    edge_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for triple in triples:
        source = triple["src"].strip()
        relation = triple["rel"].strip()
        destination = triple["dst"].strip()
        graph_entities.update((source, destination))
        edge_counts[source][relation] += 1
        edge_counts[destination][relation] += 1

    aliases_by_entity_and_type: dict[tuple[str, str], list[str]] = defaultdict(list)
    for alias_row in aliases:
        entity_id = alias_row["entity_id"].strip()
        if entity_id not in graph_entities:
            continue
        alias_type = alias_row["alias_type"].strip()
        alias = alias_row["alias"].strip()
        if alias:
            aliases_by_entity_and_type[(entity_id, alias_type)].append(alias)

    ambiguous: list[tuple[str, list[str]]] = []
    for (entity_id, alias_type), values in aliases_by_entity_and_type.items():
        if alias_type != "DRUG_NAME_ORIGINAL":
            continue
        names = distinct_in_order(values)
        if len(names) > 1:
            ambiguous.append((entity_id, names))

    ambiguous.sort(key=lambda item: (-len(item[1]), item[0]))

    rows: list[dict[str, str | int]] = []
    for entity_id, names in ambiguous:
        folded_name_count = len({name.casefold() for name in names})
        relation_counts = edge_counts[entity_id]
        chebi_ids = distinct_in_order(
            compact_identifier(value)
            for value in aliases_by_entity_and_type.get((entity_id, "CHEBI_ID"), [])
        )
        drugbank_ids = distinct_in_order(
            compact_identifier(value)
            for value in aliases_by_entity_and_type.get((entity_id, "DRUGBANK_ID"), [])
        )
        provenance = provenance_by_entity.get(entity_id, {})
        audit_flags = []
        if len(chebi_ids) > 1 or len(drugbank_ids) > 1:
            audit_flags.append("multiple_external_identifiers")
        if int(provenance.get("ingredient_inheritance_records", 0)) > 0:
            audit_flags.append("identifier_inherited_from_ingredient")
        if int(
            provenance.get("external_id_without_matching_ingredient_records", 0)
        ) > 0:
            audit_flags.append("external_id_without_matching_ingredient")
        if int(provenance.get("missing_source_file_records", 0)) > 0:
            audit_flags.append("missing_source_file")

        rows.append(
            {
                "entity_id": entity_id,
                "id_namespace": entity_id.split(":", 1)[0],
                "current_first_name": names[0],
                "original_name_count": len(names),
                "case_insensitive_name_count": folded_name_count,
                "only_case_variation": "yes" if folded_name_count == 1 else "no",
                "original_names": " | ".join(names),
                "chebi_ids": " | ".join(chebi_ids),
                "drugbank_ids": " | ".join(drugbank_ids),
                "source_record_count": provenance.get("source_record_count", 0),
                "source_files": provenance.get("source_files", ""),
                "identifier_bearing_ingredients": provenance.get(
                    "identifier_bearing_ingredients", ""
                ),
                "identifier_origin": provenance.get("identifier_origin", "unavailable"),
                "direct_name_match_records": provenance.get(
                    "direct_name_match_records", 0
                ),
                "ingredient_inheritance_records": provenance.get(
                    "ingredient_inheritance_records", 0
                ),
                "external_id_without_matching_ingredient_records": provenance.get(
                    "external_id_without_matching_ingredient_records", 0
                ),
                "missing_source_file_records": provenance.get(
                    "missing_source_file_records", 0
                ),
                "audit_flags": " | ".join(audit_flags),
                "total_graph_edges": sum(relation_counts.values()),
                "indication_edges": relation_counts["INDICATION"],
                "contraindication_edges": relation_counts["CONTRAINDICATION"],
                "has_ingredient_edges": relation_counts["HAS_INGREDIENT"],
                "interacts_with_gene_edges": relation_counts["INTERACTS_WITH_GENE"],
            }
        )

    entity_ids = [str(row["entity_id"]) for row in rows]
    if len(entity_ids) != len(set(entity_ids)):
        raise ValueError("The generated report contains duplicate entity identifiers")
    if any(int(row["original_name_count"]) < 2 for row in rows):
        raise ValueError("The generated report contains a non-ambiguous entity")

    return rows


def write_report(path: Path, rows: list[dict[str, str | int]]) -> None:
    """Write the audit report using stable UTF-8 and Unix line endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an audit of final-graph drug entities associated with "
            "more than one original name."
        )
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="KG build directory containing triples/ and mappings/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination CSV for the generated audit",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="MedJsonify input directory containing csv_files/, txt_files/ and xml_files/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dir = args.build_dir.resolve()
    output_path = args.output.resolve()
    source_dir = args.source_dir.resolve()
    triples_path = build_dir / "triples" / "triples_with_dgidb.csv"
    aliases_path = build_dir / "mappings" / "aliases.csv"
    drugs_path = build_dir / "interim" / "drugs_raw.csv"
    drug_map_path = build_dir / "mappings" / "drug_name_to_entity.json"

    triples = read_csv(triples_path, {"src", "rel", "dst"})
    aliases = read_csv(aliases_path, {"alias", "entity_id", "alias_type"})
    drug_rows = read_csv(
        drugs_path,
        {"source", "name_original", "name_norm", "drugbank_id", "chebi_id", "file"},
    )
    if not drug_map_path.is_file():
        raise FileNotFoundError(f"Input file not found: {drug_map_path}")
    loaded_drug_map = json.loads(drug_map_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_drug_map, dict):
        raise ValueError(f"Expected a JSON object in {drug_map_path}")
    drug_name_to_entity = {
        str(name): str(entity_id) for name, entity_id in loaded_drug_map.items()
    }
    provenance = build_provenance_by_entity(
        drug_rows,
        drug_name_to_entity,
        source_dir,
    )
    rows = build_report_rows(triples, aliases, provenance)
    write_report(output_path, rows)

    case_only = sum(row["only_case_variation"] == "yes" for row in rows)
    print(f"Generated: {output_path}")
    print(f"Entities with multiple exact source names: {len(rows)}")
    print(f"Entities differing only in letter case: {case_only}")
    print(f"Triples SHA-256: {sha256_file(triples_path)}")
    print(f"Aliases SHA-256: {sha256_file(aliases_path)}")
    print(f"Raw drugs SHA-256: {sha256_file(drugs_path)}")
    print(f"Drug mapping SHA-256: {sha256_file(drug_map_path)}")
    print(f"Report SHA-256: {sha256_file(output_path)}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_constructor.src.utils.log import get_logger

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


def resolve_entities(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 03: Build canonical entity identifiers and alias mappings.

    This step resolves canonical entities for:
        - drugs
        - ingredients
        - diseases

    Drug name variants are mapped to a single canonical entity ID. When a
    DrugBank identifier is available, it is preferred as the canonical ID.
    Otherwise, a normalized name-based identifier is used.

    Inputs:
        - data/output/interim/drugs_raw.csv
        - data/output/interim/ingredients_raw.csv
        - data/output/interim/edges_raw.csv

    Outputs:
        - data/output/mappings/entity_table.csv
        - data/output/mappings/aliases.csv
        - data/output/mappings/resolution_report.json
        - data/output/mappings/drug_name_to_entity.json

    Args:
        cfg: Pipeline configuration dictionary. Kept for interface consistency.
        paths: Object containing pipeline paths. Must provide `output_dir`.
        force: If True, overwrite existing outputs.
    """
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    drugs_csv = interim_dir / "drugs_raw.csv"
    ingredients_csv = interim_dir / "ingredients_raw.csv"
    edges_csv = interim_dir / "edges_raw.csv"

    entity_table_csv = mappings_dir / "entity_table.csv"
    aliases_csv = mappings_dir / "aliases.csv"
    report_json = mappings_dir / "resolution_report.json"
    drug_map_json = mappings_dir / "drug_name_to_entity.json"

    output_files = [entity_table_csv, aliases_csv, report_json, drug_map_json]
    if not force and all(path.exists() for path in output_files):
        logger.info("Mapping outputs already exist. Skipping step (use --force to re-run).")
        return

    if not drugs_csv.exists() or not edges_csv.exists():
        raise FileNotFoundError("Missing interim inputs. Run step 02_parse_sources first.")

    drug_rows = read_csv(drugs_csv)
    ingredient_rows = read_csv(ingredients_csv) if ingredients_csv.exists() else []
    edge_rows = read_csv(edges_csv)

    # -------------------------------------------------------------------------
    # Drug entity resolution
    # -------------------------------------------------------------------------
    # For each normalized drug name, count how often each DrugBank ID appears.
    normalized_name_to_drugbank_ids: Dict[str, Counter] = defaultdict(Counter)

    for row in drug_rows:
        name_norm = (row.get("name_norm") or "").strip()
        drugbank_id = (row.get("drugbank_id") or "").strip()

        if not name_norm:
            continue

        if drugbank_id:
            normalized_name_to_drugbank_ids[name_norm][drugbank_id] += 1

    # Build the final mapping from normalized drug name to canonical entity ID.
    # Stats are counted after the dict is fully built to reflect unique entities,
    # not raw CSV rows (which may contain the same drug from multiple files).
    drug_name_to_entity: Dict[str, str] = {}

    for row in drug_rows:
        name_norm = (row.get("name_norm") or "").strip()
        if not name_norm:
            continue

        if normalized_name_to_drugbank_ids.get(name_norm):
            best_drugbank_id, _ = normalized_name_to_drugbank_ids[name_norm].most_common(1)[0]
            entity_id = f"DRUGBANK:{best_drugbank_id}"
        else:
            entity_id = f"DRUG:{name_norm}"

        drug_name_to_entity[name_norm] = entity_id

    drug_resolution_stats = Counter(
        "resolved_by_drugbank" if entity_id.startswith("DRUGBANK:") else "resolved_by_name"
        for entity_id in drug_name_to_entity.values()
    )

    # -------------------------------------------------------------------------
    # Ingredient entity resolution
    # -------------------------------------------------------------------------
    ingredient_entities: Dict[str, Tuple[str, str]] = {}

    for row in ingredient_rows:
        name_norm = (row.get("name_norm") or "").strip()
        unii_code = (row.get("unii_code") or "").strip()

        if unii_code:
            entity_id = f"UNII:{unii_code}"
            label = name_norm if name_norm else unii_code
            ingredient_entities[entity_id] = ("INGREDIENT", label)
        elif name_norm:
            entity_id = f"ING:{name_norm}"
            ingredient_entities[entity_id] = ("INGREDIENT", name_norm)

    # -------------------------------------------------------------------------
    # Disease entity resolution
    # -------------------------------------------------------------------------
    disease_entities: Dict[str, Tuple[str, str]] = {}

    for row in edge_rows:
        dst = (row.get("dst") or "").strip()
        if dst.startswith("DOID:"):
            disease_entities[dst] = ("DISEASE", dst)

    # -------------------------------------------------------------------------
    # Build entity_table.csv
    # -------------------------------------------------------------------------
    entity_rows: List[List[Any]] = []

    unique_drug_entities = set(drug_name_to_entity.values())
    for entity_id in sorted(unique_drug_entities):
        label = entity_id.replace("DRUGBANK:", "").replace("DRUG:", "")
        entity_rows.append([entity_id, "DRUG", label])

    for entity_id in sorted(disease_entities.keys()):
        entity_rows.append([entity_id, "DISEASE", entity_id])

    for entity_id, (entity_type, label) in sorted(ingredient_entities.items()):
        entity_rows.append([entity_id, entity_type, label])

    write_csv(entity_table_csv, ["entity_id", "entity_type", "label"], entity_rows)
    logger.info("Saved: %s", entity_table_csv)

    # -------------------------------------------------------------------------
    # Build aliases.csv
    # -------------------------------------------------------------------------
    alias_rows: List[List[Any]] = []

    for row in drug_rows:
        name_original = (row.get("name_original") or "").strip()
        name_norm = (row.get("name_norm") or "").strip()

        if not name_norm:
            continue

        entity_id = drug_name_to_entity.get(name_norm)
        if not entity_id:
            continue

        if name_original:
            alias_rows.append([name_original, entity_id, "DRUG_NAME_ORIGINAL"])
        alias_rows.append([name_norm, entity_id, "DRUG_NAME_NORM"])

    for row in ingredient_rows:
        name_original = (row.get("name_original") or "").strip()
        name_norm = (row.get("name_norm") or "").strip()
        unii_code = (row.get("unii_code") or "").strip()

        if unii_code:
            entity_id = f"UNII:{unii_code}"
        elif name_norm:
            entity_id = f"ING:{name_norm}"
        else:
            continue

        if name_original:
            alias_rows.append([name_original, entity_id, "ING_NAME_ORIGINAL"])
        if name_norm:
            alias_rows.append([name_norm, entity_id, "ING_NAME_NORM"])
        if unii_code:
            alias_rows.append([unii_code, entity_id, "UNII_CODE"])

    write_csv(aliases_csv, ["alias", "entity_id", "alias_type"], alias_rows)
    logger.info("Saved: %s", aliases_csv)

    # -------------------------------------------------------------------------
    # Save auxiliary mapping outputs
    # -------------------------------------------------------------------------
    drug_map_json.write_text(
        json.dumps(drug_name_to_entity, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved: %s", drug_map_json)

    report = {
        "drug_resolution": dict(drug_resolution_stats),
        "unique_drug_entities": len(unique_drug_entities),
        "unique_disease_entities": len(disease_entities),
        "unique_ingredient_entities": len(ingredient_entities),
        "note": "Drug entities prefer DRUGBANK:<id> when available; otherwise DRUG:<NAME_NORM>.",
    }

    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved: %s", report_json)


# Alias used by the pipeline runner
run = resolve_entities


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"

    resolve_entities(cfg={}, paths=Paths(), force=True)
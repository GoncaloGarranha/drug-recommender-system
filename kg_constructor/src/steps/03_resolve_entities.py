from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_constructor.src.utils.io import read_csv, write_csv
from kg_constructor.src.utils.log import get_logger
from kg_constructor.src.utils.validate import (
    validate_aliases,
    validate_entity_table,
)

logger = get_logger(__name__)


def format_prefixed_identifier(raw_identifier: str, prefix: str) -> str:
    """
    Format an external identifier with a canonical namespace prefix.

    Examples:
        http://purl.obolibrary.org/obo/CHEBI_31672 -> CHEBI:31672
        CHEBI_31672 -> CHEBI:31672
        CHEBI:31672 -> CHEBI:31672
        31672 -> CHEBI:31672
        DRUGBANK:https://go.drugbank.com/drugs/DB00051 -> DRUGBANK:DB00051
        drugbank:db00051 -> DRUGBANK:DB00051
    """
    identifier = str(raw_identifier).strip()
    normalized_prefix = prefix.upper()
    expected_prefix = f"{normalized_prefix}:"

    if identifier.upper().startswith(expected_prefix):
        identifier = identifier.split(":", 1)[1].strip()

    if identifier.lower().startswith("http"):
        identifier = identifier.rstrip("/").rsplit("/", 1)[-1].strip()

    identifier = identifier.replace("_", ":")

    if identifier.upper().startswith(expected_prefix):
        identifier = identifier.split(":", 1)[1].strip()

    if normalized_prefix == "DRUGBANK":
        identifier = identifier.upper()

    return f"{expected_prefix}{identifier}"


def resolve_entities(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 03: Build canonical entity identifiers and alias mappings.

    This step resolves canonical entities for:
        - drugs
        - ingredients
        - diseases

    Drug name variants are mapped to a single canonical entity ID. The
    canonical identifier strategy is controlled by:
        cfg["entity_resolution"]["prefer_drugbank_id"]
        cfg["entity_resolution"]["prefer_chebi_id"]

    Resolution order:
        1. DrugBank ID, when enabled and available.
        2. ChEBI ID, when enabled and available.
        3. Normalized drug name fallback.

    Disease entities keep DOID identifiers as canonical IDs, while labels are
    enriched from the MedJsonify text metadata when available.

    Inputs:
        - data/output/interim/drugs_raw.csv
        - data/output/interim/ingredients_raw.csv
        - data/output/interim/diseases_raw.csv
        - data/output/interim/edges_raw.csv

    Outputs:
        - data/output/mappings/entity_table.csv
        - data/output/mappings/aliases.csv
        - data/output/mappings/resolution_report.json
        - data/output/mappings/drug_name_to_entity.json

    Args:
        cfg: Pipeline configuration dictionary.
        paths: Object containing pipeline paths. Must provide `output_dir`.
        force: If True, overwrite existing outputs.
    """
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    mappings_dir = output_dir / "mappings"
    drugs_csv = interim_dir / "drugs_raw.csv"
    ingredients_csv = interim_dir / "ingredients_raw.csv"
    diseases_csv = interim_dir / "diseases_raw.csv"
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
    disease_rows = read_csv(diseases_csv) if diseases_csv.exists() else []
    edge_rows = read_csv(edges_csv)

    # -------------------------------------------------------------------------
    # Drug entity resolution
    # -------------------------------------------------------------------------
    entity_resolution_cfg = cfg.get("entity_resolution", {})
    prefer_drugbank_id = bool(entity_resolution_cfg.get("prefer_drugbank_id", True))
    prefer_chebi_id = bool(entity_resolution_cfg.get("prefer_chebi_id", True))
    reject_ambiguous_chebi_ids = bool(
        entity_resolution_cfg.get("reject_ambiguous_chebi_ids", True)
    )

    logger.info(
        "Drug entity resolution config | prefer_drugbank_id=%s | prefer_chebi_id=%s | reject_ambiguous_chebi_ids=%s",
        prefer_drugbank_id,
        prefer_chebi_id,
        reject_ambiguous_chebi_ids,
    )

    normalized_name_to_drugbank_ids: Dict[str, Counter] = defaultdict(Counter)
    normalized_name_to_chebi_ids: Dict[str, Counter] = defaultdict(Counter)
    chebi_id_to_names: Dict[str, set] = defaultdict(set)

    for row in drug_rows:
        name_norm = (row.get("name_norm") or "").strip()
        drugbank_id = (row.get("drugbank_id") or "").strip()
        chebi_id = (row.get("chebi_id") or "").strip()

        if not name_norm:
            continue

        if drugbank_id:
            normalized_name_to_drugbank_ids[name_norm][drugbank_id] += 1

        if chebi_id:
            normalized_name_to_chebi_ids[name_norm][chebi_id] += 1
            chebi_id_to_names[chebi_id].add(name_norm)

    ambiguous_chebi_ids = set()

    if reject_ambiguous_chebi_ids:
        ambiguous_chebi_ids = {
            identifier
            for identifier, names in chebi_id_to_names.items()
            if len(names) > 1
        }

    logger.info(
        "Ambiguous ChEBI identifiers rejected: %d",
        len(ambiguous_chebi_ids),
    )

    drug_name_to_entity: Dict[str, str] = {}

    for row in drug_rows:
        name_norm = (row.get("name_norm") or "").strip()
        if not name_norm:
            continue

        entity_id = None

        if prefer_drugbank_id and normalized_name_to_drugbank_ids.get(name_norm):
            best_drugbank_id, _ = normalized_name_to_drugbank_ids[name_norm].most_common(1)[0]
            entity_id = format_prefixed_identifier(best_drugbank_id, "DRUGBANK")

        if (
            entity_id is None
            and prefer_chebi_id
            and normalized_name_to_chebi_ids.get(name_norm)
        ):
            best_chebi_id, _ = normalized_name_to_chebi_ids[name_norm].most_common(1)[0]

            if best_chebi_id not in ambiguous_chebi_ids:
                entity_id = format_prefixed_identifier(best_chebi_id, "CHEBI")

        if entity_id is None:
            entity_id = f"DRUG:{name_norm}"

        drug_name_to_entity[name_norm] = entity_id

    drug_resolution_stats = Counter(
        "resolved_by_drugbank" if entity_id.startswith("DRUGBANK:")
        else "resolved_by_chebi" if entity_id.startswith("CHEBI:")
        else "resolved_by_name"
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
            entity_id = format_prefixed_identifier(unii_code, "UNII")
            label = name_norm if name_norm else unii_code
            ingredient_entities[entity_id] = ("INGREDIENT", label)
        elif name_norm:
            entity_id = f"ING:{name_norm}"
            ingredient_entities[entity_id] = ("INGREDIENT", name_norm)

    # -------------------------------------------------------------------------
    # Disease entity resolution
    # -------------------------------------------------------------------------
    disease_entities: Dict[str, Tuple[str, str]] = {}

    for row in disease_rows:
        doid = (row.get("doid") or "").strip()
        label = (row.get("label") or "").strip()

        if doid.startswith("DOID:"):
            disease_entities[doid] = ("DISEASE", label if label else doid)

    # Fallback: keep DOID entities that appear in edges even if not present in diseases_raw.csv.
    for row in edge_rows:
        dst = (row.get("dst") or "").strip()
        if dst.startswith("DOID:") and dst not in disease_entities:
            disease_entities[dst] = ("DISEASE", dst)

    # -------------------------------------------------------------------------
    # Build entity_table.csv
    # -------------------------------------------------------------------------
    entity_rows: List[List[Any]] = []

    unique_drug_entities = set(drug_name_to_entity.values())
    for entity_id in sorted(unique_drug_entities):
        label = (
            entity_id
            .replace("DRUGBANK:", "")
            .replace("CHEBI:", "")
            .replace("DRUG:", "")
        )
        entity_rows.append([entity_id, "DRUG", label])

    for entity_id, (entity_type, label) in sorted(disease_entities.items()):
        entity_rows.append([entity_id, entity_type, label])

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
        drugbank_id = (row.get("drugbank_id") or "").strip()
        chebi_id = (row.get("chebi_id") or "").strip()

        if not name_norm:
            continue

        entity_id = drug_name_to_entity.get(name_norm)
        if not entity_id:
            continue

        if name_original:
            alias_rows.append([name_original, entity_id, "DRUG_NAME_ORIGINAL"])
        alias_rows.append([name_norm, entity_id, "DRUG_NAME_NORM"])

        if drugbank_id:
            alias_rows.append([drugbank_id, entity_id, "DRUGBANK_ID"])
        if chebi_id:
            alias_rows.append([chebi_id, entity_id, "CHEBI_ID"])

    for row in ingredient_rows:
        name_original = (row.get("name_original") or "").strip()
        name_norm = (row.get("name_norm") or "").strip()
        unii_code = (row.get("unii_code") or "").strip()

        if unii_code:
            entity_id = format_prefixed_identifier(unii_code, "UNII")
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

    # Disease aliases: include readable labels and full MedJsonify text when available.
    for row in disease_rows:
        doid = (row.get("doid") or "").strip()
        label = (row.get("label") or "").strip()
        text = (row.get("text") or "").strip()

        if not doid.startswith("DOID:"):
            continue

        alias_rows.append([doid, doid, "DOID_ID"])
        if label and label != doid:
            alias_rows.append([label, doid, "DISEASE_LABEL"])
        if text and text not in {label, doid}:
            alias_rows.append([text, doid, "DISEASE_TEXT"])

    write_csv(aliases_csv, ["alias", "entity_id", "alias_type"], alias_rows)
    logger.info("Saved: %s", aliases_csv)

    # -------------------------------------------------------------------------
    # Validate mapping outputs
    # -------------------------------------------------------------------------
    entity_table_rows = read_csv(entity_table_csv)
    alias_rows_dict = read_csv(aliases_csv)

    validation_report = {
        "entity_table": validate_entity_table(entity_table_rows),
        "aliases": validate_aliases(alias_rows_dict),
    }

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
        "entity_resolution_config": {
            "prefer_drugbank_id": prefer_drugbank_id,
            "prefer_chebi_id": prefer_chebi_id,
        },
        "disease_metadata": {
            "diseases_with_label": sum(
                1 for _, (_, label) in disease_entities.items() if label and not label.startswith("DOID:")
            ),
            "diseases_without_label": sum(
                1 for _, (_, label) in disease_entities.items() if not label or label.startswith("DOID:")
            ),
            "source": "MedJsonify doid_entities.text",
        },
        "validation": validation_report,
        "note": (
            "Drug entity resolution is configurable. Resolution order is "
            "DRUGBANK:<id> when prefer_drugbank_id is true and a DrugBank ID exists; "
            "CHEBI:<id> when prefer_chebi_id is true and a ChEBI ID exists; "
            "otherwise DRUG:<NAME_NORM>. Disease labels are enriched from "
            "MedJsonify doid_entities.text when available."
        ),
    }

    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved: %s", report_json)


# Alias used by the pipeline runner
run = resolve_entities

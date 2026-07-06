from __future__ import annotations

from kg_constructor.src.utils.paths import build_output_paths
"""
Step 02: Parse input JSON files and generate clean, normalized CSV tables
for drugs, ingredients, diseases, and edges.

All names and identifiers are normalized at this stage to ensure consistency
and simplify canonical mapping in the following steps.
"""


import json
from pathlib import Path
from typing import Any, Dict, List

from kg_constructor.src.utils.io import (
    iter_json_files,
    load_json_file,
    read_csv,
    write_csv,
)
from kg_constructor.src.utils.log import get_logger
from kg_constructor.src.utils.text import (
    clean_disease_label,
    clean_whitespace,
    normalize_doid,
    normalize_canonical_label,
)
from kg_constructor.src.utils.validate import (
    validate_diseases_raw,
    validate_drugs_raw,
    validate_edges_raw,
    validate_ingredients_raw,
)

logger = get_logger(__name__)

SOURCE_FOLDERS = ("xml_files", "txt_files", "csv_files")


def extract_drugs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract valid drug records from the parsed JSON structure.
    """
    drugs = data.get("drug", [])
    if isinstance(drugs, list):
        return [drug for drug in drugs if isinstance(drug, dict)]
    return []


def extract_ingredients(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract valid ingredient records from the parsed JSON structure.
    """
    ingredients = data.get("ingredients", [])
    if isinstance(ingredients, list):
        return [ingredient for ingredient in ingredients if isinstance(ingredient, dict)]
    return []


def extract_doid_entities(data: Dict[str, Any], section_name: str) -> List[Dict[str, str]]:
    """
    Extract normalized DOID entities and their text metadata from a section.
    """
    section = data.get(section_name, {})
    if not isinstance(section, dict):
        return []

    entities = section.get("doid_entities", [])
    if not isinstance(entities, list):
        return []

    results: List[Dict[str, str]] = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        doid_raw = entity.get("doid_id")
        text_raw = entity.get("text")

        if not isinstance(doid_raw, str) or not doid_raw.strip():
            continue

        doid = normalize_doid(doid_raw)
        text = clean_whitespace(text_raw) if isinstance(text_raw, str) and text_raw.strip() else ""
        label = clean_disease_label(text) if text else doid

        results.append({
            "doid": doid,
            "label": label if label else doid,
            "text": text,
        })

    return results


def parse_sources(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 02: Parse input JSON files and generate normalized interim CSV tables.

    Outputs:
        - data/output/interim/drugs_raw.csv
        - data/output/interim/ingredients_raw.csv
        - data/output/interim/diseases_raw.csv
        - data/output/interim/edges_raw.csv
        - data/output/reports/interim_validation_report.json
    """
    input_dir = Path(paths.input_dir)
    output_dir = Path(paths.output_dir)
    out = build_output_paths(paths.output_dir)
    interim_dir = out.interim
    reports_dir = out.reports
    interim_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    drugs_csv = interim_dir / "drugs_raw.csv"
    ingredients_csv = interim_dir / "ingredients_raw.csv"
    diseases_csv = interim_dir / "diseases_raw.csv"
    edges_csv = interim_dir / "edges_raw.csv"
    validation_report_json = reports_dir / "interim_validation_report.json"

    output_files = [drugs_csv, ingredients_csv, diseases_csv, edges_csv]
    if not force and all(path.exists() for path in output_files):
        logger.info("Interim outputs already exist. Skipping step (use --force to re-run).")
        return

    drug_rows: List[List[Any]] = []
    ingredient_rows: List[List[Any]] = []
    disease_rows: List[List[Any]] = []
    edge_rows: List[List[Any]] = []

    seen_drugs = set()
    seen_ingredients = set()
    seen_diseases = set()
    seen_edges = set()

    for folder_name in SOURCE_FOLDERS:
        folder_path = input_dir / folder_name

        if not folder_path.exists():
            logger.warning("Missing input folder: %s", folder_path)
            continue

        for json_file in iter_json_files(folder_path):
            data = load_json_file(json_file)

            drug_records = extract_drugs(data)
            ingredient_records = extract_ingredients(data)

            for drug in drug_records:
                drug_name = drug.get("name")
                if not isinstance(drug_name, str) or not drug_name.strip():
                    continue

                name_original = clean_whitespace(drug_name)
                name_norm = normalize_canonical_label(name_original)

                drugbank_id = drug.get("drugbank_id")
                chebi_id = drug.get("chebi_id")

                drugbank_id = drugbank_id.strip() if isinstance(drugbank_id, str) else ""
                chebi_id = chebi_id.strip() if isinstance(chebi_id, str) else ""

                drug_key = (name_norm, drugbank_id, chebi_id)
                if drug_key in seen_drugs:
                    continue

                seen_drugs.add(drug_key)
                drug_rows.append([
                    folder_name,
                    name_original,
                    name_norm,
                    drugbank_id,
                    chebi_id,
                    json_file.name,
                ])

            for ingredient in ingredient_records:
                ingredient_name = ingredient.get("name")
                ingredient_code = ingredient.get("code")

                name_original = (
                    clean_whitespace(ingredient_name)
                    if isinstance(ingredient_name, str) and ingredient_name.strip()
                    else ""
                )
                name_norm = normalize_canonical_label(name_original) if name_original else ""
                unii_code = (
                    clean_whitespace(ingredient_code)
                    if isinstance(ingredient_code, str) and ingredient_code.strip()
                    else ""
                )

                if not name_norm and not unii_code:
                    continue

                ingredient_key = (name_norm, unii_code)
                if ingredient_key in seen_ingredients:
                    continue

                seen_ingredients.add(ingredient_key)
                ingredient_rows.append([
                    folder_name,
                    name_original,
                    name_norm,
                    unii_code,
                    json_file.name,
                ])

            primary_drug_name_norm = ""

            if drug_records:
                first_drug_name = drug_records[0].get("name")
                if isinstance(first_drug_name, str) and first_drug_name.strip():
                    primary_drug_name_norm = normalize_canonical_label(first_drug_name)

            if not primary_drug_name_norm:
                continue

            for relation_type, section_name in (
                ("INDICATION", "indications"),
                ("CONTRAINDICATION", "contraindications"),
            ):
                doid_entities = extract_doid_entities(data, section_name)

                for disease in doid_entities:
                    doid_id = disease["doid"]
                    disease_label = disease["label"]
                    disease_text = disease["text"]

                    if doid_id not in seen_diseases:
                        seen_diseases.add(doid_id)
                        disease_rows.append([
                            doid_id,
                            disease_label,
                            disease_text,
                            section_name,
                            json_file.name,
                        ])

                    edge_key = (primary_drug_name_norm, relation_type, doid_id)
                    if edge_key in seen_edges:
                        continue

                    seen_edges.add(edge_key)
                    edge_rows.append([
                        primary_drug_name_norm,
                        relation_type,
                        doid_id,
                        folder_name,
                        json_file.name,
                    ])

            for ingredient in ingredient_records:
                ingredient_name = ingredient.get("name")
                ingredient_code = ingredient.get("code")

                name_original = (
                    clean_whitespace(ingredient_name)
                    if isinstance(ingredient_name, str) and ingredient_name.strip()
                    else ""
                )
                name_norm = normalize_canonical_label(name_original) if name_original else ""
                unii_code = (
                    clean_whitespace(ingredient_code)
                    if isinstance(ingredient_code, str) and ingredient_code.strip()
                    else ""
                )

                if unii_code:
                    ingredient_id = f"UNII:{unii_code}"
                elif name_norm:
                    ingredient_id = f"ING:{name_norm}"
                else:
                    continue

                edge_key = (primary_drug_name_norm, "HAS_INGREDIENT", ingredient_id)
                if edge_key in seen_edges:
                    continue

                seen_edges.add(edge_key)
                edge_rows.append([
                    primary_drug_name_norm,
                    "HAS_INGREDIENT",
                    ingredient_id,
                    folder_name,
                    json_file.name,
                ])

    write_csv(
        drugs_csv,
        ["source", "name_original", "name_norm", "drugbank_id", "chebi_id", "file"],
        drug_rows,
    )
    write_csv(
        ingredients_csv,
        ["source", "name_original", "name_norm", "unii_code", "file"],
        ingredient_rows,
    )
    write_csv(
        diseases_csv,
        ["doid", "label", "text", "source_section", "file"],
        disease_rows,
    )
    write_csv(edges_csv, ["src", "rel", "dst", "source", "file"], edge_rows)

    validation_report = {
        "drugs_raw": validate_drugs_raw(read_csv(drugs_csv)),
        "ingredients_raw": validate_ingredients_raw(read_csv(ingredients_csv)),
        "diseases_raw": validate_diseases_raw(read_csv(diseases_csv)),
        "edges_raw": validate_edges_raw(read_csv(edges_csv)),
    }

    validation_report_json.write_text(
        json.dumps(validation_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Saved: %s", drugs_csv)
    logger.info("Saved: %s", ingredients_csv)
    logger.info("Saved: %s", diseases_csv)
    logger.info("Saved: %s", edges_csv)
    logger.info("Saved: %s", validation_report_json)


run = parse_sources

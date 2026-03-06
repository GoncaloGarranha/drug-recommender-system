"""
Step 02: Parse input JSON files and generate clean, normalized CSV tables
for drugs, ingredients, diseases, and edges.

All names and identifiers are normalized at this stage to ensure consistency
and simplify canonical mapping in the following steps.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)

SOURCE_FOLDERS = ("xml_files", "txt_files", "csv_files")
DOID_PATTERN = re.compile(r"(DOID[_:])(\d+)$", re.IGNORECASE)


def iter_json_files(folder: Path) -> Iterable[Path]:
    """
    Yield all JSON files from a given folder.

    Args:
        folder: Folder to scan.

    Yields:
        Path objects for .json files.
    """
    if not folder.exists():
        return

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            yield file_path


def load_json_file(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file and raise a clear error if parsing fails.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON file: {path} | Error: {exc}") from exc


def clean_whitespace(text: str) -> str:
    """
    Remove leading/trailing spaces and collapse repeated internal whitespace.

    Args:
        text: Input text.

    Returns:
        Cleaned text.
    """
    return " ".join(text.strip().split())


def normalize_name(text: str) -> str:
    """
    Normalize entity names for canonical matching.

    This normalization:
        - removes extra whitespace
        - converts the text to uppercase

    Args:
        text: Raw entity name.

    Returns:
        Normalized entity name.
    """
    return clean_whitespace(text).upper()


def normalize_doid(doid_raw: str) -> str:
    """
    Normalize a DOID identifier to the canonical format 'DOID:xxxx'.

    Supported cases:
        - URLs ending in a DOID identifier
        - DOID:xxxx
        - DOID_xxxx

    If no known pattern is matched, the original cleaned value is returned.

    Args:
        doid_raw: Raw DOID value.

    Returns:
        Normalized DOID identifier.
    """
    value = doid_raw.strip()

    if value.lower().startswith("http"):
        value = value.rsplit("/", 1)[-1]

    match = DOID_PATTERN.search(value)
    if match:
        return f"DOID:{match.group(2)}"

    if value.upper().startswith("DOID:"):
        return f"DOID:{value.split(':', 1)[1].strip()}"

    return value


def extract_drugs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract valid drug records from the parsed JSON structure.

    Args:
        data: Parsed JSON content.

    Returns:
        List of drug dictionaries.
    """
    drugs = data.get("drug", [])
    if isinstance(drugs, list):
        return [drug for drug in drugs if isinstance(drug, dict)]
    return []


def extract_ingredients(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract valid ingredient records from the parsed JSON structure.

    Args:
        data: Parsed JSON content.

    Returns:
        List of ingredient dictionaries.
    """
    ingredients = data.get("ingredients", [])
    if isinstance(ingredients, list):
        return [ingredient for ingredient in ingredients if isinstance(ingredient, dict)]
    return []


def extract_doid_ids(data: Dict[str, Any], section_name: str) -> List[str]:
    """
    Extract normalized DOID identifiers from a given section.

    The expected sections are typically:
        - indications
        - contraindications

    Args:
        data: Parsed JSON content.
        section_name: Section to inspect.

    Returns:
        List of normalized DOID identifiers.
    """
    section = data.get(section_name, {})
    if not isinstance(section, dict):
        return []

    entities = section.get("doid_entities", [])
    if not isinstance(entities, list):
        return []

    doid_ids: List[str] = []

    for entity in entities:
        if not isinstance(entity, dict):
            continue

        doid_id = entity.get("doid_id")
        if isinstance(doid_id, str) and doid_id.strip():
            doid_ids.append(normalize_doid(doid_id))

    return doid_ids


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


def parse_sources(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 02: Parse input JSON files and generate normalized interim CSV tables.

    Outputs:
        - data/output/interim/drugs_raw.csv
        - data/output/interim/ingredients_raw.csv
        - data/output/interim/diseases_raw.csv
        - data/output/interim/edges_raw.csv

    Args:
        cfg: Pipeline configuration dictionary. Kept for interface consistency.
        paths: Object containing pipeline paths. Must provide `input_dir` and `output_dir`.
        force: If True, overwrite existing outputs.
    """
    input_dir = Path(paths.input_dir)
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    drugs_csv = interim_dir / "drugs_raw.csv"
    ingredients_csv = interim_dir / "ingredients_raw.csv"
    diseases_csv = interim_dir / "diseases_raw.csv"
    edges_csv = interim_dir / "edges_raw.csv"

    output_files = [drugs_csv, ingredients_csv, diseases_csv, edges_csv]
    if not force and all(path.exists() for path in output_files):
        logger.info("Interim outputs already exist. Skipping step (use --force to re-run).")
        return

    drug_rows: List[List[Any]] = []
    ingredient_rows: List[List[Any]] = []
    disease_rows: List[List[Any]] = []
    edge_rows: List[List[Any]] = []

    # Sets used to avoid duplicated entity rows in the interim tables
    seen_drugs = set()
    seen_ingredients = set()
    seen_diseases = set()

    for folder_name in SOURCE_FOLDERS:
        folder_path = input_dir / folder_name

        if not folder_path.exists():
            logger.warning("Missing input folder: %s", folder_path)
            continue

        for json_file in iter_json_files(folder_path):
            data = load_json_file(json_file)

            drug_records = extract_drugs(data)
            ingredient_records = extract_ingredients(data)

            # -----------------------------------------------------------------
            # Drug table
            # -----------------------------------------------------------------
            for drug in drug_records:
                drug_name = drug.get("name")
                if not isinstance(drug_name, str) or not drug_name.strip():
                    continue

                name_original = clean_whitespace(drug_name)
                name_norm = normalize_name(name_original)

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

            # -----------------------------------------------------------------
            # Ingredient table
            # -----------------------------------------------------------------
            for ingredient in ingredient_records:
                ingredient_name = ingredient.get("name")
                ingredient_code = ingredient.get("code")

                name_original = (
                    clean_whitespace(ingredient_name)
                    if isinstance(ingredient_name, str) and ingredient_name.strip()
                    else ""
                )
                name_norm = normalize_name(name_original) if name_original else ""
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

            # -----------------------------------------------------------------
            # Disease table and relation edges
            # -----------------------------------------------------------------
            primary_drug_name_original = ""
            primary_drug_name_norm = ""

            if drug_records:
                first_drug_name = drug_records[0].get("name")
                if isinstance(first_drug_name, str) and first_drug_name.strip():
                    primary_drug_name_original = clean_whitespace(first_drug_name)
                    primary_drug_name_norm = normalize_name(primary_drug_name_original)

            if not primary_drug_name_norm:
                continue

            # Drug -> Disease edges for indications and contraindications
            for relation_type, section_name in (
                ("INDICATION", "indications"),
                ("CONTRAINDICATION", "contraindications"),
            ):
                doid_ids = extract_doid_ids(data, section_name)

                for doid_id in doid_ids:
                    if doid_id not in seen_diseases:
                        seen_diseases.add(doid_id)
                        disease_rows.append([doid_id])

                    edge_rows.append([
                        primary_drug_name_norm,
                        relation_type,
                        doid_id,
                        folder_name,
                        json_file.name,
                    ])

            # Drug -> Ingredient edges
            for ingredient in ingredient_records:
                ingredient_name = ingredient.get("name")
                ingredient_code = ingredient.get("code")

                name_original = (
                    clean_whitespace(ingredient_name)
                    if isinstance(ingredient_name, str) and ingredient_name.strip()
                    else ""
                )
                name_norm = normalize_name(name_original) if name_original else ""
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
    write_csv(diseases_csv, ["doid"], disease_rows)
    write_csv(edges_csv, ["src", "rel", "dst", "source", "file"], edge_rows)

    logger.info("Saved: %s", drugs_csv)
    logger.info("Saved: %s", ingredients_csv)
    logger.info("Saved: %s", diseases_csv)
    logger.info("Saved: %s", edges_csv)


# Alias used by the pipeline runner
run = parse_sources


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"

    parse_sources(cfg={}, paths=Paths(), force=True)
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)

SOURCE_FOLDERS = ("xml_files", "txt_files", "csv_files")


def iter_json_files(folder: Path) -> Iterable[Path]:
    """
    Yield all JSON files in a given folder.

    Args:
        folder: Folder to scan.

    Yields:
        Path objects for files with the .json extension.
    """
    if not folder.exists():
        return

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            yield file_path


def load_json_file(path: Path) -> Dict[str, Any]:
    """
    Load a JSON file and raise a clear error message if parsing fails.

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content as a dictionary.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON file: {path} | Error: {exc}") from exc


def extract_drugs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract valid drug records from the input JSON structure.

    Args:
        data: Parsed JSON content.

    Returns:
        A list of drug dictionaries.
    """
    drugs = data.get("drug", [])
    if isinstance(drugs, list):
        return [drug for drug in drugs if isinstance(drug, dict)]
    return []


def extract_doid_ids(data: Dict[str, Any]) -> List[str]:
    """
    Extract DOID identifiers from indications and contraindications.

    Args:
        data: Parsed JSON content.

    Returns:
        A list of DOID identifiers.
    """
    doid_ids: List[str] = []

    for section_name in ("indications", "contraindications"):
        section = data.get(section_name, {})
        if not isinstance(section, dict):
            continue

        entities = section.get("doid_entities", [])
        if not isinstance(entities, list):
            continue

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            doid_id = entity.get("doid_id")
            if isinstance(doid_id, str) and doid_id.strip():
                doid_ids.append(doid_id.strip())

    return doid_ids


def extract_ordo_ids(data: Dict[str, Any]) -> List[str]:
    """
    Extract ORDO identifiers from indications and contraindications.

    Args:
        data: Parsed JSON content.

    Returns:
        A list of ORDO identifiers.
    """
    ordo_ids: List[str] = []

    for section_name in ("indications", "contraindications"):
        section = data.get(section_name, {})
        if not isinstance(section, dict):
            continue

        entities = section.get("orphanet_entities", [])
        if not isinstance(entities, list):
            continue

        for entity in entities:
            if not isinstance(entity, dict):
                continue

            ordo_id = entity.get("ordo_id")
            if isinstance(ordo_id, str) and ordo_id.strip():
                ordo_ids.append(ordo_id.strip())

    return ordo_ids


def extract_ingredients(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract valid ingredient records from the input JSON structure.

    Args:
        data: Parsed JSON content.

    Returns:
        A list of ingredient dictionaries.
    """
    ingredients = data.get("ingredients", [])
    if isinstance(ingredients, list):
        return [ingredient for ingredient in ingredients if isinstance(ingredient, dict)]
    return []


def count_json_files(input_dir: Path) -> Dict[str, int]:
    """
    Count the number of JSON files available in each source folder.

    Args:
        input_dir: Root input directory.

    Returns:
        Dictionary mapping folder name to JSON file count.
    """
    counts: Dict[str, int] = {}

    for folder_name in SOURCE_FOLDERS:
        folder_path = input_dir / folder_name
        counts[folder_name] = sum(1 for _ in iter_json_files(folder_path)) if folder_path.exists() else 0

    return counts


def run_stats(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Generate an audit report with basic statistics about the input JSON files.

    This step scans all JSON files available in the MedJsonify-derived input folders
    and reports:
        - number of input files per folder
        - number of unique drugs
        - number of unique identifiers (DrugBank, ChEBI)
        - number of unique ingredients
        - most frequent DOID and ORDO occurrences

    Outputs:
        - data/output/reports/audit_report.json
        - data/output/reports/top_doids.csv

    Args:
        cfg: Pipeline configuration dictionary. Kept for interface consistency.
        paths: Object containing pipeline paths. Must provide `input_dir` and `output_dir`.
        force: If True, overwrite existing outputs.
    """
    input_dir = Path(paths.input_dir)
    output_dir = Path(paths.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_report_path = reports_dir / "audit_report.json"
    top_doids_csv_path = reports_dir / "top_doids.csv"

    if not force and audit_report_path.exists() and top_doids_csv_path.exists():
        logger.info("Audit outputs already exist. Skipping step (use --force to re-run).")
        return

    file_counts = count_json_files(input_dir)

    drug_names = set()
    drugbank_ids = set()
    chebi_ids = set()
    ingredient_names = set()
    ingredient_codes = set()
    doid_counts = Counter()
    ordo_counts = Counter()

    for folder_name in SOURCE_FOLDERS:
        folder_path = input_dir / folder_name

        if not folder_path.exists():
            logger.warning("Missing input folder: %s", folder_path)
            continue

        for json_file in iter_json_files(folder_path):
            data = load_json_file(json_file)

            for drug in extract_drugs(data):
                name = drug.get("name")
                if isinstance(name, str) and name.strip():
                    drug_names.add(name.strip())

                drugbank_id = drug.get("drugbank_id")
                if isinstance(drugbank_id, str) and drugbank_id.strip():
                    drugbank_ids.add(drugbank_id.strip())

                chebi_id = drug.get("chebi_id")
                if isinstance(chebi_id, str) and chebi_id.strip():
                    chebi_ids.add(chebi_id.strip())

            for doid_id in extract_doid_ids(data):
                doid_counts[doid_id] += 1

            for ordo_id in extract_ordo_ids(data):
                ordo_counts[ordo_id] += 1

            for ingredient in extract_ingredients(data):
                ingredient_name = ingredient.get("name")
                if isinstance(ingredient_name, str) and ingredient_name.strip():
                    ingredient_names.add(ingredient_name.strip())

                ingredient_code = ingredient.get("code")
                if isinstance(ingredient_code, str) and ingredient_code.strip():
                    ingredient_codes.add(ingredient_code.strip())

    report = {
        "inputs": {
            "input_dir": str(input_dir),
            "file_counts": file_counts,
        },
        "entities": {
            "unique_drug_names": len(drug_names),
            "unique_drugbank_ids": len(drugbank_ids),
            "unique_chebi_ids": len(chebi_ids),
            "unique_ingredient_names": len(ingredient_names),
            "unique_ingredient_codes": len(ingredient_codes),
            "unique_doids": len(doid_counts),
            "unique_ordos": len(ordo_counts),
        },
        "top_doids": doid_counts.most_common(20),
        "top_ordos": ordo_counts.most_common(20),
    }

    audit_report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved audit report: %s", audit_report_path)

    csv_lines = ["doid,count"]
    for doid_id, count in doid_counts.most_common(200):
        csv_lines.append(f"{doid_id},{count}")

    top_doids_csv_path.write_text("\n".join(csv_lines), encoding="utf-8")
    logger.info("Saved top DOIDs CSV: %s", top_doids_csv_path)


# Alias used by the pipeline runner
run = run_stats


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"

    run_stats(cfg={}, paths=Paths(), force=True)
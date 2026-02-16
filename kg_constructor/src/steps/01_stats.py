from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)

def get_json_files(folder: Path) -> Iterable[Path]:
    """Return all .json files in a folder."""
    if not folder.exists():
        return
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == ".json":
            yield file


def load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file and raise a clear error if it fails."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} | Error: {e}") from e


def get_drugs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list under 'drug' if present."""
    drugs = data.get("drug", [])
    if isinstance(drugs, list):
        return [d for d in drugs if isinstance(d, dict)]
    return []


def get_doid_ids(data: Dict[str, Any]) -> List[str]:
    """Extract DOID IDs from indications and contraindications."""
    doid_ids: List[str] = []
    for section in ("indications", "contraindications"):
        block = data.get(section, {})
        if not isinstance(block, dict):
            continue
        entities = block.get("doid_entities", [])
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if isinstance(entity, dict):
                doid = entity.get("doid_id")
                if isinstance(doid, str) and doid.strip():
                    doid_ids.append(doid.strip())
    return doid_ids


def get_ordo_ids(data: Dict[str, Any]) -> List[str]:
    """Extract ORDO IDs from indications and contraindications."""
    ordo_ids: List[str] = []
    for section in ("indications", "contraindications"):
        block = data.get(section, {})
        if not isinstance(block, dict):
            continue
        entities = block.get("orphanet_entities", [])
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if isinstance(entity, dict):
                ordo = entity.get("ordo_id")
                if isinstance(ordo, str) and ordo.strip():
                    ordo_ids.append(ordo.strip())
    return ordo_ids


def get_ingredients(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ingredient records from 'ingredients' if present."""
    ingredients = data.get("ingredients", [])
    if isinstance(ingredients, list):
        return [item for item in ingredients if isinstance(item, dict)]
    return []


def count_json_files(input_dir: Path) -> Dict[str, int]:
    """Count number of JSON files in each input folder."""
    counts = {}
    for folder_name in ("xml_files", "txt_files", "csv_files"):
        folder = input_dir / folder_name
        counts[folder_name] = sum(1 for _ in get_json_files(folder)) if folder.exists() else 0
    return counts



def run_stats(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 01: Generate a report with statistics about the input JSON files.
    Outputs:
      - data/output/reports/audit_report.json
      - data/output/reports/top_doids.csv
    """
    input_dir = Path(paths.input_dir)
    output_dir = Path(paths.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_report_path = reports_dir / "audit_report.json"
    top_doids_csv_path = reports_dir / "top_doids.csv"

    if not force and audit_report_path.exists() and top_doids_csv_path.exists():
        logger.info("Audit outputs already exist. Skipping (use --force to re-run).")
        return

    file_counts = count_json_files(input_dir)

    drug_names = set()
    drugbank_ids = set()
    chebi_ids = set()
    ingredient_names = set()
    ingredient_codes = set()
    doid_counts = Counter()
    ordo_counts = Counter()

    for folder_name in ("xml_files", "txt_files", "csv_files"):
        folder = input_dir / folder_name
        if not folder.exists():
            logger.warning(f"Missing input folder: {folder}")
            continue
        for json_file in get_json_files(folder):
            data = load_json(json_file)
            for drug in get_drugs(data):
                name = drug.get("name")
                if isinstance(name, str) and name.strip():
                    drug_names.add(name.strip())
                dbid = drug.get("drugbank_id")
                if isinstance(dbid, str) and dbid.strip():
                    drugbank_ids.add(dbid.strip())
                chebi = drug.get("chebi_id")
                if isinstance(chebi, str) and chebi.strip():
                    chebi_ids.add(chebi.strip())
            for doid in get_doid_ids(data):
                doid_counts[doid] += 1
            for ordo in get_ordo_ids(data):
                ordo_counts[ordo] += 1
            for ingredient in get_ingredients(data):
                name = ingredient.get("name")
                if isinstance(name, str) and name.strip():
                    ingredient_names.add(name.strip())
                code = ingredient.get("code")
                if isinstance(code, str) and code.strip():
                    ingredient_codes.add(code.strip())

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

    audit_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved audit report: {audit_report_path}")

    lines = ["doid,count"]
    for doid, count in doid_counts.most_common(200):
        lines.append(f"{doid},{count}")
    top_doids_csv_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved top DOIDs CSV: {top_doids_csv_path}")

# Alias for pipeline compatibility
run = run_stats

if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"
    run_stats(config={}, paths=Paths(), force=True)

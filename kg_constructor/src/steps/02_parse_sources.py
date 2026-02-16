"""
Step 02: Parse input JSON files and generate clean, normalized CSV tables for drugs, ingredients, diseases, and edges.
All names and IDs are normalized here para garantir consistência e facilitar o mapeamento canônico nos próximos passos.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)

DOID_PATTERN = re.compile(r"(DOID[_:])(\d+)$", re.IGNORECASE)



def get_json_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() == ".json":
            yield file


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} | Error: {e}") from e


def clean_whitespace(text: str) -> str:
    return " ".join(text.strip().split())


def normalize_name(text: str) -> str:
    """
    Normalize a drug, ingredient, or disease name for canonical mapping.
    Removes extra whitespace and converts to uppercase.
    """
    return clean_whitespace(text).upper()


def normalize_doid(doid_raw: str) -> str:
    """
    Normalize a DOID identifier to the canonical format 'DOID:xxxx'.
    Handles URLs, DOID:xxxx, and fallback cases.
    """
    s = doid_raw.strip()
    # URL case
    if s.lower().startswith("http"):
        s = s.rsplit("/", 1)[-1]
    m = DOID_PATTERN.search(s)
    if m:
        return f"DOID:{m.group(2)}"
    # Already DOID:xxxx
    if s.upper().startswith("DOID:"):
        return f"DOID:{s.split(':', 1)[1].strip()}"
    # Fallback: return as-is (still helpful for debugging)
    return s


def get_drugs(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    drugs = data.get("drug", [])
    if isinstance(drugs, list):
        return [d for d in drugs if isinstance(d, dict)]
    return []


def get_ingredients(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    ingredients = data.get("ingredients", [])
    if isinstance(ingredients, list):
        return [x for x in ingredients if isinstance(x, dict)]
    return []


def get_doid_ids(data: Dict[str, Any], section: str) -> List[str]:
    block = data.get(section, {})
    if not isinstance(block, dict):
        return []
    entities = block.get("doid_entities", [])
    if not isinstance(entities, list):
        return []
    doid_ids: List[str] = []
    for entity in entities:
        if isinstance(entity, dict):
            d = entity.get("doid_id")
            if isinstance(d, str) and d.strip():
                doid_ids.append(normalize_doid(d))
    return doid_ids


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)



def parse_sources(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 02: Parse input JSON files and generate clean CSV tables.
    Outputs (data/output/interim):
      - drugs_raw.csv
      - ingredients_raw.csv
      - diseases_raw.csv
      - edges_raw.csv
    """
    input_dir = Path(paths.input_dir)
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)

    drugs_csv = interim_dir / "drugs_raw.csv"
    ingredients_csv = interim_dir / "ingredients_raw.csv"
    diseases_csv = interim_dir / "diseases_raw.csv"
    edges_csv = interim_dir / "edges_raw.csv"

    if not force and all(p.exists() for p in [drugs_csv, ingredients_csv, diseases_csv, edges_csv]):
        logger.info("Interim outputs already exist. Skipping (use --force to re-run).")
        return

    drug_rows: List[List[Any]] = []
    ingredient_rows: List[List[Any]] = []
    disease_rows: List[List[Any]] = []
    edge_rows: List[List[Any]] = []

    # Use sets to avoid excessive duplicates in diseases table
    seen_diseases = set()
    seen_drugs = set()
    seen_ingredients = set()

    for folder_name in ("xml_files", "txt_files", "csv_files"):
        folder = input_dir / folder_name
        if not folder.exists():
            logger.warning(f"Missing input folder: {folder}")
            continue

        for json_file in get_json_files(folder):
            data = load_json(json_file)

            # Drugs
            for drug in get_drugs(data):
                name = drug.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue

                name_original = clean_whitespace(name)
                name_norm = normalize_name(name_original)

                drugbank_id = drug.get("drugbank_id") if isinstance(drug.get("drugbank_id"), str) else ""
                chebi_id = drug.get("chebi_id") if isinstance(drug.get("chebi_id"), str) else ""

                key = (name_norm, drugbank_id.strip(), chebi_id.strip())
                if key not in seen_drugs:
                    seen_drugs.add(key)
                    drug_rows.append([folder_name, name_original, name_norm, drugbank_id.strip(), chebi_id.strip(), json_file.name])

            # Ingredients
            for ingredient in get_ingredients(data):
                iname = ingredient.get("name")
                icode = ingredient.get("code")

                iname_original = clean_whitespace(iname) if isinstance(iname, str) and iname.strip() else ""
                iname_norm = normalize_name(iname_original) if iname_original else ""
                icode_val = clean_whitespace(icode) if isinstance(icode, str) and icode.strip() else ""

                if not iname_norm and not icode_val:
                    continue

                ikey = (iname_norm, icode_val)
                if ikey not in seen_ingredients:
                    seen_ingredients.add(ikey)
                    ingredient_rows.append([folder_name, iname_original, iname_norm, icode_val, json_file.name])

            # Diseases and Edges
            drug_recs = get_drugs(data)
            primary_drug_name_norm = ""
            primary_drug_name_original = ""
            if drug_recs:
                dn = drug_recs[0].get("name")
                if isinstance(dn, str) and dn.strip():
                    primary_drug_name_original = clean_whitespace(dn)
                    primary_drug_name_norm = normalize_name(primary_drug_name_original)

            if primary_drug_name_norm:
                for rel_type, section in [("INDICATION", "indications"), ("CONTRAINDICATION", "contraindications")]:
                    doids = get_doid_ids(data, section)
                    for doid in doids:
                        if doid not in seen_diseases:
                            seen_diseases.add(doid)
                            disease_rows.append([doid])

                        edge_rows.append([
                            primary_drug_name_norm,
                            rel_type,
                            doid,
                            folder_name,
                            json_file.name
                        ])

                # HAS_INGREDIENT edges
                for ingredient in get_ingredients(data):
                    iname = ingredient.get("name")
                    icode = ingredient.get("code")

                    iname_original = clean_whitespace(iname) if isinstance(iname, str) and iname.strip() else ""
                    iname_norm = normalize_name(iname_original) if iname_original else ""
                    icode_val = clean_whitespace(icode) if isinstance(icode, str) and icode.strip() else ""

                    if icode_val:
                        ing_id = f"UNII:{icode_val}"
                    elif iname_norm:
                        ing_id = f"ING:{iname_norm}"
                    else:
                        continue

                    edge_rows.append([
                        primary_drug_name_norm,
                        "HAS_INGREDIENT",
                        ing_id,
                        folder_name,
                        json_file.name
                    ])

    write_csv(drugs_csv, ["source", "name_original", "name_norm", "drugbank_id", "chebi_id", "file"], drug_rows)
    write_csv(ingredients_csv, ["source", "name_original", "name_norm", "unii_code", "file"], ingredient_rows)
    write_csv(diseases_csv, ["doid"], disease_rows)
    write_csv(edges_csv, ["src", "rel", "dst", "source", "file"], edge_rows)

    logger.info(f"Saved: {drugs_csv}")
    logger.info(f"Saved: {ingredients_csv}")
    logger.info(f"Saved: {diseases_csv}")
    logger.info(f"Saved: {edges_csv}")


# Alias for pipeline compatibility
run = parse_sources


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"
    parse_sources(config={}, paths=Paths(), force=True)

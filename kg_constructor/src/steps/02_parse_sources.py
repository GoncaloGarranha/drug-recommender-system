from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)

DOID_URL_RE = re.compile(r"(DOID[_:])(\d+)$", re.IGNORECASE)


def _iter_json_files(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            yield p


def _safe_load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} | Error: {e}") from e


def _normalize_whitespace(s: str) -> str:
    return " ".join(s.strip().split())


def _normalize_name(s: str) -> str:
    # Keep it simple and deterministic (good for a thesis)
    return _normalize_whitespace(s).upper()


def _normalize_doid(doid_raw: str) -> str:
    """
    Convert DOID identifiers to a canonical format: DOID:xxxx
    Accepts:
      - http://purl.obolibrary.org/obo/DOID_10763
      - DOID_10763
      - DOID:10763
    """
    s = doid_raw.strip()

    # URL case
    if s.lower().startswith("http"):
        tail = s.rsplit("/", 1)[-1]  # e.g., DOID_10763
        s = tail

    m = DOID_URL_RE.search(s)
    if m:
        return f"DOID:{m.group(2)}"

    # Already DOID:xxxx
    if s.upper().startswith("DOID:"):
        return f"DOID:{s.split(':', 1)[1].strip()}"

    # Fallback: return as-is (still helpful for debugging)
    return s


def _extract_drug_records(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    drug = obj.get("drug", [])
    if isinstance(drug, list):
        return [d for d in drug if isinstance(d, dict)]
    return []


def _extract_ingredients(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    ing = obj.get("ingredients", [])
    if isinstance(ing, list):
        return [x for x in ing if isinstance(x, dict)]
    return []


def _extract_doid_entities(obj: Dict[str, Any], block_key: str) -> List[str]:
    block = obj.get(block_key, {})
    if not isinstance(block, dict):
        return []

    ents = block.get("doid_entities", [])
    if not isinstance(ents, list):
        return []

    doids: List[str] = []
    for e in ents:
        if isinstance(e, dict):
            d = e.get("doid_id")
            if isinstance(d, str) and d.strip():
                doids.append(_normalize_doid(d))
    return doids


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 02_parse_sources: parse raw JSON files into clean intermediate tables.

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

    drugs_path = interim_dir / "drugs_raw.csv"
    ingredients_path = interim_dir / "ingredients_raw.csv"
    diseases_path = interim_dir / "diseases_raw.csv"
    edges_path = interim_dir / "edges_raw.csv"

    if not force and all(p.exists() for p in [drugs_path, ingredients_path, diseases_path, edges_path]):
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

    # We parse all sources, but most relations come from xml_files
    sources = ("xml_files", "txt_files", "csv_files")

    for src in sources:
        folder = input_dir / src
        if not folder.exists():
            logger.warning(f"Missing input folder: {folder}")
            continue

        for jf in _iter_json_files(folder):
            obj = _safe_load_json(jf)

            # --- Drugs (raw) ---
            for d in _extract_drug_records(obj):
                name = d.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue

                name_original = _normalize_whitespace(name)
                name_norm = _normalize_name(name_original)

                drugbank_id = d.get("drugbank_id") if isinstance(d.get("drugbank_id"), str) else ""
                chebi_id = d.get("chebi_id") if isinstance(d.get("chebi_id"), str) else ""

                key = (name_norm, drugbank_id.strip(), chebi_id.strip())
                if key not in seen_drugs:
                    seen_drugs.add(key)
                    drug_rows.append([src, name_original, name_norm, drugbank_id.strip(), chebi_id.strip(), jf.name])

            # --- Ingredients (mostly from xml_files) ---
            for ing in _extract_ingredients(obj):
                iname = ing.get("name")
                icode = ing.get("code")

                iname_original = _normalize_whitespace(iname) if isinstance(iname, str) and iname.strip() else ""
                iname_norm = _normalize_name(iname_original) if iname_original else ""
                icode_val = _normalize_whitespace(icode) if isinstance(icode, str) and icode.strip() else ""

                if not iname_norm and not icode_val:
                    continue

                ikey = (iname_norm, icode_val)
                if ikey not in seen_ingredients:
                    seen_ingredients.add(ikey)
                    ingredient_rows.append([src, iname_original, iname_norm, icode_val, jf.name])

            # --- Diseases + Edges (from indications/contraindications) ---
            # We attach edges to drug names for now; Step 03 will resolve canonical IDs.
            # Use the first drug name in the file as the "primary" drug reference.
            drug_recs = _extract_drug_records(obj)
            primary_drug_name_norm = ""
            primary_drug_name_original = ""
            if drug_recs:
                dn = drug_recs[0].get("name")
                if isinstance(dn, str) and dn.strip():
                    primary_drug_name_original = _normalize_whitespace(dn)
                    primary_drug_name_norm = _normalize_name(primary_drug_name_original)

            if primary_drug_name_norm:
                for rel_key, block_key in [("INDICATION", "indications"), ("CONTRAINDICATION", "contraindications")]:
                    doids = _extract_doid_entities(obj, block_key)
                    for doid in doids:
                        if doid not in seen_diseases:
                            seen_diseases.add(doid)
                            disease_rows.append([doid])

                        edge_rows.append([
                            primary_drug_name_norm,
                            rel_key,
                            doid,
                            src,
                            jf.name
                        ])

                # HAS_INGREDIENT edges
                for ing in _extract_ingredients(obj):
                    iname = ing.get("name")
                    icode = ing.get("code")

                    iname_original = _normalize_whitespace(iname) if isinstance(iname, str) and iname.strip() else ""
                    iname_norm = _normalize_name(iname_original) if iname_original else ""
                    icode_val = _normalize_whitespace(icode) if isinstance(icode, str) and icode.strip() else ""

                    # Define an ingredient key string (stable, deterministic)
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
                        src,
                        jf.name
                    ])

    _write_csv(drugs_path, ["source", "name_original", "name_norm", "drugbank_id", "chebi_id", "file"], drug_rows)
    _write_csv(ingredients_path, ["source", "name_original", "name_norm", "unii_code", "file"], ingredient_rows)
    _write_csv(diseases_path, ["doid"], disease_rows)
    _write_csv(edges_path, ["src", "rel", "dst", "source", "file"], edge_rows)

    logger.info(f"Saved: {drugs_path}")
    logger.info(f"Saved: {ingredients_path}")
    logger.info(f"Saved: {diseases_path}")
    logger.info(f"Saved: {edges_path}")

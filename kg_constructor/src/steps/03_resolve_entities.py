from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def resolve_entities(cfg: Dict[str, Any], paths, force: bool = False) -> None:
        """
        Step 03: Build canonical entity IDs and create alias mappings for drugs, ingredients, and diseases.
        All drug name variants (brand, generic, synonyms) are mapped to a single canonical entity ID.
        Inputs:
            - data/output/interim/drugs_raw.csv
            - data/output/interim/ingredients_raw.csv
            - data/output/interim/edges_raw.csv
        Outputs:
            - data/output/mappings/entity_table.csv (canonical entities)
            - data/output/mappings/aliases.csv (all aliases mapped to canonical IDs)
            - data/output/mappings/resolution_report.json (stats)
            - data/output/mappings/drug_name_to_entity.json (mapping dict)
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

    if not force and all(p.exists() for p in [entity_table_csv, aliases_csv, report_json, drug_map_json]):
        logger.info("Mapping outputs already exist. Skipping (use --force to re-run).")
        return

    if not drugs_csv.exists() or not edges_csv.exists():
        raise FileNotFoundError("Missing interim inputs. Run step 02_parse_sources first.")

    # Read input CSVs as lists of dicts
    drug_rows = read_csv(drugs_csv)
    ingredient_rows = read_csv(ingredients_csv) if ingredients_csv.exists() else []
    edge_rows = read_csv(edges_csv)

    # --- DRUG ENTITY RESOLUTION ---
    # Map normalized drug names to DrugBank IDs (if available)
    normalized_name_to_drugbank_ids: Dict[str, Counter] = defaultdict(Counter)
    for drug_row in drug_rows:
        normalized_name = (drug_row.get("name_norm") or "").strip()
        drugbank_id = (drug_row.get("drugbank_id") or "").strip()
        if not normalized_name:
            continue
        if drugbank_id:
            normalized_name_to_drugbank_ids[normalized_name][drugbank_id] += 1

    # Map each normalized name to a canonical entity ID (DrugBank or fallback)
    drug_name_to_entity_id: Dict[str, str] = {}
    drug_resolution_stats = Counter()
    for drug_row in drug_rows:
        normalized_name = (drug_row.get("name_norm") or "").strip()
        if not normalized_name:
            continue
        if normalized_name_to_drugbank_ids.get(normalized_name):
            best_dbid, _ = normalized_name_to_drugbank_ids[normalized_name].most_common(1)[0]
            entity_id = f"DRUGBANK:{best_dbid}"
            drug_resolution_stats["resolved_by_drugbank"] += 1
        else:
            entity_id = f"DRUG:{normalized_name}"
            drug_resolution_stats["resolved_by_name"] += 1
        drug_name_to_entity_id[normalized_name] = entity_id

    # Resolve INGREDIENT entities
    ingredient_entities: Dict[str, Tuple[str, str]] = {}

    for r in ingredients:
        name_norm = (r.get("name_norm") or "").strip()
        unii = (r.get("unii_code") or "").strip()

        if unii:
            iid = f"UNII:{unii}"
            label = unii if not name_norm else name_norm
            ingredient_entities[iid] = ("INGREDIENT", label)
        elif name_norm:
            iid = f"ING:{name_norm}"
            ingredient_entities[iid] = ("INGREDIENT", name_norm)

    # Resolve DISEASE entities
    disease_entities: Dict[str, Tuple[str, str]] = {}
    for e in edges:
        dst = (e.get("dst") or "").strip()
        if dst.startswith("DOID:"):
            disease_entities[dst] = ("DISEASE", dst)

    # Build entity_table.csv
    entity_rows: List[List[Any]] = []

    # Drugs
    unique_drug_entities = set(drug_name_to_entity.values())
    for eid in sorted(unique_drug_entities):
        label = eid.replace("DRUGBANK:", "").replace("DRUG:", "")
        entity_rows.append([eid, "DRUG", label])

    # Diseases
    for eid in sorted(disease_entities.keys()):
        entity_rows.append([eid, "DISEASE", eid])

    # Ingredients
    for eid, (etype, label) in sorted(ingredient_entities.items()):
        entity_rows.append([eid, etype, label])

    write_csv(entity_table_csv, ["entity_id", "entity_type", "label"], entity_rows)
    logger.info(f"Saved: {entity_table_csv}")

    # Build aliases.csv
    alias_rows: List[List[Any]] = []

    for r in drugs:
        name_original = (r.get("name_original") or "").strip()
        name_norm = (r.get("name_norm") or "").strip()
        if not name_norm:
            continue

        eid = drug_name_to_entity.get(name_norm)
        if not eid:
            continue

        if name_original:
            alias_rows.append([name_original, eid, "DRUG_NAME_ORIGINAL"])
        alias_rows.append([name_norm, eid, "DRUG_NAME_NORM"])

    for r in ingredients:
        name_original = (r.get("name_original") or "").strip()
        name_norm = (r.get("name_norm") or "").strip()
        unii = (r.get("unii_code") or "").strip()

        if unii:
            iid = f"UNII:{unii}"
        elif name_norm:
            iid = f"ING:{name_norm}"
        else:
            continue

        if name_original:
            alias_rows.append([name_original, iid, "ING_NAME_ORIGINAL"])
        if name_norm:
            alias_rows.append([name_norm, iid, "ING_NAME_NORM"])
        if unii:
            alias_rows.append([unii, iid, "UNII_CODE"])

    write_csv(aliases_csv, ["alias", "entity_id", "alias_type"], alias_rows)
    logger.info(f"Saved: {aliases_csv}")

    drug_map_json.write_text(json.dumps(drug_name_to_entity, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {drug_map_json}")
    report = {
        "drug_resolution": dict(drug_resolution_stats),
        "unique_drug_entities": len(unique_drug_entities),
        "unique_disease_entities": len(disease_entities),
        "unique_ingredient_entities": len(ingredient_entities),
        "note": "Drug entities prefer DRUGBANK:<id> when available; otherwise DRUG:<NAME_NORM>.",
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {report_json}")


# Alias for pipeline compatibility
run = resolve_entities


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"
    resolve_entities(config={}, paths=Paths(), force=True)

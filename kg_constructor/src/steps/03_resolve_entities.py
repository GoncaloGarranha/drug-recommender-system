from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 03_resolve_entities: build canonical entity IDs and alias mappings.

    Inputs:
      - data/output/interim/drugs_raw.csv
      - data/output/interim/ingredients_raw.csv
      - data/output/interim/edges_raw.csv

    Outputs:
      - data/output/mappings/entity_table.csv
      - data/output/mappings/aliases.csv
      - data/output/mappings/resolution_report.json
      - data/output/mappings/drug_name_to_entity.json
    """
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    mappings_dir = output_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    drugs_path = interim_dir / "drugs_raw.csv"
    ingredients_path = interim_dir / "ingredients_raw.csv"
    edges_path = interim_dir / "edges_raw.csv"

    entity_table_path = mappings_dir / "entity_table.csv"
    aliases_path = mappings_dir / "aliases.csv"
    report_path = mappings_dir / "resolution_report.json"
    drug_map_path = mappings_dir / "drug_name_to_entity.json"

    if not force and all(p.exists() for p in [entity_table_path, aliases_path, report_path, drug_map_path]):
        logger.info("Mapping outputs already exist. Skipping (use --force to re-run).")
        return

    if not drugs_path.exists() or not edges_path.exists():
        raise FileNotFoundError("Missing interim inputs. Run step 02_parse_sources first.")

    drugs = _read_csv(drugs_path)
    ingredients = _read_csv(ingredients_path) if ingredients_path.exists() else []
    edges = _read_csv(edges_path)

    # ------------------------------------------------------------
    # 1) Resolve DRUG entities: build name_norm -> canonical entity_id
    # ------------------------------------------------------------
    name_to_drugbank_ids: Dict[str, Counter] = defaultdict(Counter)

    for r in drugs:
        name_norm = (r.get("name_norm") or "").strip()
        drugbank_id = (r.get("drugbank_id") or "").strip()

        if not name_norm:
            continue
        if drugbank_id:
            name_to_drugbank_ids[name_norm][drugbank_id] += 1

    drug_name_to_entity: Dict[str, str] = {}
    drug_resolution_stats = Counter()

    for r in drugs:
        name_norm = (r.get("name_norm") or "").strip()
        if not name_norm:
            continue

        # If the same name_norm appears with multiple drugbank IDs, pick the most frequent
        if name_to_drugbank_ids.get(name_norm):
            best_dbid, _ = name_to_drugbank_ids[name_norm].most_common(1)[0]
            entity_id = f"DRUGBANK:{best_dbid}"
            drug_resolution_stats["resolved_by_drugbank"] += 1
        else:
            entity_id = f"DRUG:{name_norm}"
            drug_resolution_stats["resolved_by_name"] += 1

        drug_name_to_entity[name_norm] = entity_id

    # ------------------------------------------------------------
    # 2) Resolve INGREDIENT entities
    # ------------------------------------------------------------
    ingredient_entities: Dict[str, Tuple[str, str]] = {}
    # entity_id -> (type, label)
    # We'll use UNII if available, else ING:<NAME_NORM>

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

    # ------------------------------------------------------------
    # 3) Resolve DISEASE entities (from edges destination)
    # ------------------------------------------------------------
    disease_entities: Dict[str, Tuple[str, str]] = {}
    for e in edges:
        dst = (e.get("dst") or "").strip()
        if dst.startswith("DOID:"):
            disease_entities[dst] = ("DISEASE", dst)

    # ------------------------------------------------------------
    # 4) Build entity_table.csv
    # ------------------------------------------------------------
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

    _write_csv(entity_table_path, ["entity_id", "entity_type", "label"], entity_rows)
    logger.info(f"Saved: {entity_table_path}")

    # ------------------------------------------------------------
    # 5) Build aliases.csv
    # ------------------------------------------------------------
    alias_rows: List[List[Any]] = []

    # Drug aliases: map both original and normalized names to the canonical entity
    # We use drugs_raw for original name coverage
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

    # Ingredient aliases
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

    _write_csv(aliases_path, ["alias", "entity_id", "alias_type"], alias_rows)
    logger.info(f"Saved: {aliases_path}")

    # ------------------------------------------------------------
    # 6) Save drug_name_to_entity.json (useful for edge remapping + inference)
    # ------------------------------------------------------------
    drug_map_path.write_text(json.dumps(drug_name_to_entity, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {drug_map_path}")

    # ------------------------------------------------------------
    # 7) Report
    # ------------------------------------------------------------
    report = {
        "drug_resolution": dict(drug_resolution_stats),
        "unique_drug_entities": len(unique_drug_entities),
        "unique_disease_entities": len(disease_entities),
        "unique_ingredient_entities": len(ingredient_entities),
        "note": "Drug entities prefer DRUGBANK:<id> when available; otherwise DRUG:<NAME_NORM>.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {report_path}")

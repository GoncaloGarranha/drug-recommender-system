from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def _iter_json_files(folder: Path) -> Iterable[Path]:
    """Yield all .json files under a folder (non-recursive)."""
    if not folder.exists():
        return
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() == ".json":
            yield p


def _safe_load_json(path: Path) -> Dict[str, Any]:
    """Load JSON safely, raising a clear error if parsing fails."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path} | Error: {e}") from e


def _extract_drug_records(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return the list under 'drug' if present. Many files store drug info as a list.
    """
    drug = obj.get("drug", [])
    if isinstance(drug, list):
        return [d for d in drug if isinstance(d, dict)]
    return []


def _extract_doid_entities(obj: Dict[str, Any]) -> List[str]:
    """
    Extract DOID IDs from indications and contraindications.
    Expected structure:
      obj['indications']['doid_entities'] = [{"doid_id": "..."} ...]
    """
    doids: List[str] = []

    for block_key in ("indications", "contraindications"):
        block = obj.get(block_key, {})
        if not isinstance(block, dict):
            continue

        ents = block.get("doid_entities", [])
        if not isinstance(ents, list):
            continue

        for e in ents:
            if isinstance(e, dict):
                doid = e.get("doid_id")
                if isinstance(doid, str) and doid.strip():
                    doids.append(doid.strip())

    return doids


def _extract_ordo_entities(obj: Dict[str, Any]) -> List[str]:
    """
    Extract ORDO IDs from indications and contraindications (optional).
    Expected:
      obj['indications']['orphanet_entities'] = [{"ordo_id": "..."} ...]
    """
    ordos: List[str] = []

    for block_key in ("indications", "contraindications"):
        block = obj.get(block_key, {})
        if not isinstance(block, dict):
            continue

        ents = block.get("orphanet_entities", [])
        if not isinstance(ents, list):
            continue

        for e in ents:
            if isinstance(e, dict):
                ordo = e.get("ordo_id")
                if isinstance(ordo, str) and ordo.strip():
                    ordos.append(ordo.strip())

    return ordos


def _extract_ingredients(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract ingredient records from 'ingredients' (if present)."""
    ing = obj.get("ingredients", [])
    if isinstance(ing, list):
        return [x for x in ing if isinstance(x, dict)]
    return []


def _count_files(input_dir: Path) -> Dict[str, int]:
    """Count number of JSON files per source directory."""
    counts = {}
    for src in ("xml_files", "txt_files", "csv_files"):
        folder = input_dir / src
        counts[src] = sum(1 for _ in _iter_json_files(folder)) if folder.exists() else 0
    return counts


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 01_stats: build an audit report over the raw input JSON files.

    Outputs:
      - data/output/reports/audit_report.json
      - data/output/reports/top_doids.csv
    """
    input_dir = Path(paths.input_dir)
    output_dir = Path(paths.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    audit_path = reports_dir / "audit_report.json"
    top_doids_path = reports_dir / "top_doids.csv"

    # Skip if outputs exist and not forcing
    if not force and audit_path.exists() and top_doids_path.exists():
        logger.info("Audit outputs already exist. Skipping (use --force to re-run).")
        return

    file_counts = _count_files(input_dir)

    # Global counters
    drug_names = set()
    drugbank_ids = set()
    chebi_ids = set()

    ingredient_names = set()
    ingredient_codes = set()

    doid_counter = Counter()
    ordo_counter = Counter()

    # Iterate through all sources
    sources = ("xml_files", "txt_files", "csv_files")
    for src in sources:
        folder = input_dir / src
        if not folder.exists():
            logger.warning(f"Missing input folder: {folder}")
            continue

        for jf in _iter_json_files(folder):
            obj = _safe_load_json(jf)

            # Drugs
            for d in _extract_drug_records(obj):
                name = d.get("name")
                if isinstance(name, str) and name.strip():
                    drug_names.add(name.strip())

                # Optional IDs depending on source
                dbid = d.get("drugbank_id")
                if isinstance(dbid, str) and dbid.strip():
                    drugbank_ids.add(dbid.strip())

                chebi = d.get("chebi_id")
                if isinstance(chebi, str) and chebi.strip():
                    chebi_ids.add(chebi.strip())

            # DOID / ORDO
            for doid in _extract_doid_entities(obj):
                doid_counter[doid] += 1
            for ordo in _extract_ordo_entities(obj):
                ordo_counter[ordo] += 1

            # Ingredients
            for ing in _extract_ingredients(obj):
                n = ing.get("name")
                if isinstance(n, str) and n.strip():
                    ingredient_names.add(n.strip())
                c = ing.get("code")
                if isinstance(c, str) and c.strip():
                    ingredient_codes.add(c.strip())

    # Build report
    report = {
        "inputs": {
            "input_dir": str(input_dir),
            "file_counts": file_counts,
        },
        "entities": {
            "unique_drug_names": len(drug_names),
            "unique_drugbank_ids": len(drugbank_ids),
            "unique_chebi_ids_in_drug_field": len(chebi_ids),
            "unique_ingredient_names": len(ingredient_names),
            "unique_ingredient_codes": len(ingredient_codes),
            "unique_doids": len(doid_counter),
            "unique_ordos": len(ordo_counter),
        },
        "top_doids": doid_counter.most_common(20),
        "top_ordos": ordo_counter.most_common(20),
    }

    audit_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved audit report: {audit_path}")

    # Save top DOIDs CSV (for quick inspection)
    lines = ["doid,count"]
    for doid, c in doid_counter.most_common(200):
        lines.append(f"{doid},{c}")
    top_doids_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Saved top DOIDs CSV: {top_doids_path}")

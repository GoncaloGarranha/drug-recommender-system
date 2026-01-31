from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

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
    Step 04_build_triples: convert raw edges into canonical triples (src, rel, dst).

    Inputs:
      - data/output/interim/edges_raw.csv
      - data/output/mappings/drug_name_to_entity.json

    Output:
      - data/output/triples/triples.csv
    """
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    mappings_dir = output_dir / "mappings"
    triples_dir = output_dir / "triples"
    triples_dir.mkdir(parents=True, exist_ok=True)

    edges_path = interim_dir / "edges_raw.csv"
    drug_map_path = mappings_dir / "drug_name_to_entity.json"
    triples_path = triples_dir / "triples.csv"

    if not force and triples_path.exists():
        logger.info("Triples output already exists. Skipping (use --force to re-run).")
        return

    if not edges_path.exists():
        raise FileNotFoundError("Missing edges_raw.csv. Run 02_parse_sources first.")
    if not drug_map_path.exists():
        raise FileNotFoundError("Missing drug_name_to_entity.json. Run 03_resolve_entities first.")

    edges = _read_csv(edges_path)
    drug_name_to_entity = json.loads(drug_map_path.read_text(encoding="utf-8"))

    triples: List[List[str]] = []
    skipped_no_drug_mapping = 0
    skipped_bad_row = 0

    for e in edges:
        src_name_norm = (e.get("src") or "").strip()
        rel = (e.get("rel") or "").strip()
        dst = (e.get("dst") or "").strip()

        if not src_name_norm or not rel or not dst:
            skipped_bad_row += 1
            continue

        src_entity = drug_name_to_entity.get(src_name_norm)
        if not src_entity:
            skipped_no_drug_mapping += 1
            continue

        triples.append([src_entity, rel, dst])

    _write_csv(triples_path, ["src", "rel", "dst"], triples)
    logger.info(f"Saved: {triples_path}")
    logger.info(f"Triples: {len(triples)} | Skipped (no drug mapping): {skipped_no_drug_mapping} | Skipped (bad row): {skipped_bad_row}")

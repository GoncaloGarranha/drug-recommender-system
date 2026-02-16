"""
Step 04: Convert raw edges into canonical triples (src, rel, dst).
Usa o mapeamento de nomes para entidades canônicas gerado no passo 03 para garantir que todos os triplos usam apenas nós normalizados.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List

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


def build_triples(cfg: Dict[str, Any], paths, force: bool = False) -> None:
        """
        Step 04: Convert raw edges into canonical triples (src, rel, dst).
        Inputs:
            - data/output/interim/edges_raw.csv
            - data/output/mappings/drug_name_to_entity.json
        Output:
            - data/output/triples/triples.csv
        Usa o mapeamento de nomes normalizados para entidades canônicas (drug_name_to_entity.json)
        para garantir que todos os triplos usam apenas nós canônicos, evitando duplicidade.
        """
    output_dir = Path(paths.output_dir)
    interim_dir = output_dir / "interim"
    mappings_dir = output_dir / "mappings"
    triples_dir = output_dir / "triples"
    triples_dir.mkdir(parents=True, exist_ok=True)

    edges_csv = interim_dir / "edges_raw.csv"
    drug_map_json = mappings_dir / "drug_name_to_entity.json"
    triples_csv = triples_dir / "triples.csv"

    if not force and triples_csv.exists():
        logger.info("Triples output already exists. Skipping (use --force to re-run).")
        return

    if not edges_csv.exists():
        raise FileNotFoundError("Missing edges_raw.csv. Run 02_parse_sources first.")
    if not drug_map_json.exists():
        raise FileNotFoundError("Missing drug_name_to_entity.json. Run 03_resolve_entities first.")

    edges = read_csv(edges_csv)
    # Carrega o mapeamento de nomes normalizados para IDs canônicos de fármacos
    drug_name_to_entity = json.loads(drug_map_json.read_text(encoding="utf-8"))

    triples: List[List[str]] = []
    skipped_no_drug_mapping = 0
    skipped_bad_row = 0

    for edge in edges:
        src_name_norm = (edge.get("src") or "").strip()
        rel = (edge.get("rel") or "").strip()
        dst = (edge.get("dst") or "").strip()

        if not src_name_norm or not rel or not dst:
            skipped_bad_row += 1
            continue

        src_entity = drug_name_to_entity.get(src_name_norm)
        if not src_entity:
            skipped_no_drug_mapping += 1
            continue

        triples.append([src_entity, rel, dst])

    write_csv(triples_csv, ["src", "rel", "dst"], triples)
    logger.info(f"Saved: {triples_csv}")
    logger.info(f"Triples: {len(triples)} | Skipped (no drug mapping): {skipped_no_drug_mapping} | Skipped (bad row): {skipped_bad_row}")


# Alias for pipeline compatibility
run = build_triples


if __name__ == "__main__":
    class Paths:
        input_dir = "kg_constructor/data/input"
        output_dir = "kg_constructor/data/output"
    build_triples(config={}, paths=Paths(), force=True)

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rgcn.src.utils.log import get_logger

logger = get_logger(__name__)


def read_csv(path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file into a list of dictionaries.
    """
    with path.open("r", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """
    Write rows to a CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)


def infer_entity_type(entity_id: str) -> str:
    """
    Infer the entity type from its canonical prefix.
    """
    if entity_id.startswith("DRUGBANK:") or entity_id.startswith("DRUG:"):
        return "DRUG"
    if entity_id.startswith("DOID:"):
        return "DISEASE"
    if entity_id.startswith("UNII:") or entity_id.startswith("ING:"):
        return "INGREDIENT"
    return "UNKNOWN"


def validate_triples(rows: List[Dict[str, str]]) -> None:
    """
    Validate the input triples before graph preparation.
    """
    if not rows:
        raise ValueError("Input triple file is empty.")

    required_columns = {"src", "rel", "dst"}
    first_row_keys = set(rows[0].keys())

    missing_columns = required_columns - first_row_keys
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    indication_count = 0

    for row in rows:
        src = (row.get("src") or "").strip()
        rel = (row.get("rel") or "").strip()
        dst = (row.get("dst") or "").strip()

        if not src or not rel or not dst:
            raise ValueError("Found triple row with empty src, rel, or dst.")

        if rel == "INDICATION":
            indication_count += 1

    if indication_count == 0:
        raise ValueError("No INDICATION triples found. At least one is required.")


def build_vocabularies(
    rows: List[Dict[str, str]],
) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, int], Dict[int, str], Dict[str, str]]:
    """
    Build entity and relation vocabularies from canonical triples.
    """
    entity_ids = set()
    relation_ids = set()

    for row in rows:
        entity_ids.add((row.get("src") or "").strip())
        entity_ids.add((row.get("dst") or "").strip())
        relation_ids.add((row.get("rel") or "").strip())

    sorted_entities = sorted(entity_ids)
    sorted_relations = sorted(relation_ids)

    entity2id = {entity_id: idx for idx, entity_id in enumerate(sorted_entities)}
    id2entity = {idx: entity_id for entity_id, idx in entity2id.items()}

    rel2id = {relation: idx for idx, relation in enumerate(sorted_relations)}
    id2rel = {idx: relation for relation, idx in rel2id.items()}

    entity_types = {entity_id: infer_entity_type(entity_id) for entity_id in sorted_entities}

    return entity2id, id2entity, rel2id, id2rel, entity_types


def index_triples(
    rows: List[Dict[str, str]],
    entity2id: Dict[str, int],
    rel2id: Dict[str, int],
) -> List[List[Any]]:
    """
    Convert canonical triples into indexed triples.
    """
    indexed_rows: List[List[Any]] = []

    for row in rows:
        src = (row.get("src") or "").strip()
        rel = (row.get("rel") or "").strip()
        dst = (row.get("dst") or "").strip()

        indexed_rows.append([
            entity2id[src],
            rel2id[rel],
            entity2id[dst],
            src,
            rel,
            dst,
        ])

    return indexed_rows


def build_metadata(
    rows: List[Dict[str, str]],
    entity_types: Dict[str, str],
) -> Dict[str, Any]:
    """
    Build summary statistics for the prepared graph.
    """
    relation_counts = Counter()
    entity_type_counts = Counter(entity_types.values())

    for row in rows:
        relation = (row.get("rel") or "").strip()
        relation_counts[relation] += 1

    return {
        "num_entities": len(entity_types),
        "num_relations": len(relation_counts),
        "num_triples": len(rows),
        "relation_distribution": dict(relation_counts),
        "entity_type_distribution": dict(entity_type_counts),
    }


def prepare_graph(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 00: Prepare graph artifacts for the R-GCN pipeline.

    This step reads canonical triples from the KG constructor, validates them,
    builds entity/relation vocabularies, indexes the triples, and saves all
    required artifacts for downstream training and inference.
    """
    triples_path = Path(cfg["data"]["triples_path"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    entity2id_path = processed_dir / "entity2id.json"
    id2entity_path = processed_dir / "id2entity.json"
    rel2id_path = processed_dir / "rel2id.json"
    id2rel_path = processed_dir / "id2rel.json"
    entity_types_path = processed_dir / "entity_types.json"
    triples_indexed_path = processed_dir / "triples_indexed.csv"
    metadata_path = processed_dir / "graph_metadata.json"

    output_files = [
        entity2id_path,
        id2entity_path,
        rel2id_path,
        id2rel_path,
        entity_types_path,
        triples_indexed_path,
        metadata_path,
    ]

    if not force and all(path.exists() for path in output_files):
        logger.info("Processed graph artifacts already exist. Skipping step (use --force to re-run).")
        return

    if not triples_path.exists():
        raise FileNotFoundError(f"Triples file not found: {triples_path}")

    triple_rows = read_csv(triples_path)
    validate_triples(triple_rows)

    entity2id, id2entity, rel2id, id2rel, entity_types = build_vocabularies(triple_rows)
    indexed_rows = index_triples(triple_rows, entity2id, rel2id)
    metadata = build_metadata(triple_rows, entity_types)

    entity2id_path.write_text(json.dumps(entity2id, indent=2, ensure_ascii=False), encoding="utf-8")
    id2entity_path.write_text(json.dumps(id2entity, indent=2, ensure_ascii=False), encoding="utf-8")
    rel2id_path.write_text(json.dumps(rel2id, indent=2, ensure_ascii=False), encoding="utf-8")
    id2rel_path.write_text(json.dumps(id2rel, indent=2, ensure_ascii=False), encoding="utf-8")
    entity_types_path.write_text(json.dumps(entity_types, indent=2, ensure_ascii=False), encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    write_csv(
        triples_indexed_path,
        ["src_id", "rel_id", "dst_id", "src", "rel", "dst"],
        indexed_rows,
    )

    logger.info("Saved: %s", entity2id_path)
    logger.info("Saved: %s", id2entity_path)
    logger.info("Saved: %s", rel2id_path)
    logger.info("Saved: %s", id2rel_path)
    logger.info("Saved: %s", entity_types_path)
    logger.info("Saved: %s", triples_indexed_path)
    logger.info("Saved: %s", metadata_path)


run = prepare_graph

if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rgcn/configs/baseline.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file_obj:
        cfg = yaml.safe_load(file_obj)

    class Paths:
        def __init__(self):
            self.output_dir = cfg["data"]["processed_dir"]

    prepare_graph(cfg=cfg, paths=Paths(), force=args.force)
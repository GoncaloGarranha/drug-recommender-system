from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from models.shared.config import load_config
from models.shared.data_validation import (
    infer_entity_type,
    validate_no_empty_values,
    validate_required_relation,
    validate_triples_schema,
)
from models.shared.io import ensure_dir, read_csv, save_json, write_csv

from .utils.log import get_logger

logger = get_logger(__name__)


def validate_triples(rows: List[Dict[str, str]], target_relation: str | None = None) -> None:
    """
    Validate the input triples before graph preparation.

    This validation ensures that the canonical triples have the expected schema,
    contain no empty values, and optionally include the configured target
    relation.
    """
    validate_triples_schema(rows)
    validate_no_empty_values(rows)

    if target_relation:
        validate_required_relation(rows, target_relation)


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

        row_out: List[Any] = [
            entity2id[src],
            rel2id[rel],
            entity2id[dst],
            src,
            rel,
            dst,
        ]

        if "split" in row:
            row_out.append((row.get("split") or "").strip())

        indexed_rows.append(row_out)

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


def prepare_graph(cfg: Dict[str, Any], paths=None, force: bool = False) -> None:
    """
    Prepare graph artifacts for the R-GCN pipeline.

    This step reads the canonical triples file explicitly configured in the YAML,
    validates it, builds entity/relation vocabularies, indexes the triples, and
    saves all artifacts required for downstream training and inference.

    Important:
        The input triples file is never replaced implicitly. If a filtered
        dataset should be used, the YAML config must point explicitly to that
        filtered file.
    """
    triples_path = Path(cfg["data"]["triples_path"])
    processed_dir = ensure_dir(cfg["data"]["processed_dir"])

    logger.info("Using triples file configured in YAML: %s", triples_path)

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
        logger.info("Processed graph artifacts already exist. Skipping step. Use --force to re-run.")
        return

    if not triples_path.exists():
        raise FileNotFoundError(f"Triples file not found: {triples_path}")

    triple_rows = read_csv(triples_path)
    target_relation = cfg.get("task", {}).get("target_relation")
    validate_triples(triple_rows, target_relation=target_relation)

    entity2id, id2entity, rel2id, id2rel, entity_types = build_vocabularies(triple_rows)
    indexed_rows = index_triples(triple_rows, entity2id, rel2id)
    metadata = build_metadata(triple_rows, entity_types)

    header = ["src_id", "rel_id", "dst_id", "src", "rel", "dst"]

    if any("split" in row for row in triple_rows):
        header.append("split")

    save_json(entity2id_path, entity2id)
    save_json(id2entity_path, id2entity)
    save_json(rel2id_path, rel2id)
    save_json(id2rel_path, id2rel)
    save_json(entity_types_path, entity_types)
    save_json(metadata_path, metadata)

    write_csv(
        triples_indexed_path,
        header,
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


def main() -> None:
    """
    CLI entrypoint for graph preparation.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    prepare_graph(cfg=cfg, force=args.force)


if __name__ == "__main__":
    main()
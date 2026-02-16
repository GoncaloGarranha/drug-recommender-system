from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import torch

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def collect_entities_and_relations(edges: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build stable entity2id and rel2id mappings from a list of [src, rel, dst] edges.
    """
    entities = set()
    relations = set()

    for src, rel, dst in edges:
        entities.add(src)
        entities.add(dst)
        relations.add(rel)

    entity_to_id = {e: i for i, e in enumerate(sorted(entities))}
    relation_to_id = {r: i for i, r in enumerate(sorted(relations))}
    return entity_to_id, relation_to_id


def index_dgl(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 07: Create the DGL training graph and metadata.
    Inputs:
      - data/processed/edge_splits.json
    Outputs:
      - data/processed/train_graph.bin
      - data/processed/graph_metadata.json
      - data/processed/entity2id.json
      - data/processed/rel2id.json
    """
    processed_dir = Path(paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    splits_json = processed_dir / "edge_splits.json"
    graph_bin = processed_dir / "train_graph.bin"
    metadata_json = processed_dir / "graph_metadata.json"
    entity_to_id_json = processed_dir / "entity2id.json"
    relation_to_id_json = processed_dir / "rel2id.json"

    if not force and all(p.exists() for p in [graph_bin, metadata_json, entity_to_id_json, relation_to_id_json]):
        logger.info("DGL artifacts already exist. Skipping (use --force to re-run).")
        return

    if not splits_json.exists():
        raise FileNotFoundError("Missing edge_splits.json. Run 06_split_edges first.")

    splits = json.loads(splits_json.read_text(encoding="utf-8"))
    train_graph_edges: List[List[str]] = splits["train_graph_edges"]

    entity_to_id, relation_to_id = collect_entities_and_relations(train_graph_edges)

    src_ids = []
    dst_ids = []
    rel_ids = []

    num_original_rels = len(relation_to_id)

    for src, rel, dst in train_graph_edges:
        if src not in entity_to_id or dst not in entity_to_id or rel not in relation_to_id:
            continue
        src_idx = entity_to_id[src]
        rel_idx = relation_to_id[rel]
        dst_idx = entity_to_id[dst]
        
        src_ids.append(src_idx)
        dst_ids.append(dst_idx)
        rel_ids.append(rel_idx)
        
        src_ids.append(dst_idx)
        dst_ids.append(src_idx)
        rel_ids.append(rel_idx + num_original_rels)

    logger.info(f"Num entities: {len(entity_to_id)}")
    logger.info(f"Num relations: {len(relation_to_id)}")
    logger.info(f"Num edges (with inverses): {len(src_ids)}")

    src_tensor = torch.tensor(src_ids, dtype=torch.int64)
    dst_tensor = torch.tensor(dst_ids, dtype=torch.int64)
    rel_tensor = torch.tensor(rel_ids, dtype=torch.int64)

    graph = dgl.graph((src_tensor, dst_tensor), num_nodes=len(entity_to_id))
    graph.edata["etype"] = rel_tensor
    graph.edata["norm"] = torch.ones(graph.num_edges(), 1)

    dgl.save_graphs(str(graph_bin), [graph])
    logger.info(f"Saved training graph: {graph_bin}")

    entity_to_id_json.write_text(json.dumps(entity_to_id, indent=2, ensure_ascii=False), encoding="utf-8")
    relation_to_id_json.write_text(json.dumps(relation_to_id, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {entity_to_id_json}")
    logger.info(f"Saved: {relation_to_id_json}")

    metadata = {
        "n_entities": len(entity_to_id),
        "n_relations": len(relation_to_id),
        "relations": {r: relation_to_id[r] for r in sorted(relation_to_id.keys())},
        "train_graph_edges": len(train_graph_edges),
        "splits_counts": splits.get("counts", {}),
        "seed": splits.get("seed", None),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved metadata: {metadata_json}")

    logger.info(f"Nodes: {metadata['n_entities']} | Edges: {graph.num_edges()} | Relations: {metadata['n_relations']}")


run = index_dgl


if __name__ == "__main__":
    from kg_constructor.src.utils.config import load_config

    class Paths:
        def __init__(self, root: Path):
            self.root = root
            self.processed_dir = root / "kg_constructor" / "data" / "processed"

    root = Path(__file__).parent.parent.parent.parent
    config = load_config(root / "kg_constructor" / "configs" / "config.yaml")
    paths = Paths(root)
    
    index_dgl(config, paths, force=True)

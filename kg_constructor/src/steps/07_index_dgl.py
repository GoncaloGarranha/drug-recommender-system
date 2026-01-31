from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import torch

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def _collect_entities_and_relations(edges: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Build stable entity2id and rel2id mappings from a list of [src, rel, dst] edges.
    """
    entities = set()
    rels = set()

    for s, r, t in edges:
        entities.add(s)
        entities.add(t)
        rels.add(r)

    entity2id = {e: i for i, e in enumerate(sorted(entities))}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}
    return entity2id, rel2id


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 07_index_dgl: create the DGL training graph and metadata.

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

    splits_path = processed_dir / "edge_splits.json"
    graph_path = processed_dir / "train_graph.bin"
    meta_path = processed_dir / "graph_metadata.json"
    entity2id_path = processed_dir / "entity2id.json"
    rel2id_path = processed_dir / "rel2id.json"

    if not force and all(p.exists() for p in [graph_path, meta_path, entity2id_path, rel2id_path]):
        logger.info("DGL artifacts already exist. Skipping (use --force to re-run).")
        return

    if not splits_path.exists():
        raise FileNotFoundError("Missing edge_splits.json. Run 06_split_edges first.")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    train_graph_edges: List[List[str]] = splits["train_graph_edges"]

    # Build mappings from training graph edges only (important: prevents leakage)
    entity2id, rel2id = _collect_entities_and_relations(train_graph_edges)

    # Build graph edge tensors
    src_ids = []
    dst_ids = []
    rel_ids = []

    num_original_rels = len(rel2id)

    for s, r, t in train_graph_edges:
        if s not in entity2id or t not in entity2id or r not in rel2id:
            continue
        s_idx, r_idx, t_idx = entity2id[s], rel2id[r], entity2id[t]
        # Aresta Direta (Drug -> Disease)
        src_ids.append(s_idx)
        dst_ids.append(t_idx)
        rel_ids.append(r_idx)
        # ARESTA INVERSA (Disease -> Drug)
        src_ids.append(t_idx)
        dst_ids.append(s_idx)
        rel_ids.append(r_idx + num_original_rels)  # Novo ID de relação inversa

    print("[DEBUG KG_CONSTRUCTOR] Num entidades:", len(entity2id))
    print("[DEBUG KG_CONSTRUCTOR] Num relações:", len(rel2id))
    print("[DEBUG KG_CONSTRUCTOR] Num arestas:", len(src_ids))
    print("[DEBUG KG_CONSTRUCTOR] Exemplos src_ids:", src_ids[:10])
    print("[DEBUG KG_CONSTRUCTOR] Exemplos dst_ids:", dst_ids[:10])
    print("[DEBUG KG_CONSTRUCTOR] Exemplos rel_ids:", rel_ids[:10])

    src_tensor = torch.tensor(src_ids, dtype=torch.int64)
    dst_tensor = torch.tensor(dst_ids, dtype=torch.int64)
    rel_tensor = torch.tensor(rel_ids, dtype=torch.int64)

    g = dgl.graph((src_tensor, dst_tensor), num_nodes=len(entity2id))

    # Store relation type per edge (DGL convention for RGCN)
    g.edata["etype"] = rel_tensor
    # Add edge normalization (all ones, shape [num_edges, 1])
    g.edata["norm"] = torch.ones(g.num_edges(), 1)

    # Persist DGL graph
    dgl.save_graphs(str(graph_path), [g])
    logger.info(f"Saved training graph: {graph_path}")

    # Save mappings
    entity2id_path.write_text(json.dumps(entity2id, indent=2, ensure_ascii=False), encoding="utf-8")
    rel2id_path.write_text(json.dumps(rel2id, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {entity2id_path}")
    logger.info(f"Saved: {rel2id_path}")

    # Metadata for downstream training scripts
    metadata = {
        "n_entities": len(entity2id),
        "n_relations": len(rel2id),
        "relations": {r: rel2id[r] for r in sorted(rel2id.keys())},
        "train_graph_edges": len(train_graph_edges),
        "splits_counts": splits.get("counts", {}),
        "seed": splits.get("seed", None),
    }
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved metadata: {meta_path}")

    logger.info(f"Nodes: {metadata['n_entities']} | Edges: {g.num_edges()} | Relations: {metadata['n_relations']}")

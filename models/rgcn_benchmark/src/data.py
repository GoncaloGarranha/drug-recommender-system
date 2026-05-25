from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

from models.shared.graph_artifacts import load_prepared_graph
from models.shared.splits import get_or_create_splits

# ── Backend detection ─────────────────────────────────────────────────────────
try:
    import dgl
    _BACKEND = "dgl"
except ImportError:
    _BACKEND = "pyg"
# ─────────────────────────────────────────────────────────────────────────────

HEAD = "head"
TAIL = "tail"


def load_link_prediction_data(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Load indexed triples and official/random splits for generic link prediction."""
    prepared = load_prepared_graph(cfg["data"]["processed_dir"])
    split_cfg = cfg["split"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    train, valid, test, non_target = get_or_create_splits(
        output_dir=processed_dir / "splits",
        triples=prepared["triples"],
        target_relation=None,
        seed=int(cfg.get("seed", 42)),
        train_ratio=float(split_cfg.get("train_ratio", 0.8)),
        val_ratio=float(split_cfg.get("val_ratio", 0.1)),
        test_ratio=float(split_cfg.get("test_ratio", 0.1)),
        split_mode=split_cfg.get("mode"),
    )
    prepared.update({"train": train, "valid": valid, "test": test, "non_target": non_target})
    return prepared


def build_encoder_graph(
    train_triples: List[Dict[str, str]],
    num_nodes: int,
    num_relations: int,
    device: torch.device,
    add_inverse_edges: bool = True,
) -> Tuple[Any, torch.Tensor]:
    """Build the encoder graph.

    Returns a DGL graph when DGL is available, otherwise a PyG edge_index
    tensor of shape [2, E]. The edge_types tensor is always returned second.
    """
    src_ids: List[int] = []
    dst_ids: List[int] = []
    rel_ids: List[int] = []

    for row in train_triples:
        src = int(row["src_id"])
        rel = int(row["rel_id"])
        dst = int(row["dst_id"])
        src_ids.append(src)
        dst_ids.append(dst)
        rel_ids.append(rel)
        if add_inverse_edges:
            src_ids.append(dst)
            dst_ids.append(src)
            rel_ids.append(rel + num_relations)

    edge_types = torch.tensor(rel_ids, dtype=torch.long, device=device)

    if _BACKEND == "dgl":
        graph = dgl.graph((src_ids, dst_ids), num_nodes=num_nodes)
        return graph.to(device), edge_types
    else:  # pyg
        edge_index = torch.tensor([src_ids, dst_ids], dtype=torch.long, device=device)
        return edge_index, edge_types

def graph_num_edges(graph) -> int:
    """Return number of edges regardless of backend."""
    if _BACKEND == "dgl":
        return int(graph.num_edges())
    else:
        return int(graph.shape[1])


def graph_num_nodes(graph) -> int:
    """Return number of nodes regardless of backend."""
    if _BACKEND == "dgl":
        return int(graph.num_nodes())
    else:
        # PyG edge_index doesn't store num_nodes directly — use max + 1
        return int(graph.max().item()) + 1


def known_positives(
    triples: Iterable[Dict[str, str]],
) -> Dict[Tuple[Any, ...], set]:
    known: Dict[Tuple[Any, ...], set] = defaultdict(set)
    for row in triples:
        h = int(row["src_id"])
        r = int(row["rel_id"])
        t = int(row["dst_id"])
        known[(TAIL, h, r)].add(t)
        known[(HEAD, r, t)].add(h)
    return known


def triples_to_tensors(
    triples: List[Dict[str, str]],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    heads = torch.tensor([int(row["src_id"]) for row in triples], dtype=torch.long, device=device)
    rels  = torch.tensor([int(row["rel_id"]) for row in triples], dtype=torch.long, device=device)
    tails = torch.tensor([int(row["dst_id"]) for row in triples], dtype=torch.long, device=device)
    return heads, rels, tails
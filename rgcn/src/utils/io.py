import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import torch


def _read_triples_csv(triples_csv: Path) -> List[Tuple[str, str, str]]:
    with triples_csv.open("r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        triples: List[Tuple[str, str, str]] = []
        for row in reader:
            src = (row.get("src") or "").strip()
            rel = (row.get("rel") or "").strip()
            dst = (row.get("dst") or "").strip()
            if src and rel and dst:
                triples.append((src, rel, dst))
    return triples


def _build_splits_from_triples(
    triples: List[Tuple[str, str, str]],
    split_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    random_gen = random.Random(seed)

    train_ratio = float(split_cfg.get("train", 0.8))
    valid_ratio = float(split_cfg.get("valid", 0.1))
    test_ratio = float(split_cfg.get("test", 0.1))
    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0 (train + valid + test).")

    prediction_relations = set(split_cfg.get("prediction_relations", ["INDICATION", "CONTRAINDICATION"]))
    structural_relations = set(split_cfg.get("structural_relations", ["HAS_INGREDIENT"]))

    prediction_edges: List[Tuple[str, str, str]] = []
    structural_edges: List[Tuple[str, str, str]] = []
    relation_counts = Counter()

    for src, rel, dst in triples:
        relation_counts[rel] += 1
        triple = (src, rel, dst)
        if rel in prediction_relations:
            prediction_edges.append(triple)
        elif rel in structural_relations:
            structural_edges.append(triple)
        else:
            prediction_edges.append(triple)

    random_gen.shuffle(prediction_edges)

    total_prediction_edges = len(prediction_edges)
    num_train = int(total_prediction_edges * train_ratio)
    num_valid = int(total_prediction_edges * valid_ratio)

    train_edges = prediction_edges[:num_train]
    valid_edges = prediction_edges[num_train : num_train + num_valid]
    test_edges = prediction_edges[num_train + num_valid :]
    train_graph_edges = structural_edges + train_edges

    return {
        "seed": seed,
        "ratios": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "prediction_relations": sorted(list(prediction_relations)),
        "structural_relations": sorted(list(structural_relations)),
        "counts": {
            "all_triples": len(triples),
            "prediction_edges": len(prediction_edges),
            "structural_edges": len(structural_edges),
            "train_graph_edges": len(train_graph_edges),
            "train": len(train_edges),
            "valid": len(valid_edges),
            "test": len(test_edges),
        },
        "train_graph_edges": [list(edge) for edge in train_graph_edges],
        "valid_edges": [list(edge) for edge in valid_edges],
        "test_edges": [list(edge) for edge in test_edges],
        "relation_counts_in_filtered_triples": dict(relation_counts),
    }


def _collect_entities_relations(edges: List[List[str]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    entities = set()
    relations = set()
    for src, rel, dst in edges:
        entities.add(src)
        entities.add(dst)
        relations.add(rel)
    entity2id = {entity: index for index, entity in enumerate(sorted(entities))}
    rel2id = {relation: index for index, relation in enumerate(sorted(relations))}
    return entity2id, rel2id


def build_artifacts_from_triples(
    processed_dir: str,
    triples_csv_path: str,
    split_cfg: Dict[str, Any],
    seed: int,
) -> None:
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    triples_csv = Path(triples_csv_path)
    if not triples_csv.exists():
        raise FileNotFoundError(f"Missing triples CSV file: {triples_csv}")

    triples = _read_triples_csv(triples_csv)
    raw_splits = _build_splits_from_triples(triples, split_cfg=split_cfg, seed=seed)

    train_graph_edges: List[List[str]] = raw_splits.get("train_graph_edges", [])
    entity2id, rel2id = _collect_entities_relations(train_graph_edges)

    src_ids: List[int] = []
    dst_ids: List[int] = []
    rel_ids: List[int] = []

    num_relations = len(rel2id)
    for src, rel, dst in train_graph_edges:
        if src not in entity2id or dst not in entity2id or rel not in rel2id:
            continue
        src_id = entity2id[src]
        dst_id = entity2id[dst]
        rel_id = rel2id[rel]

        src_ids.append(src_id)
        dst_ids.append(dst_id)
        rel_ids.append(rel_id)

        src_ids.append(dst_id)
        dst_ids.append(src_id)
        rel_ids.append(rel_id + num_relations)

    graph = dgl.graph(
        (
            torch.tensor(src_ids, dtype=torch.int64),
            torch.tensor(dst_ids, dtype=torch.int64),
        ),
        num_nodes=len(entity2id),
    )
    graph.edata["etype"] = torch.tensor(rel_ids, dtype=torch.int64)
    graph.edata["norm"] = torch.ones(graph.num_edges(), 1)

    dgl.save_graphs(str(processed_path / "train_graph.bin"), [graph])
    (processed_path / "edge_splits.json").write_text(
        json.dumps(raw_splits, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (processed_path / "entity2id.json").write_text(
        json.dumps(entity2id, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (processed_path / "rel2id.json").write_text(
        json.dumps(rel2id, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    metadata = {
        "n_entities": len(entity2id),
        "n_relations": len(rel2id),
        "relations": {relation: rel2id[relation] for relation in sorted(rel2id.keys())},
        "train_graph_edges": len(train_graph_edges),
        "splits_counts": raw_splits.get("counts", {}),
        "seed": raw_splits.get("seed"),
    }
    (processed_path / "graph_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def ensure_processed_artifacts(
    data_cfg: Dict[str, Any],
    seed: int,
):
    processed_dir = data_cfg["processed_dir"]
    processed_path = Path(processed_dir)

    required = [
        processed_path / "train_graph.bin",
        processed_path / "graph_metadata.json",
        processed_path / "edge_splits.json",
        processed_path / "entity2id.json",
        processed_path / "rel2id.json",
    ]

    force_rebuild = bool(data_cfg.get("force_rebuild_artifacts", False))
    if force_rebuild or not all(path.exists() for path in required):
        triples_csv = data_cfg.get("triples_csv")
        if not triples_csv:
            missing = [str(path) for path in required if not path.exists()]
            raise FileNotFoundError(
                "Missing processed artifacts and no data.triples_csv configured. "
                f"Missing files: {missing}"
            )
        build_artifacts_from_triples(
            processed_dir=processed_dir,
            triples_csv_path=triples_csv,
            split_cfg=data_cfg.get("split", {}),
            seed=seed,
        )

    return load_processed_artifacts(processed_dir)


def load_processed_artifacts(processed_dir: str):
    processed_path = Path(processed_dir)

    graph_path = processed_path / "train_graph.bin"
    meta_path = processed_path / "graph_metadata.json"
    splits_path = processed_path / "edge_splits.json"
    entity2id_path = processed_path / "entity2id.json"
    rel2id_path = processed_path / "rel2id.json"

    for file_path in [graph_path, meta_path, splits_path, entity2id_path, rel2id_path]:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing {file_path}")

    graph_list, _ = dgl.load_graphs(str(graph_path))
    graph = graph_list[0]

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_splits = json.loads(splits_path.read_text(encoding="utf-8"))
    entity2id = json.loads(entity2id_path.read_text(encoding="utf-8"))
    rel2id = json.loads(rel2id_path.read_text(encoding="utf-8"))

    prediction_relations = set(raw_splits.get("prediction_relations", ["INDICATION", "CONTRAINDICATION"]))

    def to_split_ids(edge_list):
        src_ids, rel_ids, dst_ids = [], [], []
        for src, rel, dst in edge_list:
            if src not in entity2id or dst not in entity2id or rel not in rel2id:
                continue
            src_ids.append(entity2id[src])
            rel_ids.append(rel2id[rel])
            dst_ids.append(entity2id[dst])
        return {"src": src_ids, "rel": rel_ids, "dst": dst_ids}

    train_graph_edges = raw_splits.get("train_graph_edges", [])
    train_edges = [edge for edge in train_graph_edges if edge[1] in prediction_relations]

    splits = {
        "train_graph": to_split_ids(train_graph_edges),
        "train": to_split_ids(train_edges),
        "valid": to_split_ids(raw_splits.get("valid_edges", [])),
        "test": to_split_ids(raw_splits.get("test_edges", [])),
        "counts": raw_splits.get("counts", {}),
        "seed": raw_splits.get("seed", None),
    }

    metadata["entity2id"] = entity2id
    metadata["rel2id"] = rel2id
    metadata["id2entity"] = {value: key for key, value in entity2id.items()}
    metadata["id2rel"] = {value: key for key, value in rel2id.items()}

    edge_types = graph.edata["etype"]
    return graph, edge_types, metadata, splits

import json
from pathlib import Path
import dgl


def load_processed_artifacts(processed_dir: str):
    p = Path(processed_dir)

    graph_path = p / "train_graph.bin"
    meta_path = p / "graph_metadata.json"
    splits_path = p / "edge_splits.json"
    entity2id_path = p / "entity2id.json"
    rel2id_path = p / "rel2id.json"

    for fp in [graph_path, meta_path, splits_path, entity2id_path, rel2id_path]:
        if not fp.exists():
            raise FileNotFoundError(f"Missing {fp}")

    g_list, _ = dgl.load_graphs(str(graph_path))
    g = g_list[0]

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    raw_splits = json.loads(splits_path.read_text(encoding="utf-8"))
    entity2id = json.loads(entity2id_path.read_text(encoding="utf-8"))
    rel2id = json.loads(rel2id_path.read_text(encoding="utf-8"))

    prediction_rels = set(raw_splits.get("prediction_relations", ["INDICATION", "CONTRAINDICATION"]))

    def to_split_ids(edge_list):
        src, rel, dst = [], [], []
        for s, r, t in edge_list:
            if s not in entity2id or t not in entity2id or r not in rel2id:
                continue
            src.append(entity2id[s])
            rel.append(rel2id[r])
            dst.append(entity2id[t])
        return {"src": src, "rel": rel, "dst": dst}

    train_graph_edges = raw_splits.get("train_graph_edges", [])
    # Training positives should be prediction edges only (avoid HAS_INGREDIENT in the loss)
    train_edges = [e for e in train_graph_edges if e[1] in prediction_rels]

    splits = {
        "train_graph": to_split_ids(train_graph_edges),
        "train": to_split_ids(train_edges),
        "valid": to_split_ids(raw_splits.get("valid_edges", [])),
        "test": to_split_ids(raw_splits.get("test_edges", [])),
        "counts": raw_splits.get("counts", {}),
        "seed": raw_splits.get("seed", None),
    }

    # Provide both mappings for inference/labeling later
    meta["entity2id"] = entity2id
    meta["rel2id"] = rel2id
    meta["id2entity"] = {v: k for k, v in entity2id.items()}
    meta["id2rel"] = {v: k for k, v in rel2id.items()}

    # In the new graph, the edge type tensor is stored as 'etype' (RGCN convention)
    etype = g.edata["etype"]

    return g, etype, meta, splits

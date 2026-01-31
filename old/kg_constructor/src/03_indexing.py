import os
import json
import numpy as np
import pandas as pd
import torch
import dgl

TRIPLES_PATH = "kg_constructor/data/output/triples.csv"
PROCESSED_DIR = "kg_constructor/data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

PREDICATE_COL = "predicate"
SUBJECT_COL = "subject"
OBJECT_COL = "object"

SEED = 42
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

SELF_LOOP_REL_NAME = "self_loop"


def run_indexing_v2() -> None:
    """
    Indexing pipeline for R-GCN link prediction (single-graph setting).

    Outputs:
    - train_graph.bin: DGLGraph with training edges (forward + reverse + self-loops)
    - edge_splits.json: forward edges split into train/valid/test for evaluation
    - graph_metadata.json: entity/relation mappings and counts
    """
    print("Starting Advanced DGL Indexing...")

    if not os.path.exists(TRIPLES_PATH):
        raise FileNotFoundError(f"{TRIPLES_PATH} not found. Run extraction first.")

    df = pd.read_csv(TRIPLES_PATH).drop_duplicates()

    # Basic validation
    for col in [SUBJECT_COL, PREDICATE_COL, OBJECT_COL]:
        if col not in df.columns:
            raise ValueError(f"triples.csv missing required column: {col}")

    # 1) Entity indexing (single unified node space)
    all_entities = pd.concat([df[SUBJECT_COL].astype(str), df[OBJECT_COL].astype(str)]).unique()
    entity_to_idx = {ent: i for i, ent in enumerate(all_entities)}

    # 2) Relation indexing: forward + reverse + explicit self_loop relation
    forward_rels = df[PREDICATE_COL].astype(str).unique().tolist()

    rel_to_idx = {}
    forward_to_reverse = {}

    for rel in forward_rels:
        fwd_id = len(rel_to_idx)
        rel_to_idx[rel] = fwd_id

        rev_name = f"rev_{rel}"
        rev_id = len(rel_to_idx)
        rel_to_idx[rev_name] = rev_id

        forward_to_reverse[str(fwd_id)] = rev_id

    self_loop_rel_id = len(rel_to_idx)
    rel_to_idx[SELF_LOOP_REL_NAME] = self_loop_rel_id

    print(f"Entities: {len(entity_to_idx)} | Relations: {len(rel_to_idx)}")

    # 3) Convert triples to indexed arrays (forward edges only for splitting/evaluation)
    src = df[SUBJECT_COL].astype(str).map(entity_to_idx).to_numpy()
    dst = df[OBJECT_COL].astype(str).map(entity_to_idx).to_numpy()
    rel = df[PREDICATE_COL].astype(str).map(rel_to_idx).to_numpy()

    num_edges = len(src)
    if num_edges == 0:
        raise ValueError("No edges found after deduplication.")

    # 4) Reproducible split
    rng = np.random.default_rng(SEED)
    indices = np.arange(num_edges)
    rng.shuffle(indices)

    n_train = int(TRAIN_RATIO * num_edges)
    n_valid = int(VALID_RATIO * num_edges)
    n_test = num_edges - n_train - n_valid

    train_idx = indices[:n_train]
    valid_idx = indices[n_train:n_train + n_valid]
    test_idx = indices[n_train + n_valid:]

    # 5) Build training graph edges (forward + reverse)
    train_src = src[train_idx]
    train_dst = dst[train_idx]
    train_rel = rel[train_idx]

    # Reverse edges: (v, u) with relation rev_r
    train_rel_rev = np.array([forward_to_reverse[str(int(r))] for r in train_rel], dtype=np.int64)

    final_src = np.concatenate([train_src, train_dst])
    final_dst = np.concatenate([train_dst, train_src])
    final_rel = np.concatenate([train_rel, train_rel_rev])

    # 6) Add explicit self-loops with a dedicated relation type
    n_nodes = len(entity_to_idx)
    loop_nodes = np.arange(n_nodes, dtype=np.int64)
    loop_src = loop_nodes
    loop_dst = loop_nodes
    loop_rel = np.full(shape=n_nodes, fill_value=self_loop_rel_id, dtype=np.int64)

    final_src = np.concatenate([final_src, loop_src])
    final_dst = np.concatenate([final_dst, loop_dst])
    final_rel = np.concatenate([final_rel, loop_rel])

    g_train = dgl.graph(
        (torch.tensor(final_src, dtype=torch.long), torch.tensor(final_dst, dtype=torch.long)),
        num_nodes=n_nodes
    )
    g_train.edata["etype"] = torch.tensor(final_rel, dtype=torch.long)

    # 7) Save artifacts
    train_graph_path = os.path.join(PROCESSED_DIR, "train_graph.bin")
    dgl.save_graphs(train_graph_path, [g_train])

    splits = {
        "train": {"src": src[train_idx].tolist(), "dst": dst[train_idx].tolist(), "rel": rel[train_idx].tolist()},
        "valid": {"src": src[valid_idx].tolist(), "dst": dst[valid_idx].tolist(), "rel": rel[valid_idx].tolist()},
        "test": {"src": src[test_idx].tolist(), "dst": dst[test_idx].tolist(), "rel": rel[test_idx].tolist()},
    }
    splits_path = os.path.join(PROCESSED_DIR, "edge_splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f)

    counts = {
        "triples_deduped": int(num_edges),
        "train_edges_forward": int(len(train_idx)),
        "valid_edges_forward": int(len(valid_idx)),
        "test_edges_forward": int(len(test_idx)),
        "train_graph_edges_total": int(g_train.num_edges()),
    }

    metadata = {
        "n_entities": int(n_nodes),
        "n_relations": int(len(rel_to_idx)),
        "entity_to_idx": entity_to_idx,
        "rel_to_idx": rel_to_idx,
        "forward_to_reverse": forward_to_reverse,
        "self_loop_rel_id": int(self_loop_rel_id),
        "counts": counts,
        "seed": SEED,
        "split": {"train": TRAIN_RATIO, "valid": VALID_RATIO, "test": TEST_RATIO},
    }

    metadata_path = os.path.join(PROCESSED_DIR, "graph_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("Advanced indexing completed.")
    print(f"Saved training graph: {train_graph_path}")
    print(f"Saved splits: {splits_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    run_indexing_v2()

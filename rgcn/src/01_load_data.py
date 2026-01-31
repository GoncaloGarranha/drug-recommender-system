import argparse
from rgcn.src.utils.io import load_processed_artifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="kg_constructor/data/processed")
    args = ap.parse_args()

    g, etype, meta, splits = load_processed_artifacts(args.processed_dir)

    print("Artifacts loaded successfully.")
    print(f"Nodes: {g.num_nodes()}")
    print(f"Edges: {g.num_edges()}")
    print(f"Relations: {meta.get('rel2id', {}).keys()}")
    print(f"n_entities (meta): {meta.get('n_entities')}")
    print(f"n_relations (meta): {meta.get('n_relations')}")
    print(f"Split sizes: train={len(splits['train']['src'])} valid={len(splits['valid']['src'])} test={len(splits['test']['src'])}")

    if "counts" in splits:
        print("Counts (from edge_splits.json):", splits["counts"])

if __name__ == "__main__":
    main()

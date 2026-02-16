"""
Data Loader Validation Script

This script validates that all required artifacts from the KG constructor pipeline
are correctly formatted and loadable for R-GCN training.

Usage:
    python -m rgcn.src.01_load_data --processed_dir kg_constructor/data/processed

Output:
    - Graph statistics (nodes, edges, relations)
    - Entity and relation counts
    - Train/valid/test split sizes
"""

import argparse
from rgcn.src.utils.io import load_processed_artifacts

def main():
    """
    Main function to load and validate processed graph artifacts.
    
    Loads the DGL graph, edge types, metadata, and edge splits from the
    KG constructor output directory and prints validation statistics.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="kg_constructor/data/processed")
    args = ap.parse_args()

    graph, edge_types, metadata, splits = load_processed_artifacts(args.processed_dir)

    print("Artifacts loaded successfully.")
    print(f"Nodes: {graph.num_nodes()}")
    print(f"Edges: {graph.num_edges()}")
    print(f"Relations: {metadata.get('rel2id', {}).keys()}")
    print(f"n_entities (meta): {metadata.get('n_entities')}")
    print(f"n_relations (meta): {metadata.get('n_relations')}")
    print(
        "Split sizes: "
        f"train={len(splits['train']['src'])} "
        f"valid={len(splits['valid']['src'])} "
        f"test={len(splits['test']['src'])}"
    )

    if "counts" in splits:
        print("Counts (from edge_splits.json):", splits["counts"])

if __name__ == "__main__":
    main()

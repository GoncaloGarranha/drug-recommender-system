"""
R-GCN Model Evaluation Script

Evaluates a trained R-GCN model on held-out test data using filtered ranking metrics.
Computes Mean Reciprocal Rank (MRR) and Hits@K for drug-disease link prediction.

Usage:
    python -m rgcn.src.03_evaluate --runs_dir rgcn/runs
    python -m rgcn.src.03_evaluate --run_dir rgcn/runs/2026-01-28_10-03-01_drug_to_disease

Output:
    - Evaluation metrics printed to console
    - Results saved to final_indication_metrics_{split}.json
"""

import argparse
import json
import os
import glob
import numpy as np
import torch
from typing import Dict, List, Set, Tuple

from rgcn.src.utils.io import load_processed_artifacts
from rgcn.src.models.rgcn_encoder import RGCNEncoder
from rgcn.src.models.decoders import DistMultDecoder
from rgcn.src.utils.seed import set_seed

def find_latest_run(runs_dir: str) -> str:
    """
    Find the most recently modified run directory.
    
    Args:
        runs_dir: Path to directory containing training runs
        
    Returns:
        Path to the most recent run directory
    """
    run_dirs = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    run_dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return run_dirs[0]

def build_known_forward_triples(splits: dict, relation_id: int) -> Set[Tuple[int, int, int]]:
    """
    Build set of all known (Drug, INDICATION, Disease) triples across all splits.
    
    Used for filtered ranking to avoid penalizing correct predictions that
    happen to be in the training, validation, or test sets.
    
    Args:
        splits: Dictionary with 'train', 'valid', 'test' splits
        relation_id: Relation ID for INDICATION
        
    Returns:
        Set of tuples (drug_id, rel_id, disease_id)
    """
    known_fwd = set()
    for split_name in ["train", "valid", "test"]:
        s = splits[split_name]
        for h_drug, _, t_dis in zip(s["src"], s["rel"], s["dst"]):
            known_fwd.add((int(h_drug), int(relation_id), int(t_dis)))
    return known_fwd

def infer_typed_partitions(splits: dict) -> Tuple[List[int], List[int]]:
    """
    Extract lists of all drug and disease node IDs from splits.
    
    Args:
        splits: Dictionary with 'train', 'valid', 'test' splits
        
    Returns:
        Tuple of (sorted_drug_ids, sorted_disease_ids)
    """
    drug_ids = set()
    disease_ids = set()
    for split_name in ["train", "valid", "test"]:
        s = splits[split_name]
        drug_ids.update(int(x) for x in s["src"])
        disease_ids.update(int(x) for x in s["dst"])
    return sorted(list(drug_ids)), sorted(list(disease_ids))

@torch.no_grad()
def evaluate_indications(node_emb, decoder, edges, candidate_diseases, known_fwd, hits_k, device):
    """
    Evaluate INDICATION prediction: rank the correct disease among all candidate diseases.
    
    For each (drug, disease) pair, computes scores for all candidate diseases and
    ranks the true disease using filtered ranking (removes known triples).
    
    Args:
        node_emb: Node embeddings from R-GCN encoder [num_nodes, emb_dim]
        decoder: Trained DistMult decoder
        edges: Dict with 'src' (drugs), 'dst' (diseases), 'rel' (INDICATION)
        candidate_diseases: List of all disease node IDs
        known_fwd: Set of known (drug, INDICATION, disease) triples to filter
        hits_k: List of K values for Hits@K metrics
        device: torch device (cpu/cuda)
        
    Returns:
        Dict with 'mrr' and 'hits@K' metrics
    """
    drug_ids = [int(x) for x in edges["src"]]
    true_disease_ids = [int(x) for x in edges["dst"]]
    rel_fwd_id = int(edges["rel"][0])

    candidate_tensor = torch.tensor(candidate_diseases, dtype=torch.long, device=device)
    candidate_embeddings = node_emb[candidate_tensor]

    ranks = []
    hits = {k: 0 for k in hits_k}

    for drug_id, true_disease_id in zip(drug_ids, true_disease_ids):
        head_emb = node_emb[drug_id]
        rel_emb = decoder.rel_emb(torch.tensor([rel_fwd_id], device=device)).squeeze(0)

        # Scoring DistMult: <h * r, t>
        query_vec = (head_emb * rel_emb).unsqueeze(0)
        scores = torch.matmul(query_vec, candidate_embeddings.T).squeeze(0)

        # Filtered Ranking: ignore other known diseases for this drug
        for j, candidate_id in enumerate(candidate_diseases):
            if candidate_id == true_disease_id:
                continue
            if (drug_id, rel_fwd_id, candidate_id) in known_fwd:
                scores[j] = -1e9

        try:
            true_pos = candidate_diseases.index(true_disease_id)
            order = torch.argsort(scores, descending=True)
            rank = int((order == true_pos).nonzero(as_tuple=False)[0].item()) + 1
            ranks.append(rank)
            for k in hits_k:
                if rank <= k: hits[k] += 1
        except ValueError: continue

    ranks = np.array(ranks)
    return {"mrr": float(np.mean(1.0/ranks)), **{f"hits@{k}": float(hits[k]/len(ranks)) for k in hits_k}}

def main():
    """
    Main evaluation function.
    
    Workflow:
        1. Load trained model checkpoint
        2. Load processed graph artifacts
        3. Build known triples for filtered ranking
        4. Evaluate on specified split (default: test)
        5. Save results to JSON
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="kg_constructor/data/processed")
    ap.add_argument("--runs_dir", default="rgcn/runs")
    ap.add_argument("--split", default="test", choices=["valid", "test"])
    args = ap.parse_args()

    # 1) Load data and model
    graph, edge_types, metadata, splits = load_processed_artifacts(args.processed_dir)
    run_dir = find_latest_run(args.runs_dir)
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu")
    cfg = ckpt["config"]
    device = torch.device("cpu")
    set_seed(cfg["seed"])
    encoder = RGCNEncoder(
        metadata["n_entities"],
        cfg["training"]["emb_dim"],
        metadata["n_relations"],
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
    ).to(device)
    decoder = DistMultDecoder(metadata["n_relations"], cfg["training"]["emb_dim"]).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])

    # 2) Configure evaluation
    node_emb = encoder(graph.to(device), edge_types.to(device))
    drug_ids, disease_ids = infer_typed_partitions(splits)
    rel_fwd = metadata["rel2id"]["INDICATION"]
    known_fwd = build_known_forward_triples(splits, rel_fwd)

    # 3) Prepare task: (Drug, indicated_for, ?)
    eval_edges = {
        "src": splits[args.split]["src"],
        "rel": [rel_fwd] * len(splits[args.split]["rel"]),
        "dst": splits[args.split]["dst"],
    }

    # 4) Evaluate
    print(f"\nEvaluating Drug Repurposing (Drug -> Disease) on '{args.split}' split...")
    results = evaluate_indications(
        node_emb,
        decoder,
        eval_edges,
        disease_ids,
        known_fwd,
        cfg["training"]["hits_k"],
        device,
    )

    # 5) Output
    print("-" * 60)
    print(f"RESULTS FOR TASK: DRUG -> DISEASE")
    print("-" * 60)
    print(f"Mean Reciprocal Rank (MRR): {results['mrr']:.4f}")
    for k in cfg["training"]["hits_k"]:
        print(f"Hits@{k}: {results[f'hits@{k}']:.4f}")
    print("-" * 60)

    with open(os.path.join(run_dir, f"final_indication_metrics_{args.split}.json"), "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
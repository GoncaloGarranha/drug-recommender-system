"""
R-GCN Inference Script

Generates top-K disease predictions for a given drug using a trained R-GCN model.
Supports both fuzzy name matching and exact entity ID lookup.

Usage:
    # Search by drug name (fuzzy match)
    python -m rgcn.src.04_infer --query "Albuterol" --topk 10
    
    # Search by exact entity ID
    python -m rgcn.src.04_infer --entity_id "DRUG:ALBUTEROL SULFATE" --topk 10
    
    # Save results to JSON
    python -m rgcn.src.04_infer --query "Metformin" --topk 10 --json_out predictions.json
    
    # Show internal entity IDs
    python -m rgcn.src.04_infer --query "Aspirin" --topk 5 --show_ids

Output:
    - Top-K disease predictions with scores and disease names
    - Optional JSON output file with detailed results
"""

import argparse
import os
import csv
import glob
import json
import torch
from rgcn.src.utils.io import load_processed_artifacts
from rgcn.src.models.rgcn_encoder import RGCNEncoder
from rgcn.src.models.decoders import DistMultDecoder


def find_latest_run(runs_dir: str):
    """
    Find the most recently trained model.
    
    Args:
        runs_dir: Path to directory containing training runs
        
    Returns:
        Path to the most recent run directory
        
    Raises:
        FileNotFoundError: If no training runs found
    """
    candidates = [p for p in glob.glob(os.path.join(runs_dir, "*")) if os.path.isdir(p)]
    if not candidates:
        raise FileNotFoundError("No training runs found in rgcn/runs")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def load_entity_labels(csv_path: str):
    """
    Load entity ID to human-readable label mapping from CSV.
    
    Args:
        csv_path: Path to CSV file with entity labels
        
    Returns:
        Dict mapping entity IDs to labels
    """
    mapping = {}
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    mapping[row[0]] = row[1]
    return mapping


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Apply sigmoid activation to tensor.
    
    Args:
        x: Input tensor
        
    Returns:
        Sigmoid-activated tensor
    """
    return torch.sigmoid(x)


def main():
    """
    Main inference function.
    
    Workflow:
        1. Load trained model checkpoint
        2. Load processed graph artifacts
        3. Find query drug by name or ID
        4. Generate node embeddings
        5. Compute scores for all candidate diseases
        6. Rank and return top-K predictions
        7. Optionally save to JSON
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=False, help="Drug name (e.g., Albuterol)")
    ap.add_argument("--entity_id", required=False, help="Exact drug entity ID (e.g., DRUG:ALBUTEROL SULFATE)")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--show_ids", action="store_true", help="Show entity IDs (recommended for debugging)")
    ap.add_argument("--json_out", default=None, help="Save results to JSON file (e.g., out.json)")
    args = ap.parse_args()

    # Load latest trained model
    run_dir = find_latest_run("rgcn/runs")
    ckpt = torch.load(os.path.join(run_dir, "checkpoints", "best.pt"), map_location="cpu")
    cfg = ckpt["config"]
    print(f"Model loaded from: {run_dir}")

    # 2) Load graph and labels
    graph, edge_types, metadata, splits = load_processed_artifacts(cfg["data"]["processed_dir"])
    entity_labels = load_entity_labels("kg_constructor/data/output/entity_names.csv")

    entity_to_id = metadata["entity2id"]
    id_to_entity = metadata["id2entity"]

    if "INDICATION" not in metadata["rel2id"]:
        raise KeyError("Relation 'INDICATION' does not exist in rel2id.")
    indication_rel_id = metadata["rel2id"]["INDICATION"]

    # 3) Candidate diseases (dst from train/valid/test for consistent ranking)
    disease_node_ids = list(
        set(splits["train"]["dst"] + splits["valid"]["dst"] + splits["test"]["dst"])
    )
    disease_tensor = torch.tensor(disease_node_ids)

    # 4) Resolve drug (exact ID or query)
    drug_node_id = None
    drug_entity_id = None

    if args.entity_id:
        if args.entity_id not in entity_to_id:
            print(f"Error: entity_id '{args.entity_id}' does not exist in the graph.")
            return
        drug_entity_id = args.entity_id
        drug_node_id = entity_to_id[drug_entity_id]
        print(f"Drug selected by ID: {entity_labels.get(drug_entity_id, drug_entity_id)} (ID: {drug_entity_id})")

    else:
        if not args.query:
            print("Error: you must provide --query or --entity_id.")
            return

        query = args.query.lower()

        # Only consider entities that appear as src in splits (real drugs)
        drug_node_ids = set(splits["train"]["src"] + splits["valid"]["src"] + splits["test"]["src"])
        drug_entity_ids = [id_to_entity[idx] for idx in drug_node_ids if idx in id_to_entity]

        def label_for(entity_id: str) -> str:
            return entity_labels.get(entity_id, entity_id)

        exact = [
            eid
            for eid in drug_entity_ids
            if label_for(eid).lower() == query or eid.lower() == query
        ]
        starts = [
            eid
            for eid in drug_entity_ids
            if label_for(eid).lower().startswith(query) or eid.lower().startswith(query)
        ]
        subs = [
            eid
            for eid in drug_entity_ids
            if query in label_for(eid).lower() or query in eid.lower()
        ]

        candidates = exact or starts or subs
        if not candidates:
            print(f"Error: drug '{args.query}' was not found in the graph.")
            return

        # If there are multiple matches, show suggestions (user friendly)
        # Sort by best: shortest label match
        candidates = sorted(set(candidates), key=lambda eid: len(label_for(eid)))

        if len(candidates) > 1:
            print("\nMultiple possible drugs found. Suggestions:")
            for i, eid in enumerate(candidates[:5], start=1):
                print(f"  {i}) {label_for(eid)} (ID: {eid})")
            print("Tip: use --entity_id to choose the exact one.\n")

        drug_entity_id = candidates[0]
        drug_node_id = entity_to_id[drug_entity_id]
        print(f"Detected drug: {entity_labels.get(drug_entity_id, drug_entity_id)} (ID: {drug_entity_id})")

    # 5) “Known vs Novel”: indications already present in the training graph
    known_indications = set()
    tg = splits["train_graph"]  # training graph edges (also includes non-prediction edges)
    for s, r, t in zip(tg["src"], tg["rel"], tg["dst"]):
        if s == drug_node_id and r == indication_rel_id:
            known_indications.add(t)  # t is node id (int)

    # 6) Inference
    encoder = RGCNEncoder(
        metadata["n_entities"],
        cfg["training"]["emb_dim"],
        metadata["n_relations"],
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
    )
    decoder = DistMultDecoder(metadata["n_relations"], cfg["training"]["emb_dim"])
    encoder.load_state_dict(ckpt["encoder"])
    decoder.load_state_dict(ckpt["decoder"])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        emb = encoder(graph, graph.edata["etype"])

        h = emb[drug_node_id].unsqueeze(0).repeat(len(disease_node_ids), 1)
        r = decoder.rel_emb(torch.tensor([indication_rel_id]).repeat(len(disease_node_ids)))
        t = emb[disease_tensor]

        scores = torch.sum(h * r * t, dim=1)  # DistMult raw score
        probs = sigmoid(scores)               # interpretable (not calibrated, but helpful)

        topk = min(args.topk, len(scores))
        topk_scores, topk_idx = torch.topk(scores, k=topk)
        topk_probs = probs[topk_idx]

    # 7) Show results (more user friendly)
    print(f"\nTop-{topk} Likely Indications (Drug → Disease):")
    print("-" * 92)
    print(f"{'RK':>2}  {'Disease':40}  {'DOID':14}  {'Score':>8}  {'Sigmoid':>8}  {'Status':>7}")
    print("-" * 92)

    results = []
    for rank in range(topk):
        dis_node = disease_node_ids[topk_idx[rank].item()]
        dis_eid = id_to_entity[dis_node]
        dis_name = entity_labels.get(dis_eid, dis_eid)

        status = "KNOWN" if dis_node in known_indications else "CANDIDATE"

        doid_str = dis_eid if dis_eid.startswith("DOID:") else ("-" if not args.show_ids else dis_eid)

        print(
            f"{rank+1:02d}  {dis_name[:40]:40}  {doid_str:14}  {topk_scores[rank].item():8.4f}  {topk_probs[rank].item():8.4f}  {status:>7}"
        )

        results.append({
            "rank": rank + 1,
            "drug": {"id": drug_entity_id, "name": entity_labels.get(drug_entity_id, drug_entity_id)},
            "disease": {"id": dis_eid, "name": dis_name},
            "score": float(topk_scores[rank].item()),
            "sigmoid": float(topk_probs[rank].item()),
            "status": status,
        })

    # 8) Optional JSON
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] Results saved to: {args.json_out}")


if __name__ == "__main__":
    main()

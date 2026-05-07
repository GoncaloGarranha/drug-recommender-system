"""
R-GCN Inference Script

Generate top-K recommendations for a configurable task using a trained R-GCN model.

Supported task modes:
    - disease_to_drug
    - drug_to_disease
    - tail_prediction

Usage examples:
    python -m rgcn.src.04_infer --query "diabetes" --topk 10
    python -m rgcn.src.04_infer --entity_id "DOID:9352" --topk 10
    python -m rgcn.src.04_infer --run_dir rgcn/runs/<run_name> --query "asthma"
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dgl
import numpy as np
import torch

from models.shared.graph_artifacts import load_prepared_graph, resolve_processed_dir
from models.shared.io import read_csv, read_json
from models.shared.runs import find_latest_run
from models.shared.ranking import build_candidate_pool, build_training_graph, score_candidates
from models.shared.splits import build_known_positives, get_or_create_splits

from .models.decoders import DistMultDecoder
from .models.rgcn_encoder import RGCNEncoder


def load_entity_labels(csv_path: str) -> Dict[str, str]:
	"""
	Load optional human-readable labels from a CSV file.

	Expected CSV format:
		entity_id,label
	"""
	mapping: Dict[str, str] = {}

	if not os.path.exists(csv_path):
		return mapping

	with open(csv_path, "r", encoding="utf-8") as file_obj:
		reader = csv.reader(file_obj)
		next(reader, None)

		for row in reader:
			if len(row) >= 2:
				mapping[row[0]] = row[1]

	return mapping


def resolve_query_entity(
    query: str | None,
    entity_id: str | None,
    expected_entity_type: str,
    entity2id: Dict[str, int],
    entity_types: Dict[str, str],
    entity_labels: Dict[str, str],
) -> Tuple[str, int]:
    """
    Resolve the user query to a canonical entity ID and node ID.

    The resolution supports:
        - exact canonical ID lookup
        - exact label match
        - prefix match
        - substring match
    """
    if entity_id:
        if entity_id not in entity2id:
            raise ValueError(f"Entity ID not found in the graph: {entity_id}")

        actual_type = entity_types.get(entity_id, "UNKNOWN")
        if expected_entity_type != "ALL" and actual_type != expected_entity_type:
            raise ValueError(
                f"Entity '{entity_id}' has type '{actual_type}', expected '{expected_entity_type}'."
            )

        return entity_id, int(entity2id[entity_id])

    if not query:
        raise ValueError("You must provide either --query or --entity_id.")

    query_lower = query.lower()

    if expected_entity_type == "ALL":
        valid_entity_ids = list(entity2id.keys())
    else:
        valid_entity_ids = [
            eid for eid, etype in entity_types.items()
            if etype == expected_entity_type
        ]

    def label_for(eid: str) -> str:
        return entity_labels.get(eid, eid)

    exact_matches = [
        eid for eid in valid_entity_ids
        if label_for(eid).lower() == query_lower or eid.lower() == query_lower
    ]
    prefix_matches = [
        eid for eid in valid_entity_ids
        if label_for(eid).lower().startswith(query_lower) or eid.lower().startswith(query_lower)
    ]
    substring_matches = [
        eid for eid in valid_entity_ids
        if query_lower in label_for(eid).lower() or query_lower in eid.lower()
    ]

    candidates = exact_matches or prefix_matches or substring_matches
    if not candidates:
        raise ValueError(f"No matching entity found for query: {query}")

    candidates = sorted(set(candidates), key=lambda eid: len(label_for(eid)))
    selected_entity_id = candidates[0]

    if len(candidates) > 1:
        print("\nMultiple possible matches found. Top suggestions:")
        for idx, eid in enumerate(candidates[:5], start=1):
            print(f"  {idx}) {label_for(eid)} (ID: {eid})")
        print("Tip: use --entity_id to select the exact entity.\n")

    return selected_entity_id, int(entity2id[selected_entity_id])


def main() -> None:
    """
    Load a trained checkpoint and generate top-K ranked predictions for the
    configured task.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs_dir", default="rgcn/runs")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--query", default=None, help="Entity name or fuzzy query string.")
    parser.add_argument("--entity_id", default=None, help="Exact canonical entity ID.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--show_ids", action="store_true", help="Display canonical entity IDs.")
    parser.add_argument("--json_out", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--entity_labels_csv",
        default="kg_constructor/data/output/entity_names.csv",
        help="Optional CSV file mapping entity IDs to human-readable labels.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir if args.run_dir else find_latest_run(args.runs_dir)
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    device = torch.device(cfg["training"].get("device", "cpu"))

    task_cfg = cfg["task"]
    task_name = task_cfg["name"]
    processed_dir = resolve_processed_dir(cfg["data"]["processed_dir"], task_name=task_name)

    prepared = load_prepared_graph(processed_dir)
    triples = prepared["triples"]
    entity2id = prepared["entity2id"]
    id2entity = prepared["id2entity"]
    rel2id = prepared["rel2id"]
    entity_types = prepared["entity_types"]

    entity_labels = load_entity_labels(args.entity_labels_csv)

    num_nodes = len(entity2id)
    num_relations = len(rel2id)

    split_cfg = cfg["split"]

    target_relation = task_cfg["target_relation"]
    query_entity_type = task_cfg["query_entity_type"]
    candidate_entity_type = task_cfg["candidate_entity_type"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    if task_name == "tail_prediction":
        relation_id = 0
    else:
        relation_id = rel2id.get(target_relation)
        if relation_id is None:
            raise ValueError(f"Target relation not found: {target_relation}")

    seed = int(cfg.get("seed", 42))

    splits_dir = processed_dir / "splits"
    train_target, valid_target, test_target, non_target_triples = get_or_create_splits(
        output_dir=splits_dir,
        triples=triples,
        target_relation=target_relation,
        seed=seed,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        split_mode=split_cfg.get("mode") if isinstance(split_cfg, dict) else None,
    )

    graph, edge_types = build_training_graph(
        train_target=train_target,
        non_target_triples=non_target_triples,
        context_relations=context_relations,
        num_nodes=num_nodes,
        device=device,
    )

    required_candidate_ids = {
        int(row["src_id"]) if task_name == "disease_to_drug" else int(row["dst_id"])
        for row in train_target + valid_target + test_target
    }

    candidate_pool_tensor = build_candidate_pool(
        entity_types=entity_types,
        entity2id=entity2id,
        candidate_entity_type=candidate_entity_type,
        device=device,
        required_entity_ids=required_candidate_ids,
    )

    known_train_positives = build_known_positives(
        triples=train_target,
        task_name=task_name if task_name != "tail_prediction" else None,
        query_key="src_id" if task_name == "tail_prediction" else None,
        candidate_key="dst_id" if task_name == "tail_prediction" else None,
    )

    resolved_entity_id, query_node_id = resolve_query_entity(
        query=args.query,
        entity_id=args.entity_id,
        expected_entity_type=query_entity_type,
        entity2id=entity2id,
        entity_types=entity_types,
        entity_labels=entity_labels,
    )

    encoder = RGCNEncoder(
        num_nodes,
        cfg["training"]["emb_dim"],
        num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
    ).to(device)

    decoder = DistMultDecoder(
        num_relations,
        cfg["training"]["emb_dim"],
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        node_embeddings = encoder(graph, edge_types)
        scores = score_candidates(
            task_name=task_name,
            decoder=decoder,
            node_embeddings=node_embeddings,
            relation_id=relation_id,
            query_id=query_node_id,
            candidate_ids=candidate_pool_tensor,
            device=device,
        )
        probabilities = torch.sigmoid(scores)

        topk = min(args.topk, len(candidate_pool_tensor))
        top_scores, top_indices = torch.topk(scores, k=topk)
        top_probabilities = probabilities[top_indices]

    query_label = entity_labels.get(resolved_entity_id, resolved_entity_id)

    if task_name == "disease_to_drug":
        print(f"\nTop-{topk} recommended drugs for disease: {query_label}")
    elif task_name == "drug_to_disease":
        print(f"\nTop-{topk} predicted diseases for drug: {query_label}")
    else:
        print(f"\nTop-{topk} predictions for query: {query_label}")

    print("-" * 100)

    if candidate_entity_type == "ALL":
        result_label = "Entity"
    else:
        result_label = "Drug" if candidate_entity_type == "DRUG" else "Disease"
    print(
        f"{'RK':>2}  {result_label:40}  {'ID':20}  {'Score':>10}  {'Sigmoid':>10}  {'Status':>10}"
    )
    print("-" * 100)

    results = []
    known_for_query = known_train_positives.get(query_node_id, set())

    for rank in range(topk):
        candidate_node_id = int(candidate_pool_tensor[top_indices[rank]].item())
        candidate_entity_id = id2entity[str(candidate_node_id)]
        candidate_label = entity_labels.get(candidate_entity_id, candidate_entity_id)

        status = "KNOWN" if candidate_node_id in known_for_query else "CANDIDATE"
        display_id = candidate_entity_id if args.show_ids else "-"

        print(
            f"{rank + 1:02d}  "
            f"{candidate_label[:40]:40}  "
            f"{display_id:20}  "
            f"{top_scores[rank].item():10.4f}  "
            f"{top_probabilities[rank].item():10.4f}  "
            f"{status:>10}"
        )

        results.append(
            {
                "rank": rank + 1,
                "task": task_name,
                "query": {
                    "id": resolved_entity_id,
                    "name": query_label,
                    "type": query_entity_type,
                },
                "candidate": {
                    "id": candidate_entity_id,
                    "name": candidate_label,
                    "type": candidate_entity_type,
                },
                "score": float(top_scores[rank].item()),
                "sigmoid": float(top_probabilities[rank].item()),
                "status": status,
            }
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as file_obj:
            json.dump(results, file_obj, ensure_ascii=False, indent=2)
        print(f"\nSaved inference results to: {args.json_out}")


if __name__ == "__main__":
    main()
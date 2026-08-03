from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import torch

from models.shared.graph_artifacts import load_prepared_graph, resolve_processed_dir
from models.shared.ranking import (
    build_candidate_pool,
    build_training_graph,
    score_candidates,
)
from models.shared.splits import build_known_positives, get_or_create_splits, load_splits

from .components import DistMultDecoder, RGCNEncoder




def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_entity_labels(csv_path: str | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}

    if not csv_path or not os.path.exists(csv_path):
        return mapping

    with open(csv_path, "r", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        next(reader, None)

        for row in reader:
            if len(row) >= 2:
                mapping[row[0]] = row[1]

    return mapping


def normalize_drug_name(value: str) -> str:
    text = str(value or "").lower()
    text = text.replace("drug:", "")
    text = text.replace("drugbank:", "")
    text = text.replace("chebi:", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    return " ".join(text.split())


def load_pharmacological_groups(
    groups_csv: Path,
    confidence_mode: str,
) -> Dict[str, Set[str]]:
    rows = read_csv(groups_csv)

    allowed_confidences = {"high"}
    if confidence_mode == "high_medium":
        allowed_confidences = {"high", "medium"}

    drug_to_substances: Dict[str, Set[str]] = defaultdict(set)

    for row in rows:
        confidence = row.get("group_confidence", "").strip().lower()

        if confidence not in allowed_confidences:
            continue

        drug_name = row.get("drug_name", "").strip()
        active_ids = row.get("active_substance_ids", "").strip()

        if not drug_name or not active_ids:
            continue

        norm_name = normalize_drug_name(drug_name)

        for active_id in active_ids.split("|"):
            active_id = active_id.strip()
            if active_id:
                drug_to_substances[norm_name].add(active_id)

    return drug_to_substances


def get_groups_for_entity(
    entity_id: str,
    entity_label: str,
    drug_to_substances: Dict[str, Set[str]],
) -> Set[str]:
    candidates = [
        normalize_drug_name(entity_id),
        normalize_drug_name(entity_label),
    ]

    for candidate in candidates:
        if candidate in drug_to_substances:
            return drug_to_substances[candidate]

    return set()


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic evaluation of R-GCN predictions using pharmacological groups."
    )

    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--groups_csv",
        type=Path,
        required=True,
        help="Pharmacological groups CSV matching the evaluated KG period.",
    )
    parser.add_argument(
        "--confidence_mode",
        choices=["high", "high_medium"],
        default="high_medium",
    )
    parser.add_argument(
        "--entity_labels_csv",
        default=None,
    )
    parser.add_argument("--output_dir", default=None)

    args = parser.parse_args()

    run_dir = args.run_dir
    checkpoint_path = os.path.join(run_dir, "checkpoints", "best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    device = torch.device(cfg["training"].get("device", "cpu"))

    task_cfg = cfg["task"]
    task_name = task_cfg["name"]

    if task_name != "disease_to_drug":
        raise ValueError(
            f"This semantic evaluation currently supports disease_to_drug only. "
            f"Found task: {task_name}"
        )

    processed_dir = resolve_processed_dir(cfg["data"]["processed_dir"], task_name=task_name)

    prepared = load_prepared_graph(processed_dir)
    triples = prepared["triples"]
    entity2id = prepared["entity2id"]
    id2entity = prepared["id2entity"]
    rel2id = prepared["rel2id"]
    entity_types = prepared["entity_types"]

    entity_labels = load_entity_labels(args.entity_labels_csv)
    drug_to_substances = load_pharmacological_groups(
        groups_csv=args.groups_csv,
        confidence_mode=args.confidence_mode,
    )

    num_nodes = len(entity2id)
    num_relations = len(rel2id)

    model_cfg = cfg.get("model", {})
    add_inverse_edges = bool(model_cfg.get("add_inverse_edges", False))
    encoder_num_relations = num_relations * 2 if add_inverse_edges else num_relations

    split_cfg = cfg["split"]

    target_relation = task_cfg["target_relation"]
    candidate_entity_type = task_cfg["candidate_entity_type"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    relation_id = rel2id.get(target_relation)

    if relation_id is None:
        raise ValueError(f"Target relation not found: {target_relation}")

    seed = int(cfg.get("seed", 42))

    run_splits_dir = Path(run_dir) / "splits"
    global_splits_dir = processed_dir / "splits"

    if run_splits_dir.exists():
        print(f"Using splits saved in run: {run_splits_dir}")
        train_target, valid_target, test_target = load_splits(run_splits_dir)
        non_target_triples = [
            row for row in triples
            if (row.get("rel") or "").strip() != (target_relation or "")
        ]
    else:
        print(f"Using global splits: {global_splits_dir}")
        train_target, valid_target, test_target, non_target_triples = get_or_create_splits(
            output_dir=global_splits_dir,
            triples=triples,
            target_relation=target_relation,
            seed=seed,
            train_ratio=float(split_cfg["train_ratio"]),
            val_ratio=float(split_cfg["val_ratio"]),
            test_ratio=float(split_cfg["test_ratio"]),
            split_mode=split_cfg.get("mode") if isinstance(split_cfg, dict) else None,
        )

    eval_target = valid_target if args.split == "valid" else test_target

    graph, edge_types = build_training_graph(
        train_target=train_target,
        non_target_triples=non_target_triples,
        context_relations=context_relations,
        num_nodes=num_nodes,
        device=device,
        num_relations=num_relations,
        add_inverse_edges=add_inverse_edges,
    )

    required_candidate_ids = {
        int(row["src_id"])
        for row in train_target + valid_target + test_target
    }

    candidate_pool_tensor = build_candidate_pool(
        entity_types=entity_types,
        entity2id=entity2id,
        candidate_entity_type=candidate_entity_type,
        device=device,
        required_entity_ids=required_candidate_ids,
    )

    known_positives = build_known_positives(
        triples=train_target + valid_target + test_target,
        task_name=task_name,
    )

    encoder = RGCNEncoder(
        num_nodes,
        cfg["training"]["emb_dim"],
        encoder_num_relations,
        cfg["training"]["num_bases"],
        cfg["training"]["num_layers"],
        0.0,
        use_layer_norm=cfg.get("model", {}).get("use_layer_norm", True),
        use_residual=cfg.get("model", {}).get("use_residual", True),
        use_relation_gates=cfg.get("model", {}).get("use_relation_gates", False),
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

    detailed_rows: List[Dict[str, str]] = []

    total = 0
    mapped_gold = 0

    exact_hits = {1: 0, 3: 0, 10: 0}
    semantic_hits = {1: 0, 3: 0, 10: 0}
    semantic_hits_excluding_exact = {1: 0, 3: 0, 10: 0}

    raw_rr_sum = 0.0
    semantic_rr_sum = 0.0
    semantic_rr_excluding_exact_sum = 0.0
    best_jaccard_sum = 0.0

    k_values = [1, 3, 10]
    detail_topk = args.topk

    for row in eval_target:
        total += 1

        query_node_id = int(row["dst_id"])
        gold_node_id = int(row["src_id"])

        query_entity_id = row["dst"]
        gold_entity_id = row["src"]

        query_label = entity_labels.get(query_entity_id, query_entity_id)
        gold_label = entity_labels.get(gold_entity_id, gold_entity_id)

        gold_groups = get_groups_for_entity(
            entity_id=gold_entity_id,
            entity_label=gold_label,
            drug_to_substances=drug_to_substances,
        )

        if gold_groups:
            mapped_gold += 1

        with torch.no_grad():
            scores = score_candidates(
                task_name=task_name,
                decoder=decoder,
                node_embeddings=node_embeddings,
                relation_id=relation_id,
                query_id=query_node_id,
                candidate_ids=candidate_pool_tensor,
                device=device,
            )

        filtered_scores = scores.clone()

        known_for_query = known_positives.get(query_node_id, set())
        for idx, candidate_node_id_tensor in enumerate(candidate_pool_tensor):
            candidate_node_id = int(candidate_node_id_tensor.item())

            if candidate_node_id in known_for_query and candidate_node_id != gold_node_id:
                filtered_scores[idx] = float("-inf")

        ranking_size = len(candidate_pool_tensor)
        ranked_scores, ranked_indices = torch.topk(filtered_scores, k=ranking_size)

        exact_rank = None
        semantic_rank = None
        semantic_rank_excluding_exact = None
        best_jaccard = 0.0

        for rank_idx in range(1, ranking_size + 1):
            candidate_pool_index = int(ranked_indices[rank_idx - 1].item())
            candidate_node_id = int(candidate_pool_tensor[candidate_pool_index].item())
            candidate_entity_id = id2entity[str(candidate_node_id)]
            candidate_label = entity_labels.get(candidate_entity_id, candidate_entity_id)

            candidate_groups = get_groups_for_entity(
                entity_id=candidate_entity_id,
                entity_label=candidate_label,
                drug_to_substances=drug_to_substances,
            )

            is_exact = candidate_node_id == gold_node_id
            shared_groups = gold_groups & candidate_groups
            semantic_match = bool(gold_groups and candidate_groups and shared_groups)
            jac = jaccard(gold_groups, candidate_groups)

            best_jaccard = max(best_jaccard, jac)

            if is_exact and exact_rank is None:
                exact_rank = rank_idx

            if semantic_match and semantic_rank is None:
                semantic_rank = rank_idx

            if semantic_match and not is_exact and semantic_rank_excluding_exact is None:
                semantic_rank_excluding_exact = rank_idx

            if rank_idx <= detail_topk:
                detailed_rows.append(
                    {
                        "query_id": query_entity_id,
                        "query_label": query_label,
                        "gold_drug_id": gold_entity_id,
                        "gold_drug_label": gold_label,
                        "candidate_rank": str(rank_idx),
                        "candidate_drug_id": candidate_entity_id,
                        "candidate_drug_label": candidate_label,
                        "score": f"{float(ranked_scores[rank_idx - 1].item()):.6f}",
                        "exact_match": str(is_exact),
                        "semantic_match": str(semantic_match),
                        "semantic_match_excluding_exact": str(semantic_match and not is_exact),
                        "jaccard": f"{jac:.6f}",
                        "shared_active_substances": "|".join(sorted(shared_groups)),
                        "gold_has_group": str(bool(gold_groups)),
                        "candidate_has_group": str(bool(candidate_groups)),
                    }
                )

        for k in k_values:
            if exact_rank is not None and exact_rank <= k:
                exact_hits[k] += 1

            if semantic_rank is not None and semantic_rank <= k:
                semantic_hits[k] += 1

            if (
                semantic_rank_excluding_exact is not None
                and semantic_rank_excluding_exact <= k
            ):
                semantic_hits_excluding_exact[k] += 1

        if exact_rank is not None:
            raw_rr_sum += 1.0 / exact_rank

        if gold_groups:
            if semantic_rank is not None:
                semantic_rr_sum += 1.0 / semantic_rank

            if semantic_rank_excluding_exact is not None:
                semantic_rr_excluding_exact_sum += 1.0 / semantic_rank_excluding_exact

            best_jaccard_sum += best_jaccard

    output_dir = Path(args.output_dir) if args.output_dir else Path(run_dir) / "semantic_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    detailed_csv = output_dir / f"semantic_predictions_{args.split}_{args.confidence_mode}.csv"
    metrics_json = output_dir / f"semantic_metrics_{args.split}_{args.confidence_mode}.json"

    write_csv(
        detailed_csv,
        detailed_rows,
        [
            "query_id",
            "query_label",
            "gold_drug_id",
            "gold_drug_label",
            "candidate_rank",
            "candidate_drug_id",
            "candidate_drug_label",
            "score",
            "exact_match",
            "semantic_match",
            "semantic_match_excluding_exact",
            "jaccard",
            "shared_active_substances",
            "gold_has_group",
            "candidate_has_group",
        ],
    )

    metrics = {
        "split": args.split,
        "detail_topk": detail_topk,
        "confidence_mode": args.confidence_mode,
        "total_examples": total,
        "mapped_gold_examples": mapped_gold,
        "semantic_coverage": mapped_gold / total if total else 0.0,
        "metrics": {
            "raw_mrr": raw_rr_sum / total if total else 0.0,
            "semantic_mrr": semantic_rr_sum / mapped_gold if mapped_gold else 0.0,
            "semantic_mrr_excluding_exact": (
                semantic_rr_excluding_exact_sum / mapped_gold if mapped_gold else 0.0
            ),
            "mean_best_jaccard": (
                best_jaccard_sum / mapped_gold if mapped_gold else 0.0
            ),
        },
        "outputs": {
            "detailed_csv": str(detailed_csv),
        },
    }

    for k in k_values:
        metrics["metrics"][f"exact_hits@{k}"] = exact_hits[k] / total if total else 0.0
        metrics["metrics"][f"semantic_hits@{k}"] = (
            semantic_hits[k] / mapped_gold if mapped_gold else 0.0
        )
        metrics["metrics"][f"semantic_hits_excluding_exact@{k}"] = (
            semantic_hits_excluding_exact[k] / mapped_gold if mapped_gold else 0.0
        )

    metrics_json.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved detailed semantic predictions: {detailed_csv}")
    print(f"Saved semantic metrics: {metrics_json}")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
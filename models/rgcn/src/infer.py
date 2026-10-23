"""
R-GCN inference script.

Generate top-K recommendations or predictions for a configurable task using a
trained R-GCN model.

Supported task modes:
    - disease_to_drug
    - drug_to_disease
    - tail_prediction

Usage examples:
    python -m models.rgcn.src.infer --query "asthma" --topk 10
    python -m models.rgcn.src.infer --run_dir <run_dir> --entity_id "DOID:9352" --topk 10
    python -m models.rgcn.src.infer --query "asthma" --mode audit --topk 40

Modes:
    - audit: rank all candidates and identify TRAIN, VALIDATION, TEST, and
      UNOBSERVED associations.
    - exploration (default): exclude associations present in every experimental
      split and rank only UNOBSERVED pairs.

For generic benchmark tail prediction, provide the relation explicitly:
    python -m models.rgcn.src.infer \
        --run_dir results/runs/rgcn/<run_name> \
        --entity_id "<HEAD_ENTITY_ID>" \
        --relation "<RELATION_NAME>" \
        --topk 10
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple

import torch

from models.shared.graph_artifacts import load_prepared_graph, resolve_processed_dir
from models.shared.evaluation import filtered_rank
from models.shared.io import read_json, save_json
from models.shared.ranking import build_candidate_pool, build_training_graph, score_candidates
from models.shared.runs import find_latest_run
from models.shared.splits import build_known_positives, get_or_create_splits, load_splits

from .components import DistMultDecoder, RGCNEncoder
from .inference.query_resolver import (
    EntityAliases,
    QueryNotFoundError,
    QuerySelectionCancelled,
    preferred_entity_label,
    resolve_query_entity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BIOMEDICAL_RUN_DIR = (
    PROJECT_ROOT
    / "results/runs/rgcn/biomedical/disease_to_drug/hyperparam_combined_best/seed_42"
)


def load_entity_labels(csv_path: str | Path | None) -> Dict[str, str]:
    """Load ``entity_id`` and ``label`` columns from an optional CSV file."""
    if csv_path is None:
        return {}

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Entity labels CSV not found: {path}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        required = {"entity_id", "label"}
        if not required.issubset(fieldnames):
            raise ValueError(
                f"Entity labels CSV must contain columns {sorted(required)}: {path}"
            )

        for row in reader:
            entity_id = (row.get("entity_id") or "").strip()
            label = (row.get("label") or "").strip()
            if entity_id and label:
                mapping[entity_id] = label

    return mapping


def load_entity_aliases(csv_path: str | Path | None) -> EntityAliases:
    """Load aliases indexed by canonical entity ID from an optional CSV file."""
    if csv_path is None:
        return {}

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Entity aliases CSV not found: {path}")

    mapping: EntityAliases = {}
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        required = {"alias", "entity_id", "alias_type"}
        if not required.issubset(fieldnames):
            raise ValueError(
                f"Entity aliases CSV must contain columns {sorted(required)}: {path}"
            )

        for row in reader:
            entity_id = (row.get("entity_id") or "").strip()
            alias = (row.get("alias") or "").strip()
            alias_type = (row.get("alias_type") or "").strip()
            if entity_id and alias:
                mapping.setdefault(entity_id, []).append((alias, alias_type))

    return mapping


def discover_mapping_csv(
    explicit_path: str | None,
    cfg: Dict[str, Any],
    filename: str,
) -> Path | None:
    """Resolve an explicit mapping CSV or discover it beside the graph build."""
    if explicit_path:
        return Path(explicit_path)

    configured_path = cfg.get("data", {}).get(filename.replace(".csv", "_csv"))
    if configured_path:
        return Path(configured_path)

    triples_path = cfg.get("data", {}).get("triples_path")
    if not triples_path:
        return None

    triples_path = Path(triples_path)
    if triples_path.parent.name != "triples":
        return None

    candidate = triples_path.parent.parent / "mappings" / filename
    return candidate if candidate.exists() else None


def original_entity_names(
    entity_id: str,
    entity_type: str,
    entity_aliases: EntityAliases,
) -> List[str]:
    """Return distinct original names associated with one canonical entity."""
    alias_type = {
        "DRUG": "DRUG_NAME_ORIGINAL",
        "DISEASE": "DISEASE_LABEL",
        "INGREDIENT": "ING_NAME_ORIGINAL",
    }.get(entity_type)
    if alias_type is None:
        return []

    names: List[str] = []
    seen: Set[str] = set()
    for alias, current_type in entity_aliases.get(entity_id, []):
        normalized = alias.strip().casefold()
        if current_type == alias_type and normalized and normalized not in seen:
            names.append(alias.strip())
            seen.add(normalized)
    return names


def load_inference_splits(
    run_dir: str | Path,
    processed_dir: str | Path,
    triples: List[Dict[str, str]],
    target_relation: str | None,
    split_cfg: Dict[str, Any],
    split_seed: int,
) -> Tuple[
    List[Dict[str, str]],
    List[Dict[str, str]],
    List[Dict[str, str]],
    List[Dict[str, str]],
    Path,
]:
    """Load the exact split copied into the run, falling back to prepared data."""
    run_splits_dir = Path(run_dir) / "splits"
    required = (
        run_splits_dir / "train_triples.json",
        run_splits_dir / "valid_triples.json",
        run_splits_dir / "test_triples.json",
    )

    if all(path.exists() for path in required):
        train_target, valid_target, test_target = load_splits(run_splits_dir)
        if target_relation is None:
            non_target_triples = []
        else:
            non_target_triples = [
                row
                for row in triples
                if (row.get("rel") or "").strip() != target_relation
            ]
        return (
            train_target,
            valid_target,
            test_target,
            non_target_triples,
            run_splits_dir,
        )

    processed_splits_dir = Path(processed_dir) / "splits"
    train_target, valid_target, test_target, non_target_triples = get_or_create_splits(
        output_dir=processed_splits_dir,
        triples=triples,
        target_relation=target_relation,
        seed=split_seed,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        split_mode=split_cfg.get("mode"),
    )
    return (
        train_target,
        valid_target,
        test_target,
        non_target_triples,
        processed_splits_dir,
    )


def build_split_positive_maps(
    task_name: str,
    train_target: List[Dict[str, str]],
    valid_target: List[Dict[str, str]],
    test_target: List[Dict[str, str]],
) -> Dict[str, Dict[Any, set]]:
    """Index positives separately so inference can report their provenance."""
    kwargs: Dict[str, Any]
    if task_name == "tail_prediction":
        kwargs = {
            "task_name": None,
            "query_key": "src_id",
            "relation_key": "rel_id",
            "candidate_key": "dst_id",
        }
    else:
        kwargs = {"task_name": task_name}

    return {
        "TRAIN": build_known_positives(triples=train_target, **kwargs),
        "VALIDATION": build_known_positives(triples=valid_target, **kwargs),
        "TEST": build_known_positives(triples=test_target, **kwargs),
    }


def statuses_for_query(
    split_positive_maps: Dict[str, Dict[Any, set]],
    query_key: Any,
) -> Dict[int, str]:
    """Return one split status for every observed candidate of a query."""
    statuses: Dict[int, str] = {}
    for status in ("TRAIN", "VALIDATION", "TEST"):
        for candidate_id in split_positive_maps[status].get(query_key, set()):
            candidate_id = int(candidate_id)
            previous = statuses.get(candidate_id)
            if previous is not None:
                raise ValueError(
                    "A positive association occurs in more than one split for "
                    f"query={query_key}, candidate={candidate_id}: "
                    f"{previous} and {status}."
                )
            statuses[candidate_id] = status
    return statuses


def relation_candidates_for_query(
    triples: List[Dict[str, str]],
    task_name: str,
    query_node_id: int,
    relation: str,
) -> Set[int]:
    """Return candidates connected to a query by a selected graph relation."""
    candidates: Set[int] = set()
    for row in triples:
        if (row.get("rel") or "").strip() != relation:
            continue

        src_id = int(row["src_id"])
        dst_id = int(row["dst_id"])
        if task_name == "disease_to_drug" and dst_id == query_node_id:
            candidates.add(src_id)
        elif task_name == "drug_to_disease" and src_id == query_node_id:
            candidates.add(dst_id)

    return candidates


def relation_targets_by_source(
    triples: List[Dict[str, str]],
    relation: str,
) -> Dict[int, Set[int]]:
    """Index destination nodes by source node for one graph relation."""
    targets: Dict[int, Set[int]] = {}
    for row in triples:
        if (row.get("rel") or "").strip() != relation:
            continue
        targets.setdefault(int(row["src_id"]), set()).add(int(row["dst_id"]))
    return targets


def resolve_inference_run_dir(
    explicit_run_dir: str | None,
    runs_dir: str | Path,
) -> Path:
    """Resolve the requested run, preferring the designated biomedical model."""
    if explicit_run_dir:
        return Path(explicit_run_dir)

    if DEFAULT_BIOMEDICAL_RUN_DIR.exists():
        return DEFAULT_BIOMEDICAL_RUN_DIR

    biomedical_runs = Path(runs_dir) / "biomedical" / "disease_to_drug"
    return Path(find_latest_run(biomedical_runs))


def filtered_candidate_rank(
    candidate_pool_ids: List[int],
    scores: List[float],
    candidate_id: int,
    known_candidate_ids: Set[int],
) -> float:
    """Compute the same filtered rank used by the evaluation module."""
    return filtered_rank(
        candidate_ids=candidate_pool_ids,
        scores=scores,
        true_candidate_id=candidate_id,
        known_positive_ids=known_candidate_ids,
    )


def eligible_candidate_positions(
    candidate_pool_ids: List[int],
    candidate_statuses: Dict[int, str],
    contraindicated_candidates: Set[int],
    mode: str,
    exclude_contraindications: bool,
) -> Tuple[List[int], int]:
    """Select candidate positions according to explicit inference filters."""
    if mode == "exploration":
        positions = [
            position
            for position, candidate_id in enumerate(candidate_pool_ids)
            if candidate_id not in candidate_statuses
        ]
    else:
        positions = list(range(len(candidate_pool_ids)))

    contraindication_exclusions = 0
    if exclude_contraindications:
        before_filter = len(positions)
        positions = [
            position
            for position in positions
            if candidate_pool_ids[position] not in contraindicated_candidates
        ]
        contraindication_exclusions = before_filter - len(positions)

    return positions, contraindication_exclusions


def sha256_file(path: str | Path) -> str:
    """Calculate a SHA-256 digest for a provenance record."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """
    Load a trained checkpoint and generate top-K ranked predictions for the
    configured task.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Rank candidate associations with a trained R-GCN. For the default "
            "biomedical model, only --query is required."
        )
    )
    parser.add_argument("--runs_dir", default="results/runs/rgcn")
    parser.add_argument("--run_dir", default=None)
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        default=None,
        help="Entity name or partial query string.",
    )
    query_group.add_argument(
        "--entity_id",
        default=None,
        help="Exact canonical entity ID.",
    )
    parser.add_argument(
        "--relation",
        default=None,
        help="Canonical relation name for tail_prediction.",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--mode",
        choices=("audit", "exploration"),
        default="exploration",
        help=(
            "audit ranks every candidate and reports TRAIN/VALIDATION/TEST/UNOBSERVED; "
            "exploration (default) excludes every known positive association."
        ),
    )
    parser.add_argument("--show_ids", action="store_true", help="Display canonical entity IDs.")
    parser.add_argument(
        "--details",
        action="store_true",
        help="Display internal status and raw model score columns.",
    )
    parser.add_argument(
        "--exclude_contraindications",
        action="store_true",
        help=(
            "Exclude pairs with a CONTRAINDICATION relation in this graph. "
            "The relation remains part of the trained graph."
        ),
    )
    parser.add_argument("--json_out", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--entity_labels_csv",
        default=None,
        help=(
            "Optional CSV with entity_id and label columns. When omitted, the "
            "mapping produced with the graph is discovered automatically."
        ),
    )
    parser.add_argument(
        "--entity_aliases_csv",
        default=None,
        help=(
            "Optional CSV with alias, entity_id and alias_type columns. When omitted, "
            "the mapping produced with the graph is discovered automatically."
        ),
    )
    args = parser.parse_args()

    if args.topk <= 0:
        parser.error("--topk must be greater than zero.")

    try:
        run_dir = resolve_inference_run_dir(args.run_dir, args.runs_dir)
    except FileNotFoundError as error:
        parser.error(str(error))
    checkpoint_path = run_dir / "checkpoints" / "best.pt"

    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["config"]
    device = torch.device(cfg["training"].get("device", "cpu"))

    task_cfg = cfg["task"]
    task_name = task_cfg["name"]
    if args.exclude_contraindications and task_name not in {
        "disease_to_drug",
        "drug_to_disease",
    }:
        parser.error(
            "--exclude_contraindications is only available for biomedical tasks."
        )
    processed_dir = resolve_processed_dir(cfg["data"]["processed_dir"], task_name=task_name)

    prepared = load_prepared_graph(processed_dir)
    triples = prepared["triples"]
    entity2id = prepared["entity2id"]
    id2entity = prepared["id2entity"]
    rel2id = prepared["rel2id"]
    entity_types = prepared["entity_types"]

    labels_path = discover_mapping_csv(args.entity_labels_csv, cfg, "entity_table.csv")
    aliases_path = discover_mapping_csv(args.entity_aliases_csv, cfg, "aliases.csv")
    entity_labels = load_entity_labels(labels_path)
    entity_aliases = load_entity_aliases(aliases_path)

    num_nodes = len(entity2id)
    num_relations = len(rel2id)

    model_cfg = cfg.get("model", {})
    add_inverse_edges = bool(model_cfg.get("add_inverse_edges", False))
    encoder_num_relations = num_relations * 2 if add_inverse_edges else num_relations

    split_cfg = cfg["split"]

    target_relation = task_cfg["target_relation"]
    query_entity_type = task_cfg["query_entity_type"]
    candidate_entity_type = task_cfg["candidate_entity_type"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    if task_name == "tail_prediction":
        if not args.relation:
            raise ValueError("For tail_prediction you must provide --relation.")

        relation_id = rel2id.get(args.relation)

        if relation_id is None:
            raise ValueError(f"Relation not found in graph: {args.relation}")
    else:
        relation_id = rel2id.get(target_relation)

        if relation_id is None:
            raise ValueError(f"Target relation not found: {target_relation}")

    seed = int(cfg.get("seed", 42))
    split_seed = int(split_cfg.get("seed", seed))

    (
        train_target,
        valid_target,
        test_target,
        non_target_triples,
        splits_dir,
    ) = load_inference_splits(
        run_dir=run_dir,
        processed_dir=processed_dir,
        triples=triples,
        target_relation=target_relation,
        split_cfg=split_cfg,
        split_seed=split_seed,
    )

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
        int(row["src_id"]) if task_name == "disease_to_drug" else int(row["dst_id"])
        for row in train_target + valid_target + test_target
    }

    explicit_candidate_path = task_cfg.get("candidate_entities_path")
    explicit_candidate_entities = (
        read_json(explicit_candidate_path) if explicit_candidate_path else None
    )

    candidate_pool_tensor = build_candidate_pool(
        entity_types=entity_types,
        entity2id=entity2id,
        candidate_entity_type=candidate_entity_type,
        device=device,
        required_entity_ids=required_candidate_ids,
        explicit_entity_ids=explicit_candidate_entities,
    )

    split_positive_maps = build_split_positive_maps(
        task_name=task_name,
        train_target=train_target,
        valid_target=valid_target,
        test_target=test_target,
    )

    try:
        resolved_entity_id, query_node_id = resolve_query_entity(
            query=args.query,
            entity_id=args.entity_id,
            expected_entity_type=query_entity_type,
            entity2id=entity2id,
            entity_types=entity_types,
            entity_labels=entity_labels,
            entity_aliases=entity_aliases,
            interactive=bool(args.query and sys.stdin.isatty()),
        )
    except QuerySelectionCancelled:
        print("Query selection cancelled.")
        return
    except QueryNotFoundError as error:
        print(f"\n{error}")
        return
    except ValueError as error:
        parser.error(str(error))

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
        scores = score_candidates(
            task_name=task_name,
            decoder=decoder,
            node_embeddings=node_embeddings,
            relation_id=relation_id,
            query_id=query_node_id,
            candidate_ids=candidate_pool_tensor,
            device=device,
        )

    query_key: Any
    if task_name == "tail_prediction":
        query_key = (query_node_id, relation_id)
    else:
        query_key = query_node_id

    candidate_statuses = statuses_for_query(split_positive_maps, query_key)
    contraindicated_candidates = relation_candidates_for_query(
        triples=triples,
        task_name=task_name,
        query_node_id=query_node_id,
        relation="CONTRAINDICATION",
    )
    ingredient_ids_by_drug = (
        relation_targets_by_source(triples, "HAS_INGREDIENT")
        if candidate_entity_type == "DRUG"
        else {}
    )
    candidate_pool_ids = [
        int(value) for value in candidate_pool_tensor.detach().cpu().tolist()
    ]

    eligible_positions, contraindication_exclusions = eligible_candidate_positions(
        candidate_pool_ids=candidate_pool_ids,
        candidate_statuses=candidate_statuses,
        contraindicated_candidates=contraindicated_candidates,
        mode=args.mode,
        exclude_contraindications=args.exclude_contraindications,
    )

    if not eligible_positions:
        raise ValueError(
            f"No eligible candidates remain for query {resolved_entity_id} "
            f"in mode {args.mode}."
        )

    eligible_indices = torch.tensor(
        eligible_positions,
        dtype=torch.long,
        device=scores.device,
    )
    eligible_scores = scores[eligible_indices]
    topk = min(args.topk, len(eligible_positions))
    top_scores, relative_top_indices = torch.topk(eligible_scores, k=topk)
    top_indices = eligible_indices[relative_top_indices]

    query_label = preferred_entity_label(
        resolved_entity_id,
        entity_types.get(resolved_entity_id, query_entity_type),
        entity_labels,
        entity_aliases,
    )

    biomedical_task = task_name in {"disease_to_drug", "drug_to_disease"}
    technical_output = args.mode == "audit" or args.details or args.show_ids
    if candidate_entity_type == "ALL":
        result_label = "Entity"
        candidate_pool_label = "entities"
    elif candidate_entity_type == "DRUG":
        result_label = "Drug"
        candidate_pool_label = "drugs"
    else:
        result_label = "Disease"
        candidate_pool_label = "diseases"

    ranking_scope = "unobserved" if args.mode == "exploration" else "ranked"
    print(f"\nTop-{topk} {ranking_scope} {candidate_pool_label}")
    if task_name == "disease_to_drug":
        print(f"Disease: {query_label} ({resolved_entity_id})")
    elif task_name == "drug_to_disease":
        print(f"Drug: {query_label} ({resolved_entity_id})")
    else:
        print(f"Query: {query_label} ({resolved_entity_id})")
    print(f"Ranking mode: {args.mode}")
    print(f"Candidate pool: {len(candidate_pool_ids):,} {candidate_pool_label}")
    if args.mode == "exploration":
        print(
            f"Known {target_relation or args.relation} associations excluded: "
            f"{len(candidate_statuses)}"
        )
    else:
        print("Audit mode: known and unobserved associations are shown together.")
    if args.exclude_contraindications:
        print(
            "Candidates excluded by a CONTRAINDICATION relation in this graph: "
            f"{contraindication_exclusions}"
        )
    print(f"Candidates ranked: {len(eligible_positions):,}")
    if biomedical_task:
        print(
            "The ranking is computational. It does not establish a new indication, "
            "clinical efficacy, or safety."
        )

    if technical_output:
        print("Raw scores are ranking values, not calibrated probabilities.")
        if args.mode == "audit":
            print("-" * 132)
            print(
                f"{'Rank':>4}  {result_label:38}  {'ID':32}  {'Score':>10}  "
                f"{'Status':>10}  {'Filtered':>8}"
            )
            print("-" * 132)
        else:
            print("-" * 92)
            print(
                f"{'Rank':>4}  {result_label:38}  {'ID':32}  {'Score':>10}"
            )
            print("-" * 92)
    else:
        print("-" * 52)
        print(f"{'Rank':>4}  {result_label}")
        print("-" * 52)

    results = []
    all_scores = scores.detach().cpu().tolist()
    known_candidate_ids = set(candidate_statuses)

    for rank in range(topk):
        candidate_node_id = int(candidate_pool_tensor[top_indices[rank]].item())
        candidate_entity_id = id2entity[str(candidate_node_id)]
        actual_candidate_type = entity_types.get(
            candidate_entity_id,
            candidate_entity_type,
        )
        candidate_label = preferred_entity_label(
            candidate_entity_id,
            actual_candidate_type,
            entity_labels,
            entity_aliases,
        )
        original_names = original_entity_names(
            candidate_entity_id,
            actual_candidate_type,
            entity_aliases,
        )
        multiple_original_names = len(original_names) > 1
        display_label = candidate_label

        ingredient_ids = sorted(ingredient_ids_by_drug.get(candidate_node_id, set()))
        ingredients = []
        for ingredient_node_id in ingredient_ids:
            ingredient_entity_id = id2entity.get(str(ingredient_node_id))
            if ingredient_entity_id is None:
                continue
            ingredients.append(
                {
                    "id": ingredient_entity_id,
                    "name": preferred_entity_label(
                        ingredient_entity_id,
                        entity_types.get(ingredient_entity_id, "INGREDIENT"),
                        entity_labels,
                        entity_aliases,
                    ),
                }
            )

        status = candidate_statuses.get(candidate_node_id, "UNOBSERVED")
        contraindication_in_graph = candidate_node_id in contraindicated_candidates
        evaluation_filtered_rank = None
        if status in {"VALIDATION", "TEST"}:
            evaluation_filtered_rank = filtered_candidate_rank(
                candidate_pool_ids=candidate_pool_ids,
                scores=all_scores,
                candidate_id=candidate_node_id,
                known_candidate_ids=known_candidate_ids,
            )

        if technical_output and args.mode == "audit":
            filtered_display = (
                f"{evaluation_filtered_rank:g}"
                if evaluation_filtered_rank is not None
                else "-"
            )
            row_text = (
                f"{rank + 1:>4}  "
                f"{display_label[:38]:38}  "
                f"{candidate_entity_id[:32]:32}  "
                f"{top_scores[rank].item():10.4f}  "
                f"{status:>10}  "
                f"{filtered_display:>8}"
            )
            print(row_text)
        elif technical_output:
            row_text = (
                f"{rank + 1:>4}  "
                f"{display_label[:38]:38}  "
                f"{candidate_entity_id[:32]:32}  "
                f"{top_scores[rank].item():10.4f}"
            )
            print(row_text)
        else:
            row_text = f"{rank + 1:>4}  {display_label[:46]}"
            print(row_text)

        results.append(
            {
                "rank": rank + 1,
                "candidate": {
                    "id": candidate_entity_id,
                    "name": candidate_label,
                    "type": actual_candidate_type,
                    "original_names": original_names,
                    "multiple_original_names": multiple_original_names,
                    "ingredients_in_graph": ingredients,
                },
                "score": float(top_scores[rank].item()),
                "status": status,
                "evaluation_filtered_rank": evaluation_filtered_rank,
                "context": {
                    "contraindication_in_graph": contraindication_in_graph,
                },
            }
        )

    if args.json_out:
        relation_name = args.relation if task_name == "tail_prediction" else target_relation
        split_files = {
            split_name: Path(splits_dir) / filename
            for split_name, filename in (
                ("train", "train_triples.json"),
                ("validation", "valid_triples.json"),
                ("test", "test_triples.json"),
            )
        }
        run_config_path = Path(run_dir) / "config.yaml"
        output = {
            "metadata": {
                "task": task_name,
                "mode": args.mode,
                "relation": relation_name,
                "query": {
                    "id": resolved_entity_id,
                    "name": query_label,
                    "type": entity_types.get(resolved_entity_id, query_entity_type),
                },
                "candidate_pool_size": len(candidate_pool_ids),
                "eligible_candidate_count": len(eligible_positions),
                "filters": {
                    "known_target_associations_excluded": args.mode == "exploration",
                    "known_target_associations_excluded_count": (
                        len(candidate_statuses) if args.mode == "exploration" else 0
                    ),
                    "contraindications_excluded": args.exclude_contraindications,
                    "contraindications_excluded_count": contraindication_exclusions,
                },
                "known_for_query": {
                    status.lower(): sum(
                        1 for value in candidate_statuses.values() if value == status
                    )
                    for status in ("TRAIN", "VALIDATION", "TEST")
                },
                "contraindications_for_query": len(contraindicated_candidates),
                "context_semantics": {
                    "contraindication_in_graph": (
                        "The same pair has a CONTRAINDICATION relation in the graph. "
                        "The candidate is retained, and this flag is not a clinical "
                        "safety assessment."
                    ),
                    "multiple_original_names": (
                        "More than one distinct original name maps to the same "
                        "canonical entity. These names may represent synonyms, "
                        "brands, formulations, or a mapping that requires review."
                    ),
                    "ingredients_in_graph": (
                        "Ingredients connected to the candidate by HAS_INGREDIENT "
                        "relations. This may include excipients and must not be "
                        "interpreted as a list of active substances."
                    ),
                },
                "requested_topk": args.topk,
                "returned_topk": topk,
                "score_semantics": (
                    "Raw DistMult ranking score; it is not a calibrated probability "
                    "or a measure of clinical efficacy."
                ),
                "status_semantics": {
                    "TRAIN": "Association included in the training split.",
                    "VALIDATION": "Association reserved for model validation.",
                    "TEST": "Association reserved for final evaluation.",
                    "UNOBSERVED": (
                        "No target-relation triple for this pair is present in the "
                        "current graph splits; this does not establish novelty or "
                        "clinical validity."
                    ),
                },
                "run_dir": str(Path(run_dir)),
                "run_config": str(run_config_path),
                "run_config_sha256": sha256_file(run_config_path),
                "checkpoint": str(Path(checkpoint_path)),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "processed_graph_dir": str(Path(processed_dir)),
                "splits_dir": str(Path(splits_dir)),
                "split_sha256": {
                    split_name: sha256_file(path)
                    for split_name, path in split_files.items()
                },
                "entity_labels_csv": str(labels_path) if labels_path else None,
                "entity_aliases_csv": str(aliases_path) if aliases_path else None,
                "training_seed": seed,
                "split_seed": split_seed,
            },
            "results": results,
        }
        save_json(args.json_out, output)

        print(f"\nSaved inference results to: {args.json_out}")


if __name__ == "__main__":
    main()

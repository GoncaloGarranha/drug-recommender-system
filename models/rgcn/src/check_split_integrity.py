"""
Validate R-GCN split integrity and quantify possible biomedical context leakage.

Usage:
    python -m models.rgcn.src.check_split_integrity \
      --run_dir results/runs/rgcn/biomedical/disease_to_drug/with_dgidb/seed_42

Outputs:
    <run_dir>/split_integrity/integrity_report.json
    <run_dir>/split_integrity/context_paths_test.csv
    <run_dir>/split_integrity/context_paths_valid.csv
    <run_dir>/split_integrity/direct_context_same_pair.csv
    <run_dir>/split_integrity/normalized_duplicate_entities.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch

from models.shared.graph_artifacts import load_prepared_graph, resolve_processed_dir
from models.shared.splits import load_splits

Triple = Dict[str, Any]
Pair = Tuple[str, str]


def as_str(value: Any) -> str:
    return str(value).strip()


def triple_key(row: Triple, use_ids: bool = True) -> Tuple[str, str, str]:
    if use_ids and all(k in row for k in ("src_id", "rel_id", "dst_id")):
        return (as_str(row["src_id"]), as_str(row["rel_id"]), as_str(row["dst_id"]))
    return (as_str(row.get("src")), as_str(row.get("rel")), as_str(row.get("dst")))


def pair_key(row: Triple, use_ids: bool = True) -> Pair:
    if use_ids and all(k in row for k in ("src_id", "dst_id")):
        return (as_str(row["src_id"]), as_str(row["dst_id"]))
    return (as_str(row.get("src")), as_str(row.get("dst")))


def reverse_pair_key(row: Triple, use_ids: bool = True) -> Pair:
    a, b = pair_key(row, use_ids=use_ids)
    return (b, a)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_entity_name(entity: str) -> str:
    text = str(entity or "").lower()
    text = re.sub(r"^(drug|drugbank|chebi|doid|gene):", "", text)
    text = text.replace("https://go.drugbank.com/drugs/", "")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def infer_type(entity: str, entity_types: Dict[str, str], entity2id: Dict[str, Any]) -> str:
    if entity in entity_types:
        return entity_types[entity]
    eid = entity2id.get(entity)
    if eid is not None and str(eid) in entity_types:
        return entity_types[str(eid)]
    upper = entity.upper()
    if upper.startswith("DRUG") or upper.startswith("DRUGBANK") or upper.startswith("CHEBI"):
        return "drug"
    if upper.startswith("DOID"):
        return "disease"
    if upper.startswith("GENE"):
        return "gene"
    return "unknown"


def relation_inverse_names(rel: str) -> Set[str]:
    return {
        f"{rel}_INV",
        f"INV_{rel}",
        f"inverse_{rel}",
        f"{rel}_reverse",
        f"reverse_{rel}",
    }


def relation_is(row: Triple, rel_name: str) -> bool:
    return as_str(row.get("rel")) == rel_name


def build_relation_rows(triples: Iterable[Triple], rel_name: str) -> List[Triple]:
    return [r for r in triples if relation_is(r, rel_name)]


def count_intersections(a: Set[Any], b: Set[Any]) -> int:
    return len(a & b)


def pct(n: int, d: int) -> float:
    return round((100.0 * n / d), 4) if d else 0.0


def load_checkpoint_config(run_dir: Path) -> Dict[str, Any]:
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'config' entry.")
    return checkpoint["config"]


def build_context_edges(
    triples: List[Triple],
    train_target: List[Triple],
    target_relation: str,
    context_relations: Sequence[str],
) -> List[Triple]:
    context_set = set(context_relations or [target_relation])
    non_target = [r for r in triples if as_str(r.get("rel")) != target_relation]
    rows = []
    if target_relation in context_set:
        rows.extend(train_target)
    rows.extend([r for r in non_target if as_str(r.get("rel")) in context_set])
    return rows


def index_gene_context(context_edges: Iterable[Triple]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    drug_to_genes: Dict[str, Set[str]] = defaultdict(set)
    disease_to_genes: Dict[str, Set[str]] = defaultdict(set)

    for row in context_edges:
        rel = as_str(row.get("rel"))
        src = as_str(row.get("src"))
        dst = as_str(row.get("dst"))

        if rel == "INTERACTS_WITH_GENE":
            # Expected: Drug -> Gene, but accept reversed rows too.
            if src.startswith(("DRUG:", "DRUGBANK:", "CHEBI:")) and dst.startswith("GENE:"):
                drug_to_genes[as_str(row.get("src_id", src))].add(as_str(row.get("dst_id", dst)))
            elif dst.startswith(("DRUG:", "DRUGBANK:", "CHEBI:")) and src.startswith("GENE:"):
                drug_to_genes[as_str(row.get("dst_id", dst))].add(as_str(row.get("src_id", src)))

        elif rel == "ASSOCIATED_WITH_GENE":
            # Expected: Disease -> Gene, but accept reversed rows too.
            if src.startswith("DOID:") and dst.startswith("GENE:"):
                disease_to_genes[as_str(row.get("src_id", src))].add(as_str(row.get("dst_id", dst)))
            elif dst.startswith("DOID:") and src.startswith("GENE:"):
                disease_to_genes[as_str(row.get("dst_id", dst))].add(as_str(row.get("src_id", src)))

    return drug_to_genes, disease_to_genes


def index_ingredients(context_edges: Iterable[Triple]) -> Dict[str, Set[str]]:
    drug_to_ingredients: Dict[str, Set[str]] = defaultdict(set)
    for row in context_edges:
        if as_str(row.get("rel")) != "HAS_INGREDIENT":
            continue
        src = as_str(row.get("src"))
        dst = as_str(row.get("dst"))
        src_id = as_str(row.get("src_id", src))
        dst_id = as_str(row.get("dst_id", dst))
        if src.startswith(("DRUG:", "DRUGBANK:", "CHEBI:")):
            drug_to_ingredients[src_id].add(dst_id)
        elif dst.startswith(("DRUG:", "DRUGBANK:", "CHEBI:")):
            drug_to_ingredients[dst_id].add(src_id)
    return drug_to_ingredients


def build_known_drugs_by_disease(train_target: Iterable[Triple]) -> Dict[str, Set[str]]:
    disease_to_drugs: Dict[str, Set[str]] = defaultdict(set)
    for row in train_target:
        drug_id = as_str(row.get("src_id"))
        disease_id = as_str(row.get("dst_id"))
        disease_to_drugs[disease_id].add(drug_id)
    return disease_to_drugs


def context_path_rows(
    eval_rows: List[Triple],
    split_name: str,
    drug_to_genes: Dict[str, Set[str]],
    disease_to_genes: Dict[str, Set[str]],
    drug_to_ingredients: Dict[str, Set[str]],
    disease_to_train_drugs: Dict[str, Set[str]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    has_gene_path = 0
    has_ingredient_support = 0

    for row in eval_rows:
        drug_id = as_str(row.get("src_id"))
        disease_id = as_str(row.get("dst_id"))
        dgenes = drug_to_genes.get(drug_id, set())
        disgenes = disease_to_genes.get(disease_id, set())
        shared_genes = dgenes & disgenes

        ingredients = drug_to_ingredients.get(drug_id, set())
        shared_ingredient_train_drugs: Set[str] = set()
        for train_drug in disease_to_train_drugs.get(disease_id, set()):
            if ingredients & drug_to_ingredients.get(train_drug, set()):
                shared_ingredient_train_drugs.add(train_drug)

        if shared_genes:
            has_gene_path += 1
        if shared_ingredient_train_drugs:
            has_ingredient_support += 1

        out.append(
            {
                "split": split_name,
                "drug_id": drug_id,
                "drug": row.get("src"),
                "disease_id": disease_id,
                "disease": row.get("dst"),
                "shared_gene_count": len(shared_genes),
                "drug_gene_count": len(dgenes),
                "disease_gene_count": len(disgenes),
                "has_drug_gene_disease_path": bool(shared_genes),
                "drug_ingredient_count": len(ingredients),
                "shared_ingredient_train_drug_count": len(shared_ingredient_train_drugs),
                "has_ingredient_support_from_train_indications": bool(shared_ingredient_train_drugs),
            }
        )

    summary = {
        "num_edges": len(eval_rows),
        "edges_with_drug_gene_disease_path": has_gene_path,
        "pct_with_drug_gene_disease_path": pct(has_gene_path, len(eval_rows)),
        "edges_with_ingredient_support_from_train_indications": has_ingredient_support,
        "pct_with_ingredient_support_from_train_indications": pct(has_ingredient_support, len(eval_rows)),
    }
    return out, summary


def find_direct_context_same_pair(
    eval_rows: List[Triple],
    split_name: str,
    non_target_triples: Iterable[Triple],
) -> List[Dict[str, Any]]:
    by_pair: Dict[Pair, List[Triple]] = defaultdict(list)
    by_reverse_pair: Dict[Pair, List[Triple]] = defaultdict(list)
    for row in non_target_triples:
        by_pair[pair_key(row, use_ids=True)].append(row)
        by_reverse_pair[reverse_pair_key(row, use_ids=True)].append(row)

    cases: List[Dict[str, Any]] = []
    for row in eval_rows:
        p = pair_key(row, use_ids=True)
        for ctx in by_pair.get(p, []):
            cases.append(
                {
                    "split": split_name,
                    "case": "same_pair",
                    "target_drug": row.get("src"),
                    "target_disease": row.get("dst"),
                    "context_rel": ctx.get("rel"),
                    "context_src": ctx.get("src"),
                    "context_dst": ctx.get("dst"),
                }
            )
        for ctx in by_reverse_pair.get(p, []):
            cases.append(
                {
                    "split": split_name,
                    "case": "reverse_pair",
                    "target_drug": row.get("src"),
                    "target_disease": row.get("dst"),
                    "context_rel": ctx.get("rel"),
                    "context_src": ctx.get("src"),
                    "context_dst": ctx.get("dst"),
                }
            )
    return cases


def normalized_duplicate_entities(entity2id: Dict[str, Any], entity_types: Dict[str, str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for entity in entity2id.keys():
        typ = entity_types.get(entity, "")
        norm = normalize_entity_name(entity)
        if norm:
            groups[(typ, norm)].append(entity)

    rows = []
    for (typ, norm), entities in groups.items():
        if len(entities) > 1:
            rows.append(
                {
                    "entity_type": typ,
                    "normalized_name": norm,
                    "count": len(entities),
                    "entities": " | ".join(sorted(entities)[:20]),
                }
            )
    rows.sort(key=lambda r: (-int(r["count"]), r["entity_type"], r["normalized_name"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate split integrity for R-GCN runs.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_duplicate_rows", type=int, default=500)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    splits_dir = run_dir / "splits"
    if not splits_dir.exists():
        raise FileNotFoundError(f"Splits directory not found: {splits_dir}")

    cfg = load_checkpoint_config(run_dir)
    task_cfg = cfg["task"]
    task_name = task_cfg.get("name")
    target_relation = task_cfg["target_relation"]
    context_relations = task_cfg.get("context_relations", [target_relation])

    processed_dir = resolve_processed_dir(cfg["data"]["processed_dir"], task_name=task_name)
    prepared = load_prepared_graph(processed_dir)
    triples: List[Triple] = prepared["triples"]
    entity2id: Dict[str, Any] = prepared["entity2id"]
    entity_types: Dict[str, str] = prepared.get("entity_types", {})

    train_target, valid_target, test_target = load_splits(splits_dir)
    split_meta_path = splits_dir / "split_metadata.json"
    split_metadata = load_json(split_meta_path) if split_meta_path.exists() else {}

    target_rows_full = build_relation_rows(triples, target_relation)
    non_target_triples = [r for r in triples if as_str(r.get("rel")) != target_relation]
    context_edges = build_context_edges(triples, train_target, target_relation, context_relations)

    split_sets = {
        "train": {triple_key(r, use_ids=True) for r in train_target},
        "valid": {triple_key(r, use_ids=True) for r in valid_target},
        "test": {triple_key(r, use_ids=True) for r in test_target},
    }
    pair_sets = {
        "train": {pair_key(r, use_ids=True) for r in train_target},
        "valid": {pair_key(r, use_ids=True) for r in valid_target},
        "test": {pair_key(r, use_ids=True) for r in test_target},
    }

    overlaps = {
        "train_valid_exact": count_intersections(split_sets["train"], split_sets["valid"]),
        "train_test_exact": count_intersections(split_sets["train"], split_sets["test"]),
        "valid_test_exact": count_intersections(split_sets["valid"], split_sets["test"]),
        "train_valid_pair": count_intersections(pair_sets["train"], pair_sets["valid"]),
        "train_test_pair": count_intersections(pair_sets["train"], pair_sets["test"]),
        "valid_test_pair": count_intersections(pair_sets["valid"], pair_sets["test"]),
    }

    train_reverse_pairs = {reverse_pair_key(r, use_ids=True) for r in train_target}
    inverse_leakage = {
        "valid_reverse_pair_in_train": count_intersections(pair_sets["valid"], train_reverse_pairs),
        "test_reverse_pair_in_train": count_intersections(pair_sets["test"], train_reverse_pairs),
    }

    rel_counter_full = Counter(as_str(r.get("rel")) for r in triples)
    target_pair_counter = Counter(pair_key(r, use_ids=True) for r in target_rows_full)
    duplicate_target_pairs = [p for p, c in target_pair_counter.items() if c > 1]

    direct_cases = []
    direct_cases.extend(find_direct_context_same_pair(valid_target, "valid", non_target_triples))
    direct_cases.extend(find_direct_context_same_pair(test_target, "test", non_target_triples))

    drug_to_genes, disease_to_genes = index_gene_context(context_edges)
    drug_to_ingredients = index_ingredients(context_edges)
    disease_to_train_drugs = build_known_drugs_by_disease(train_target)

    valid_context_rows, valid_context_summary = context_path_rows(
        valid_target,
        "valid",
        drug_to_genes,
        disease_to_genes,
        drug_to_ingredients,
        disease_to_train_drugs,
    )
    test_context_rows, test_context_summary = context_path_rows(
        test_target,
        "test",
        drug_to_genes,
        disease_to_genes,
        drug_to_ingredients,
        disease_to_train_drugs,
    )

    duplicate_entities = normalized_duplicate_entities(entity2id, entity_types)

    report = {
        "run_dir": str(run_dir),
        "processed_dir": str(processed_dir),
        "task": task_name,
        "target_relation": target_relation,
        "context_relations": context_relations,
        "split_metadata": split_metadata,
        "counts": {
            "all_triples": len(triples),
            "target_relation_triples_full_graph": len(target_rows_full),
            "non_target_triples_full_graph": len(non_target_triples),
            "train_target": len(train_target),
            "valid_target": len(valid_target),
            "test_target": len(test_target),
            "context_edges_used_for_message_passing": len(context_edges),
            "relations_full_graph": dict(rel_counter_full),
        },
        "overlap_checks": overlaps,
        "inverse_leakage_checks": inverse_leakage,
        "duplicate_target_pair_checks": {
            "num_duplicate_target_pairs_in_full_graph": len(duplicate_target_pairs),
        },
        "direct_context_same_pair_checks": {
            "num_valid_or_test_pairs_with_non_target_relation_same_or_reverse_pair": len(direct_cases),
        },
        "biomedical_context_path_checks": {
            "valid": valid_context_summary,
            "test": test_context_summary,
        },
        "normalized_duplicate_entity_checks": {
            "num_normalized_duplicate_groups": len(duplicate_entities),
            "note": "Heuristic only. Review CSV manually; matching normalized names does not automatically imply leakage.",
        },
    }

    passed = (
        all(v == 0 for v in overlaps.values())
        and all(v == 0 for v in inverse_leakage.values())
        and len(duplicate_target_pairs) == 0
    )
    report["passed_core_integrity_checks"] = bool(passed)

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "split_integrity"
    write_json(output_dir / "integrity_report.json", report)

    field_context = [
        "split", "drug_id", "drug", "disease_id", "disease",
        "shared_gene_count", "drug_gene_count", "disease_gene_count",
        "has_drug_gene_disease_path", "drug_ingredient_count",
        "shared_ingredient_train_drug_count",
        "has_ingredient_support_from_train_indications",
    ]
    write_csv(output_dir / "context_paths_valid.csv", valid_context_rows, field_context)
    write_csv(output_dir / "context_paths_test.csv", test_context_rows, field_context)
    write_csv(
        output_dir / "direct_context_same_pair.csv",
        direct_cases,
        ["split", "case", "target_drug", "target_disease", "context_rel", "context_src", "context_dst"],
    )
    write_csv(
        output_dir / "normalized_duplicate_entities.csv",
        duplicate_entities[: args.max_duplicate_rows],
        ["entity_type", "normalized_name", "count", "entities"],
    )

    print("-" * 70)
    print("Split Integrity Report")
    print("-" * 70)
    print(f"RUN: {run_dir}")
    print(f"TASK: {task_name}")
    print(f"TARGET RELATION: {target_relation}")
    print(f"TRAIN / VALID / TEST: {len(train_target)} / {len(valid_target)} / {len(test_target)}")
    print("-" * 70)
    print("Overlap checks")
    for k, v in overlaps.items():
        print(f"{k}: {v}")
    print("-" * 70)
    print("Inverse leakage checks")
    for k, v in inverse_leakage.items():
        print(f"{k}: {v}")
    print("-" * 70)
    print(f"Duplicate target pairs in full graph: {len(duplicate_target_pairs)}")
    print(f"Direct non-target same/reverse pair cases: {len(direct_cases)}")
    print("-" * 70)
    print("Biomedical context paths")
    print(f"VALID gene path: {valid_context_summary['edges_with_drug_gene_disease_path']} / {valid_context_summary['num_edges']} ({valid_context_summary['pct_with_drug_gene_disease_path']}%)")
    print(f"TEST  gene path: {test_context_summary['edges_with_drug_gene_disease_path']} / {test_context_summary['num_edges']} ({test_context_summary['pct_with_drug_gene_disease_path']}%)")
    print(f"VALID ingredient support: {valid_context_summary['edges_with_ingredient_support_from_train_indications']} / {valid_context_summary['num_edges']} ({valid_context_summary['pct_with_ingredient_support_from_train_indications']}%)")
    print(f"TEST  ingredient support: {test_context_summary['edges_with_ingredient_support_from_train_indications']} / {test_context_summary['num_edges']} ({test_context_summary['pct_with_ingredient_support_from_train_indications']}%)")
    print("-" * 70)
    print(f"Core integrity checks passed: {passed}")
    print(f"Saved report to: {output_dir / 'integrity_report.json'}")
    print("-" * 70)


if __name__ == "__main__":
    main()

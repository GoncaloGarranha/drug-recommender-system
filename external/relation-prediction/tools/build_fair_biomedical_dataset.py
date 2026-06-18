import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

Triple = Tuple[str, str, str]


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def load_split(path: Path) -> List[Triple]:
    rows = load_json(path)
    triples = []
    for row in rows:
        src = str(row["src"]).strip()
        rel = str(row["rel"]).strip()
        dst = str(row["dst"]).strip()
        if src and rel and dst:
            triples.append((src, rel, dst))
    return triples


def load_triples_csv(path: Path) -> List[Triple]:
    triples = []
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            src = str(row["src"]).strip()
            rel = str(row["rel"]).strip()
            dst = str(row["dst"]).strip()
            if src and rel and dst:
                triples.append((src, rel, dst))
    return triples


def write_dict(path: Path, values: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for idx, value in enumerate(values):
            file_obj.write("{}\t{}\n".format(idx, value))


def write_triples(path: Path, triples: Iterable[Triple]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for src, rel, dst in triples:
            file_obj.write("{}\t{}\t{}\n".format(src, rel, dst))


def write_lines(path: Path, values: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as file_obj:
        for value in values:
            file_obj.write("{}\n".format(value))

def resolve_path(path_text: str, must_exist: bool = False) -> Path:
    """Resolve absolute, cwd-relative, or repository-root-relative paths."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    if path.exists() or (not must_exist and path.parent.exists()):
        return path

    for parent in Path.cwd().resolve().parents:
        candidate = parent / path
        if candidate.exists() or (not must_exist and candidate.parent.exists()):
            return candidate

    return path

def relation_counts(triples: Iterable[Triple]) -> Dict[str, int]:
    return dict(Counter(rel for _, rel, _ in triples))


def sorted_unique(triples: Iterable[Triple]) -> List[Triple]:
    return sorted(set(triples))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a RelationPrediction-master biomedical dataset with separate "
            "context graph and target loss triples."
        )
    )
    parser.add_argument("--processed_dir", required=True, help="R-GCN processed graph directory")
    parser.add_argument("--splits_dir", help="Directory containing R-GCN split JSON files")
    parser.add_argument("--out_dir", required=True, help="Output RelationPrediction dataset directory")
    parser.add_argument("--target_relation", default="INDICATION")
    parser.add_argument(
        "--context_relations",
        nargs="+",
        default=["INDICATION", "CONTRAINDICATION", "HAS_INGREDIENT", "ASSOCIATED_WITH_GENE"],
        help="Relations allowed in graph_train.txt. Target valid/test triples are excluded.",
    )
    parser.add_argument("--candidate_type", default="DRUG")
    parser.add_argument("--query_type", default="DISEASE")
    args = parser.parse_args()

    processed_dir = resolve_path(args.processed_dir, must_exist=True)
    splits_dir = resolve_path(args.splits_dir, must_exist=True) if args.splits_dir else processed_dir / "splits"
    out_dir = resolve_path(args.out_dir, must_exist=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_target = sorted_unique(load_split(splits_dir / "train_triples.json"))
    valid_target = sorted_unique(load_split(splits_dir / "valid_triples.json"))
    test_target = sorted_unique(load_split(splits_dir / "test_triples.json"))

    all_processed_triples = load_triples_csv(processed_dir / "triples_indexed.csv")
    entity_types = load_json(processed_dir / "entity_types.json")

    target_relation = args.target_relation
    context_relations = set(args.context_relations)
    valid_test_targets = set(valid_target) | set(test_target)
    train_target_set = set(train_target)

    graph_triples = []
    for triple in all_processed_triples:
        src, rel, dst = triple
        if rel not in context_relations:
            continue
        if rel == target_relation:
            if triple in train_target_set:
                graph_triples.append(triple)
            continue
        if triple in valid_test_targets:
            continue
        graph_triples.append(triple)

    graph_train = sorted_unique(graph_triples)
    loss_train = train_target
    train_alias = loss_train
    valid = valid_target
    test = test_target

    all_dataset_triples = graph_train + loss_train + valid + test
    entities = sorted({src for src, _, _ in all_dataset_triples} | {dst for _, _, dst in all_dataset_triples})
    relations = sorted({rel for _, rel, _ in all_dataset_triples})
    entity_set = set(entities)

    candidate_entities = sorted(
        entity for entity, entity_type in entity_types.items() if entity_type == args.candidate_type and entity in entity_set
    )
    required_candidates = sorted({src for src, rel, _ in loss_train + valid + test if rel == target_relation})
    candidate_entities = sorted(set(candidate_entities) | set(required_candidates))

    query_entities = sorted(
        entity for entity, entity_type in entity_types.items() if entity_type == args.query_type and entity in entity_set
    )
    required_queries = sorted({dst for _, rel, dst in loss_train + valid + test if rel == target_relation})
    query_entities = sorted(set(query_entities) | set(required_queries))

    write_dict(out_dir / "entities.dict", entities)
    write_dict(out_dir / "relations.dict", relations)
    write_triples(out_dir / "graph_train.txt", graph_train)
    write_triples(out_dir / "loss_train.txt", loss_train)
    write_triples(out_dir / "train.txt", train_alias)
    write_triples(out_dir / "valid.txt", valid)
    write_triples(out_dir / "test.txt", test)
    write_lines(out_dir / "candidate_drugs.txt", candidate_entities)
    write_lines(out_dir / "query_diseases.txt", query_entities)

    metadata = {
        "dataset_name": out_dir.name,
        "source_processed_dir": str(processed_dir),
        "source_splits_dir": str(splits_dir),
        "target_relation": target_relation,
        "context_relations": sorted(context_relations),
        "candidate_type": args.candidate_type,
        "query_type": args.query_type,
        "num_entities": len(entities),
        "num_relations": len(relations),
        "num_graph_train": len(graph_train),
        "num_loss_train": len(loss_train),
        "num_valid": len(valid),
        "num_test": len(test),
        "num_candidate_drugs": len(candidate_entities),
        "num_query_diseases": len(query_entities),
        "graph_train_relation_distribution": relation_counts(graph_train),
        "loss_train_relation_distribution": relation_counts(loss_train),
        "valid_relation_distribution": relation_counts(valid),
        "test_relation_distribution": relation_counts(test),
        "relations": relations,
        "notes": (
            "graph_train.txt is used by the GCN encoder; loss_train.txt/train.txt contain only "
            "target training triples for supervised optimization; valid/test contain target triples."
        ),
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=False)

    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

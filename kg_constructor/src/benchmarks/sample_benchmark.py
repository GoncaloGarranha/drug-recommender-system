from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Triple = Tuple[str, str, str]
SPLITS = ("train", "valid", "test")


@dataclass(frozen=True)
class SplitFiles:
    paths: Dict[str, Path]
    suffix: str


def _split_line(line: str, path: Path) -> Triple:
    parts = line.split("\t") if "\t" in line else line.split()
    if len(parts) != 3:
        raise ValueError(
            f"Invalid triple in {path}: expected 3 columns, got {len(parts)} | {line}"
        )
    src, rel, dst = (part.strip() for part in parts)
    if not src or not rel or not dst:
        raise ValueError(
            f"Invalid triple in {path}: empty values are not allowed | {line}"
        )
    return src, rel, dst


def _read_triple_file(path: Path) -> List[Triple]:
    triples: List[Triple] = []
    with path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            triples.append(_split_line(line, path))
    return triples


def _discover_split_files(dataset_path: Path) -> SplitFiles:
    if not dataset_path.is_dir():
        raise ValueError(f"Expected a directory with split files: {dataset_path}")

    for suffix in (".txt", ".tsv"):
        paths = {split: dataset_path / f"{split}{suffix}" for split in SPLITS}
        if all(path.exists() for path in paths.values()):
            return SplitFiles(paths=paths, suffix=suffix)

    expected = ", ".join(f"{split}.txt/.tsv" for split in SPLITS)
    raise FileNotFoundError(
        f"Could not find benchmark split files in {dataset_path}. Expected: {expected}"
    )


def _read_split_dir(dataset_path: Path) -> Tuple[Dict[str, List[Triple]], str]:
    split_files = _discover_split_files(dataset_path)
    return {
        split: _read_triple_file(path) for split, path in split_files.paths.items()
    }, split_files.suffix


def _read_reference_csv(path: Path) -> Dict[str, List[Triple]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"src", "rel", "dst", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Reference CSV must contain columns {sorted(required)}: {path}"
            )
        splits = {split: [] for split in SPLITS}
        for row in reader:
            split = (row.get("split") or "").strip()
            if split in splits:
                splits[split].append(
                    (row["src"].strip(), row["rel"].strip(), row["dst"].strip())
                )
    return splits


def _read_reference(dataset_path: Path) -> Dict[str, List[Triple]]:
    if dataset_path.is_dir():
        splits, _ = _read_split_dir(dataset_path)
        return splits
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".csv":
        return _read_reference_csv(dataset_path)
    raise ValueError(
        "Reference biomedical dataset must be a split directory or a triples.csv file with a split column"
    )


def _counts(splits: Dict[str, Sequence[Triple]]) -> Dict[str, int]:
    return {split: len(splits[split]) for split in SPLITS}


def _target_counts(
    benchmark_counts: Dict[str, int],
    reference_counts: Dict[str, int],
    preserve_proportions: bool,
) -> Dict[str, int]:
    if not preserve_proportions:
        return dict(reference_counts)

    reference_total = sum(reference_counts.values())
    benchmark_total = sum(benchmark_counts.values())
    if benchmark_total == 0:
        raise ValueError("Original benchmark dataset is empty")

    raw_targets = {
        split: reference_total * benchmark_counts[split] / benchmark_total
        for split in SPLITS
    }
    targets = {split: int(raw_targets[split]) for split in SPLITS}
    remainder = reference_total - sum(targets.values())
    for split, _ in sorted(
        raw_targets.items(), key=lambda item: item[1] - int(item[1]), reverse=True
    )[:remainder]:
        targets[split] += 1
    return targets


def _sample(triples: Sequence[Triple], size: int, rng: random.Random) -> List[Triple]:
    if size >= len(triples):
        result = list(triples)
        rng.shuffle(result)
        return result
    return rng.sample(list(triples), size)


def _known_from_train(train_triples: Iterable[Triple]) -> Tuple[set[str], set[str]]:
    entities: set[str] = set()
    relations: set[str] = set()
    for src, rel, dst in train_triples:
        entities.add(src)
        entities.add(dst)
        relations.add(rel)
    return entities, relations


def _filter_known(
    triples: Iterable[Triple], entities: set[str], relations: set[str]
) -> List[Triple]:
    return [
        triple
        for triple in triples
        if triple[0] in entities and triple[2] in entities and triple[1] in relations
    ]


def _write_split(path: Path, triples: Sequence[Triple]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        for src, rel, dst in triples:
            file_obj.write(f"{src}\t{rel}\t{dst}\n")


def sample_benchmark(
    benchmark_path: Path,
    reference_path: Path,
    output_path: Path,
    seed: int,
    preserve_proportions: bool,
) -> Dict[str, object]:
    rng = random.Random(seed)
    benchmark_splits, suffix = _read_split_dir(benchmark_path)
    reference_splits = _read_reference(reference_path)
    original_counts = _counts(benchmark_splits)
    reference_counts = _counts(reference_splits)
    targets = _target_counts(original_counts, reference_counts, preserve_proportions)

    sampled_train = _sample(benchmark_splits["train"], targets["train"], rng)
    train_entities, train_relations = _known_from_train(sampled_train)

    final_splits: Dict[str, List[Triple]] = {"train": sampled_train}
    warnings: List[str] = []
    for split in ("valid", "test"):
        eligible = _filter_known(
            benchmark_splits[split], train_entities, train_relations
        )
        final_splits[split] = _sample(eligible, targets[split], rng)
        if len(eligible) < targets[split]:
            warnings.append(
                f"Only {len(eligible)} eligible {split} triples available after train-vocabulary filtering; "
                f"requested {targets[split]}."
            )

    output_path.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        _write_split(output_path / f"{split}{suffix}", final_splits[split])

    final_entities, final_relations = _known_from_train(final_splits["train"])
    final_counts = _counts(final_splits)
    original_total = sum(original_counts.values())
    final_total = sum(final_counts.values())
    summary: Dict[str, object] = {
        "dataset_original": str(benchmark_path),
        "dataset_biomedical_reference": str(reference_path),
        "seed": seed,
        "preserve_proportions": preserve_proportions,
        "original_triples_by_split": original_counts,
        "reference_triples_by_split": reference_counts,
        "target_triples_by_split": targets,
        "final_triples_by_split": final_counts,
        "num_entities": len(final_entities),
        "num_relations": len(final_relations),
        "percent_kept_vs_original": (
            (final_total / original_total * 100.0) if original_total else 0.0
        ),
        "warnings": warnings,
    }
    with (output_path / "summary.json").open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)
        file_obj.write("\n")

    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sample benchmark KG splits to match a biomedical reference dataset size."
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
        help="Path to original benchmark split directory",
    )
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Path to biomedical reference split directory or triples.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for sampled train/valid/test files",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for deterministic sampling"
    )
    parser.add_argument(
        "--preserve-proportions",
        action="store_true",
        help="Use the original benchmark split proportions with the biomedical total size instead of exact biomedical split sizes",
    )
    args = parser.parse_args()
    sample_benchmark(
        args.benchmark,
        args.reference,
        args.output,
        args.seed,
        args.preserve_proportions,
    )


if __name__ == "__main__":
    main()

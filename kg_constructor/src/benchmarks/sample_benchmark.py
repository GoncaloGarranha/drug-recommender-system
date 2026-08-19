from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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
            f"Invalid triple in {path}: expected 3 columns, "
            f"got {len(parts)} | {line}"
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
        raise ValueError(
            f"Expected a directory with split files: {dataset_path}"
        )

    for suffix in (".txt", ".tsv"):
        paths = {
            split: dataset_path / f"{split}{suffix}"
            for split in SPLITS
        }

        if all(path.exists() for path in paths.values()):
            return SplitFiles(paths=paths, suffix=suffix)

    raise FileNotFoundError(
        f"Could not find train/valid/test .txt or .tsv files in "
        f"{dataset_path}"
    )


def _read_split_dir(
    dataset_path: Path,
) -> Tuple[Dict[str, List[Triple]], str]:
    split_files = _discover_split_files(dataset_path)

    return {
        split: _read_triple_file(path)
        for split, path in split_files.paths.items()
    }, split_files.suffix


def _counts(
    splits: Dict[str, Sequence[Triple]],
) -> Dict[str, int]:
    return {
        split: len(splits[split])
        for split in SPLITS
    }


def _allocate_split_counts(
    total: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
) -> Dict[str, int]:
    ratios = {
        "train": train_ratio,
        "valid": valid_ratio,
        "test": test_ratio,
    }

    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + valid_ratio + test_ratio must equal 1.0"
        )

    raw = {
        split: total * ratio
        for split, ratio in ratios.items()
    }

    result = {
        split: int(value)
        for split, value in raw.items()
    }

    remainder = total - sum(result.values())

    order = sorted(
        SPLITS,
        key=lambda split: raw[split] - result[split],
        reverse=True,
    )

    for split in order[:remainder]:
        result[split] += 1

    return result


def _relation_counts(
    triples: Sequence[Triple],
) -> Counter:
    return Counter(rel for _, rel, _ in triples)


def _allocate_relation_targets(
    triples: Sequence[Triple],
    size: int,
    preserve_all_relations: bool,
) -> Dict[str, int]:
    counts = _relation_counts(triples)

    if size > len(triples):
        raise ValueError(
            f"Requested {size} triples but only {len(triples)} available"
        )

    relations = sorted(counts)

    if preserve_all_relations and size < len(relations):
        raise ValueError(
            f"Cannot preserve all {len(relations)} relations "
            f"with sample size {size}"
        )

    targets = {
        relation: 0
        for relation in relations
    }

    remaining = size

    if preserve_all_relations:
        for relation in relations:
            targets[relation] = 1
            remaining -= 1

    residual_capacity = {
        relation: counts[relation] - targets[relation]
        for relation in relations
    }

    total_capacity = sum(residual_capacity.values())

    if remaining == 0:
        return targets

    raw_extra = {
        relation: (
            remaining
            * residual_capacity[relation]
            / total_capacity
        )
        for relation in relations
    }

    for relation in relations:
        extra = min(
            int(raw_extra[relation]),
            residual_capacity[relation],
        )

        targets[relation] += extra

    still_needed = size - sum(targets.values())

    ranking = sorted(
        relations,
        key=lambda relation: (
            raw_extra[relation] - int(raw_extra[relation]),
            residual_capacity[relation],
        ),
        reverse=True,
    )

    while still_needed > 0:
        changed = False

        for relation in ranking:
            if targets[relation] < counts[relation]:
                targets[relation] += 1
                still_needed -= 1
                changed = True

                if still_needed == 0:
                    break

        if not changed:
            raise RuntimeError(
                "Could not allocate requested stratified sample size"
            )

    return targets


def _stratified_sample(
    triples: Sequence[Triple],
    size: int,
    rng: random.Random,
    preserve_all_relations: bool,
) -> List[Triple]:
    if size == len(triples):
        result = list(triples)
        rng.shuffle(result)
        return result

    targets = _allocate_relation_targets(
        triples=triples,
        size=size,
        preserve_all_relations=preserve_all_relations,
    )

    by_relation: Dict[str, List[Triple]] = defaultdict(list)

    for triple in triples:
        by_relation[triple[1]].append(triple)

    sampled: List[Triple] = []

    for relation in sorted(by_relation):
        relation_triples = by_relation[relation]
        count = targets[relation]

        if count:
            sampled.extend(
                rng.sample(relation_triples, count)
            )

    rng.shuffle(sampled)

    if len(sampled) != size:
        raise RuntimeError(
            f"Sample size mismatch: {len(sampled)} != {size}"
        )

    return sampled


def _known_from_train(
    train_triples: Iterable[Triple],
) -> Tuple[set[str], set[str]]:
    entities: set[str] = set()
    relations: set[str] = set()

    for src, rel, dst in train_triples:
        entities.add(src)
        entities.add(dst)
        relations.add(rel)

    return entities, relations


def _filter_known(
    triples: Iterable[Triple],
    entities: set[str],
    relations: set[str],
) -> List[Triple]:
    return [
        triple
        for triple in triples
        if (
            triple[0] in entities
            and triple[2] in entities
            and triple[1] in relations
        )
    ]


def _write_split(
    path: Path,
    triples: Sequence[Triple],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file_obj:
        for src, rel, dst in triples:
            file_obj.write(
                f"{src}\t{rel}\t{dst}\n"
            )


def _distribution(
    triples: Sequence[Triple],
) -> Dict[str, int]:
    return dict(
        sorted(_relation_counts(triples).items())
    )


def sample_benchmark(
    benchmark_path: Path,
    output_path: Path,
    total_triples: int,
    seed: int,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    preserve_split_proportions: bool,
) -> Dict[str, object]:
    rng = random.Random(seed)

    benchmark_splits, suffix = _read_split_dir(
        benchmark_path
    )

    original_counts = _counts(
        benchmark_splits
    )

    original_total = sum(
        original_counts.values()
    )

    if total_triples > original_total:
        raise ValueError(
            f"Requested {total_triples} triples, "
            f"but benchmark contains only {original_total}"
        )

    if preserve_split_proportions:
        original_ratios = {
            split: original_counts[split] / original_total
            for split in SPLITS
        }

        targets = _allocate_split_counts(
            total=total_triples,
            train_ratio=original_ratios["train"],
            valid_ratio=original_ratios["valid"],
            test_ratio=original_ratios["test"],
        )

        effective_ratios = original_ratios

    else:
        targets = _allocate_split_counts(
            total=total_triples,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
        )

        effective_ratios = {
            "train": train_ratio,
            "valid": valid_ratio,
            "test": test_ratio,
        }

    if targets["train"] > len(benchmark_splits["train"]):
        raise ValueError(
            f"Requested {targets['train']} train triples, "
            f"but benchmark train split only contains "
            f"{len(benchmark_splits['train'])}"
        )

    sampled_train = _stratified_sample(
        triples=benchmark_splits["train"],
        size=targets["train"],
        rng=rng,
        preserve_all_relations=True,
    )

    train_entities, train_relations = (
        _known_from_train(sampled_train)
    )

    final_splits: Dict[str, List[Triple]] = {
        "train": sampled_train
    }

    eligible_counts: Dict[str, int] = {}

    for split in ("valid", "test"):
        eligible = _filter_known(
            benchmark_splits[split],
            entities=train_entities,
            relations=train_relations,
        )

        eligible_counts[split] = len(eligible)

        if len(eligible) < targets[split]:
            raise ValueError(
                f"{split}: requested {targets[split]} triples, "
                f"but only {len(eligible)} remain after "
                f"train-vocabulary filtering"
            )

        final_splits[split] = _stratified_sample(
            triples=eligible,
            size=targets[split],
            rng=rng,
            preserve_all_relations=False,
        )

    final_counts = _counts(final_splits)

    if sum(final_counts.values()) != total_triples:
        raise RuntimeError(
            "Final benchmark does not contain the requested total"
        )

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in SPLITS:
        _write_split(
            output_path / f"{split}{suffix}",
            final_splits[split],
        )

    final_entities, final_relations = (
        _known_from_train(
            final_splits["train"]
        )
    )

    summary = {
        "dataset_original": str(benchmark_path),
        "sampling_strategy": "relation_stratified",
        "seed": seed,
        "requested_total_triples": total_triples,
        "preserve_split_proportions": preserve_split_proportions,
        "split_ratios": effective_ratios,
        "original_triples_by_split": original_counts,
        "target_triples_by_split": targets,
        "final_triples_by_split": final_counts,
        "eligible_after_train_vocabulary_filter": eligible_counts,
        "num_entities_in_train": len(final_entities),
        "num_relations_in_train": len(final_relations),
        "percent_kept_vs_original": (
            total_triples
            / original_total
            * 100.0
        ),
        "relation_distribution_original": {
            split: _distribution(
                benchmark_splits[split]
            )
            for split in SPLITS
        },
        "relation_distribution_sampled": {
            split: _distribution(
                final_splits[split]
            )
            for split in SPLITS
        },
        "constraints": {
            "relation_stratified": True,
            "all_train_relations_preserved": True,
            "valid_test_entities_seen_in_train": True,
            "valid_test_relations_seen_in_train": True,
            "strict_target_sizes": True,
        },
    }

    with (
        output_path / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file_obj:
        json.dump(
            summary,
            file_obj,
            indent=2,
            ensure_ascii=False,
        )
        file_obj.write("\n")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a deterministic relation-stratified "
            "sample of a link-prediction benchmark."
        )
    )

    parser.add_argument(
        "--benchmark",
        required=True,
        type=Path,
        help="Directory containing train/valid/test .txt or .tsv files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory where the sampled splits and summary.json are written.",
    )

    parser.add_argument(
        "--total-triples",
        required=True,
        type=int,
        help="Total number of triples in the sampled benchmark.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling.",
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train ratio when original split proportions are not preserved.",
    )

    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation ratio when original split proportions are not preserved.",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test ratio when original split proportions are not preserved.",
    )

    parser.add_argument(
        "--preserve-split-proportions",
        action="store_true",
        help=(
            "Preserve the original benchmark train/valid/test "
            "proportions instead of using --train-ratio, "
            "--valid-ratio and --test-ratio."
        ),
    )

    args = parser.parse_args()

    summary = sample_benchmark(
        benchmark_path=args.benchmark,
        output_path=args.output,
        total_triples=args.total_triples,
        seed=args.seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        preserve_split_proportions=args.preserve_split_proportions,
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
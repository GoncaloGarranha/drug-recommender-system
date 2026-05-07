from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_constructor.src.utils.log import get_logger
from kg_constructor.src.utils.validate import (
    validate_no_empty_values,
    validate_relation_counts,
    validate_split_values,
    validate_triples_columns,
)

logger = get_logger(__name__)


BenchmarkRow = Tuple[str, str, str, str]


def _read_split_rows(split_path: Path, split_name: str) -> List[BenchmarkRow]:
    """Read triples from a benchmark split file.

    Supports the common benchmark formats where each row is either tab-separated
    or whitespace-separated `src rel dst`.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    rows: List[BenchmarkRow] = []
    with split_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t") if "\t" in line else line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid benchmark row in {split_path}: expected 3 columns, got {len(parts)} | {line}"
                )

            src, rel, dst = (part.strip() for part in parts)
            if not src or not rel or not dst:
                raise ValueError(f"Empty value found in benchmark row: {line}")

            rows.append((src, rel, dst, split_name))

    return rows


def _resolve_benchmark_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path, str]:
    """Resolve benchmark input/output paths from the benchmark config."""
    dataset_cfg = cfg.get("dataset", {})
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})

    if not isinstance(dataset_cfg, dict) or not isinstance(input_cfg, dict) or not isinstance(output_cfg, dict):
        raise ValueError("Invalid benchmark config structure. Expected dataset/input/output sections.")

    benchmark_name = str(dataset_cfg.get("name", "benchmark")).strip() or "benchmark"

    raw_dir = input_cfg.get("raw_dir")
    triples_path = output_cfg.get("triples_path")

    if not raw_dir:
        raise ValueError("Missing input.raw_dir in benchmark config.")
    if not triples_path:
        raise ValueError("Missing output.triples_path in benchmark config.")

    input_dir = Path(raw_dir)
    output_path = Path(triples_path)

    return input_dir, output_path, benchmark_name


def convert_benchmark(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """Convert WN18RR / FB15k-237 style benchmark splits into a canonical CSV.

    Expected raw input:
        train.txt
        valid.txt
        test.txt

    Output:
        triples.csv with columns: src, rel, dst, split
    """
    input_dir, triples_csv, benchmark_name = _resolve_benchmark_paths(cfg)
    triples_csv.parent.mkdir(parents=True, exist_ok=True)

    if not force and triples_csv.exists():
        logger.info("Benchmark triples already exist. Skipping step (use --force to re-run).")
        return

    split_files = {
        "train": input_dir / "train.txt",
        "valid": input_dir / "valid.txt",
        "test": input_dir / "test.txt",
    }

    for split_name, split_path in split_files.items():
        if not split_path.exists():
            raise FileNotFoundError(f"Missing {split_name} split file: {split_path}")

    all_rows: List[BenchmarkRow] = []
    for split_name, split_path in split_files.items():
        split_rows = _read_split_rows(split_path, split_name)
        all_rows.extend(split_rows)

    split_counts: Dict[str, int] = {}
    entities = set()
    relations = set()

    for src, rel, dst, split_name in all_rows:
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        entities.add(src)
        entities.add(dst)
        relations.add(rel)

    with triples_csv.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["src", "rel", "dst", "split"])
        writer.writerows(all_rows)

    dict_rows = [
        {"src": src, "rel": rel, "dst": dst, "split": split_name}
        for src, rel, dst, split_name in all_rows
    ]
    validate_triples_columns(dict_rows, ["src", "rel", "dst", "split"])
    validate_no_empty_values(dict_rows, ["src", "rel", "dst", "split"])
    validate_split_values(dict_rows)
    relation_counts = validate_relation_counts(dict_rows)

    logger.info("Saved benchmark triples: %s", triples_csv)
    logger.info(
        "Benchmark '%s' converted | input=%s | triples=%d | entities=%d | relations=%d | splits=%s",
        benchmark_name,
        input_dir,
        len(all_rows),
        len(entities),
        len(relations),
        split_counts,
    )
    logger.info("Benchmark relation distribution: %s", dict(relation_counts))


run = convert_benchmark


if __name__ == "__main__":
    import argparse
    from kg_constructor.src.utils.config import load_config

    parser = argparse.ArgumentParser(description="Convert benchmark splits to canonical triples.csv")
    parser.add_argument("--config", required=True, help="Path to benchmark config YAML")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    cfg = load_config(args.config)
    convert_benchmark(cfg=cfg, paths=None, force=args.force)

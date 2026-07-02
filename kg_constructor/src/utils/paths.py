from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KGOutputPaths:
    root: Path
    interim: Path
    mappings: Path
    triples: Path
    reports: Path
    benchmarks: Path
    processed: Path


def build_output_paths(output_dir: str | Path) -> KGOutputPaths:
    root = Path(output_dir)

    return KGOutputPaths(
        root=root,
        interim=root / "interim",
        mappings=root / "mappings",
        triples=root / "triples",
        reports=root / "reports",
        benchmarks=root / "benchmarks",
        processed=root / "processed",
    )


def resolve_relative_path(base_dir: str | Path, path_value: str | Path) -> Path:
    base_path = Path(base_dir)
    candidate = Path(path_value)

    if candidate.is_absolute():
        return candidate

    return base_path / candidate
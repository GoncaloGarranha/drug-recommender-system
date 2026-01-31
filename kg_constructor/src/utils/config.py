from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

@dataclass(frozen=True)
class KGPaths:
    kg_root: Path
    input_dir: Path
    output_dir: Path
    processed_dir: Path


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config não encontrado: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML inválido (não é um dicionário).")

    return cfg


def build_paths(cfg: Dict[str, Any], project_root: str | Path) -> KGPaths:
    """
    project_root: pasta a partir da qual resolves os paths (normalmente o root do repo).
    """
    root = Path(project_root).resolve()
    p = cfg.get("paths", {})

    kg_root = (root / p["kg_root"]).resolve()
    input_dir = (root / p["input_dir"]).resolve()
    output_dir = (root / p["output_dir"]).resolve()
    processed_dir = (root / p["processed_dir"]).resolve()

    return KGPaths(
        kg_root=kg_root,
        input_dir=input_dir,
        output_dir=output_dir,
        processed_dir=processed_dir,
    )
from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 06_split_edges: split filtered triples into train/valid/test edge sets.

    Input:
      - data/output/triples/triples.filtered.csv

    Output:
      - data/processed/edge_splits.json

    Policy:
      - Split only prediction edges (INDICATION, CONTRAINDICATION).
      - Keep structural edges (HAS_INGREDIENT) in the training graph only.
    """
    output_dir = Path(paths.output_dir)
    processed_dir = Path(paths.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    triples_path = output_dir / "triples" / "triples.filtered.csv"
    splits_path = processed_dir / "edge_splits.json"
    preview_path = processed_dir / "edge_splits_preview.json"

    if not force and splits_path.exists() and preview_path.exists():
        logger.info("Edge splits already exist. Skipping (use --force to re-run).")
        return

    if not triples_path.exists():
        raise FileNotFoundError("Missing triples.filtered.csv. Run 05_filter_graph first.")

    triples = _read_csv(triples_path)

    seed = int(cfg.get("project", {}).get("seed", 42))
    rnd = random.Random(seed)

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.8))
    valid_ratio = float(split_cfg.get("valid", 0.1))
    test_ratio = float(split_cfg.get("test", 0.1))

    if abs((train_ratio + valid_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0 (train + valid + test).")

    prediction_rels = set(cfg.get("split", {}).get("prediction_relations", ["INDICATION", "CONTRAINDICATION"]))
    structural_rels = set(cfg.get("split", {}).get("structural_relations", ["HAS_INGREDIENT"]))

    # Separate edges by relation type
    prediction_edges: List[Tuple[str, str, str]] = []
    structural_edges: List[Tuple[str, str, str]] = []

    rel_counter = Counter()

    for r in triples:
        src = (r.get("src") or "").strip()
        rel = (r.get("rel") or "").strip()
        dst = (r.get("dst") or "").strip()
        if not src or not rel or not dst:
            continue

        rel_counter[rel] += 1

        t = (src, rel, dst)
        if rel in prediction_rels:
            prediction_edges.append(t)
        elif rel in structural_rels:
            structural_edges.append(t)
        else:
            # Default behavior: treat unknown relations as prediction edges (conservative)
            prediction_edges.append(t)

    rnd.shuffle(prediction_edges)

    n = len(prediction_edges)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    # Put the remainder into test to guarantee sum == n
    n_test = n - n_train - n_valid

    train_edges = prediction_edges[:n_train]
    valid_edges = prediction_edges[n_train : n_train + n_valid]
    test_edges = prediction_edges[n_train + n_valid :]

    # Training graph edges include:
    # - all structural edges
    # - train prediction edges
    train_graph_edges = structural_edges + train_edges

    splits = {
        "seed": seed,
        "ratios": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "prediction_relations": sorted(list(prediction_rels)),
        "structural_relations": sorted(list(structural_rels)),
        "counts": {
            "all_triples": len(triples),
            "prediction_edges": len(prediction_edges),
            "structural_edges": len(structural_edges),
            "train_graph_edges": len(train_graph_edges),
            "train": len(train_edges),
            "valid": len(valid_edges),
            "test": len(test_edges),
        },
        # Store edges as lists so JSON is simple and portable
        "train_graph_edges": [list(t) for t in train_graph_edges],
        "valid_edges": [list(t) for t in valid_edges],
        "test_edges": [list(t) for t in test_edges],
    }

    splits_path.write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {splits_path}")

    preview = {
        "relation_counts_in_filtered_triples": dict(rel_counter),
        "counts": splits["counts"],
        "note": "train_graph_edges = structural_edges + train prediction edges",
    }
    preview_path.write_text(json.dumps(preview, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {preview_path}")

    logger.info(
        f"Split sizes | train={len(train_edges)} valid={len(valid_edges)} test={len(test_edges)} "
        f"| train_graph_edges={len(train_graph_edges)}"
    )

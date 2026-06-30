from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from kg_constructor.src.utils.io import read_csv
from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def jaccard(left: Set[str], right: Set[str]) -> float:
    """Compute Jaccard similarity between two feature sets."""
    if not left and not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_similarity_groups_from_triples(
    triples: Iterable[Dict[str, str]],
    source_relation: str = "HAS_INGREDIENT",
    jaccard_threshold: float = 0.5,
    min_shared_features: int = 1,
) -> Dict[str, List[str]]:
    """Build drug similarity groups from shared graph features."""
    drug_features: Dict[str, Set[str]] = defaultdict(set)

    for row in triples:
        src = (row.get("src") or "").strip()
        rel = (row.get("rel") or "").strip()
        dst = (row.get("dst") or "").strip()

        if src and dst and rel == source_relation:
            drug_features[src].add(dst)

    drugs = sorted(drug_features)
    groups: Dict[str, Set[str]] = {drug: {drug} for drug in drugs}

    for index, left_drug in enumerate(drugs):
        left_features = drug_features[left_drug]
        for right_drug in drugs[index + 1:]:
            right_features = drug_features[right_drug]
            shared_features = left_features & right_features

            if len(shared_features) < min_shared_features:
                continue

            if jaccard(left_features, right_features) >= jaccard_threshold:
                groups[left_drug].add(right_drug)
                groups[right_drug].add(left_drug)

    return {drug: sorted(members) for drug, members in groups.items()}


def _group_size_summary(groups: Dict[str, List[str]]) -> Dict[str, Any]:
    sizes = [len(members) for members in groups.values()]
    non_singleton = [size for size in sizes if size > 1]
    return {
        "num_groups": len(groups),
        "non_singleton_groups": len(non_singleton),
        "min_group_size": min(sizes) if sizes else 0,
        "max_group_size": max(sizes) if sizes else 0,
        "mean_group_size": (sum(sizes) / len(sizes)) if sizes else 0.0,
    }


def build_drug_similarity_groups(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """Step 07: build drug similarity groups from KG triples."""
    similarity_cfg = cfg.get("drug_similarity", {})

    if not bool(similarity_cfg.get("enabled", True)):
        logger.info("Drug similarity group generation disabled by config. Skipping step.")
        return

    output_dir = Path(paths.output_dir)
    processed_dir = Path(paths.processed_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    source_relation = str(similarity_cfg.get("source_relation", "HAS_INGREDIENT"))
    jaccard_threshold = float(similarity_cfg.get("jaccard_threshold", 0.5))
    min_shared_features = int(similarity_cfg.get("min_shared_features", 1))

    input_path = Path(similarity_cfg.get("input_path", output_dir / "triples" / "triples.filtered.csv"))
    if not input_path.is_absolute():
        input_path = Path.cwd() / input_path

    output_path = Path(similarity_cfg.get("output_path", processed_dir / "drug_similarity_groups.json"))
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path

    report_path = Path(similarity_cfg.get("report_path", reports_dir / "drug_similarity_groups_report.json"))
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path

    if not force and output_path.exists() and report_path.exists():
        logger.info("Drug similarity group outputs already exist. Skipping step (use --force to re-run).")
        return

    if not input_path.exists():
        raise FileNotFoundError(f"Missing triples file for drug similarity groups: {input_path}")

    triples = read_csv(input_path)
    groups = build_similarity_groups_from_triples(
        triples=triples,
        source_relation=source_relation,
        jaccard_threshold=jaccard_threshold,
        min_shared_features=min_shared_features,
    )

    summary = _group_size_summary(groups)
    artifact = {
        "metadata": {
            "source_triples": str(input_path),
            "source_relation": source_relation,
            "jaccard_threshold": jaccard_threshold,
            "min_shared_features": min_shared_features,
            **summary,
        },
        "groups": groups,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")

    report = {
        **summary,
        "output_path": str(output_path),
        "source_triples": str(input_path),
        "source_relation": source_relation,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("Saved drug similarity groups: %s", output_path)
    logger.info("Saved drug similarity group report: %s", report_path)


run = build_drug_similarity_groups

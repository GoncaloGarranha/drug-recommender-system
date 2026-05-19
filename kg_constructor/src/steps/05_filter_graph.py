from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set

from kg_constructor.src.utils.log import get_logger
from kg_constructor.src.utils.validate import (
    validate_no_empty_values,
    validate_relation_counts,
    validate_triples_columns,
)

logger = get_logger(__name__)


def read_csv(path: Path) -> List[Dict[str, str]]:
    """
    Read a CSV file into a list of dictionaries.

    Args:
        path: Path to the CSV file.

    Returns:
        List of rows as dictionaries.
    """
    with path.open("r", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    """
    Write rows to a CSV file.

    Args:
        path: Output CSV path.
        header: CSV header row.
        rows: Data rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)


def load_blacklist(blacklist_path: Path) -> Set[str]:
    """
    Load blacklist entries from a text file.

    Empty lines and inline comments are ignored.

    Args:
        blacklist_path: Path to the blacklist file.

    Returns:
        Set of blacklist entries.
    """
    if not blacklist_path.exists():
        logger.warning("Blacklist file not found: %s", blacklist_path)
        return set()

    blacklist: Set[str] = set()

    with blacklist_path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            cleaned_line = line.split("#")[0].strip()
            if cleaned_line:
                blacklist.add(cleaned_line)

    logger.info("Loaded %d entries from blacklist: %s", len(blacklist), blacklist_path)
    return blacklist


def filter_graph(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 05: Apply quality filters to canonical triples.

    Input:
        - data/output/triples/triples.csv

    Outputs:
        - data/output/triples/triples.filtered.csv
        - data/output/reports/filter_report.json

    Filters applied:
        1. Remove blacklisted diseases (e.g. generic hub diseases)
        2. Remove self-loops (src == dst)
        3. Remove obsolete DOID labels
        4. Remove HAS_INGREDIENT relations

    This step keeps only task-relevant and higher-quality triples for the
    final graph construction phase.

    Args:
        cfg: Pipeline configuration dictionary.
        paths: Object containing pipeline paths. Must provide `output_dir`.
        force: If True, overwrite existing outputs.
    """
    output_dir = Path(paths.output_dir)
    triples_dir = output_dir / "triples"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    triples_csv_in = triples_dir / "triples.csv"
    triples_csv_out = triples_dir / "triples.filtered.csv"
    report_json = reports_dir / "filter_report.json"

    if not force and triples_csv_out.exists() and report_json.exists():
        logger.info("Filtered outputs already exist. Skipping step (use --force to re-run).")
        return

    if not triples_csv_in.exists():
        raise FileNotFoundError("Missing triples.csv. Run step 04_build_triples first.")

    triple_rows = read_csv(triples_csv_in)
    validate_triples_columns(triple_rows, ["src", "rel", "dst"])
    validate_no_empty_values(triple_rows, ["src", "rel", "dst"])
    input_relation_counts = validate_relation_counts(triple_rows)

    config_dir = Path(__file__).parent.parent.parent / "configs"
    blacklist_path = config_dir / "blacklist.txt"
    blacklist = load_blacklist(blacklist_path)

    filters_cfg = cfg.get("filters", {})

    remove_obsolete_doid_raw = filters_cfg.get("remove_obsolete_doid_labels")
    if remove_obsolete_doid_raw is None:
        remove_obsolete_doid = True
        logger.warning(
            "Filter 'remove_obsolete_doid_labels' not found in config — using default: True"
        )
    else:
        remove_obsolete_doid = bool(remove_obsolete_doid_raw)
        logger.info("Filter 'remove_obsolete_doid_labels' set from config: %s", remove_obsolete_doid)

    remove_has_ingredient_raw = filters_cfg.get("remove_has_ingredient")
    if remove_has_ingredient_raw is None:
        remove_has_ingredient = True
        logger.warning(
            "Filter 'remove_has_ingredient' not found in config — using default: True"
        )
    else:
        remove_has_ingredient = bool(remove_has_ingredient_raw)
        logger.info("Filter 'remove_has_ingredient' set from config: %s", remove_has_ingredient)

    skip_reasons = Counter()
    filtered_rows: List[List[str]] = []

    for row in triple_rows:
        src = (row.get("src") or "").strip()
        rel = (row.get("rel") or "").strip()
        dst = (row.get("dst") or "").strip()

        if not src or not rel or not dst:
            skip_reasons["empty_fields"] += 1
            continue

        if src == dst:
            skip_reasons["self_loops"] += 1
            continue

        if remove_has_ingredient and rel == "HAS_INGREDIENT":
            skip_reasons["has_ingredient_removed"] += 1
            continue

        if dst in blacklist:
            skip_reasons["blacklisted_disease"] += 1
            continue

        if remove_obsolete_doid and dst.upper().endswith("OBSOLETE"):
            skip_reasons["obsolete_doid"] += 1
            continue

        filtered_rows.append([src, rel, dst])

    filtered_dict_rows = [
        {"src": src, "rel": rel, "dst": dst}
        for src, rel, dst in filtered_rows
    ]
    validate_triples_columns(filtered_dict_rows, ["src", "rel", "dst"])
    validate_no_empty_values(filtered_dict_rows, ["src", "rel", "dst"])
    filtered_relation_counts = validate_relation_counts(filtered_dict_rows)

    write_csv(triples_csv_out, ["src", "rel", "dst"], filtered_rows)
    logger.info("Saved: %s", triples_csv_out)
    logger.info(
        "Kept triples: %d | Removed triples: %d",
        len(filtered_rows),
        sum(skip_reasons.values()),
    )
    logger.info("Input relation distribution: %s", dict(input_relation_counts))
    logger.info("Filtered relation distribution: %s", dict(filtered_relation_counts))

    report = {
        "input_triples": len(triple_rows),
        "output_triples": len(filtered_rows),
        "removed": dict(skip_reasons),
        "blacklist_size": len(blacklist),
        "filters": {
            "remove_obsolete_doid_labels": remove_obsolete_doid,
            "remove_has_ingredient": remove_has_ingredient,
        },
    }

    report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved: %s", report_json)


# Alias used by the pipeline runner
run = filter_graph


if __name__ == "__main__":
    from kg_constructor.src.utils.config import load_config

    class Paths:
        def __init__(self, root: Path):
            self.root = root
            self.output_dir = root / "kg_constructor" / "data" / "output"

    root = Path(__file__).parent.parent.parent.parent
    cfg = load_config(root / "kg_constructor" / "configs" / "config.yaml")
    paths = Paths(root)

    filter_graph(cfg, paths, force=True)
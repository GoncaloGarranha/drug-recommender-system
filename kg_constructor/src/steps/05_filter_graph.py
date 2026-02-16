from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def load_blacklist(blacklist_path: Path) -> set:
    """Load blacklist from text file, ignoring comments and empty lines."""
    if not blacklist_path.exists():
        logger.warning(f"Blacklist file not found: {blacklist_path}")
        return set()
    
    blacklist = set()
    with blacklist_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.split("#")[0].strip()
            if line:
                blacklist.add(line)
    
    logger.info(f"Loaded {len(blacklist)} entries from blacklist: {blacklist_path}")
    return blacklist


def filter_graph(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 05: Apply cleaning filters to canonical triples.
    Inputs:
      - data/output/triples/triples.csv
    Outputs:
      - data/output/triples/triples.filtered.csv
      - data/output/reports/filter_report.json
    
    Filters applied:
      1. Remove blacklisted diseases (generic hubs like DOID:4)
      2. Remove self-loops (src == dst)
      3. Remove obsolete DOID labels
      4. Remove HAS_INGREDIENT relations (focus on predictive relations only)
    
    This step ensures only high-quality, task-relevant triples remain.
    """
    output_dir = Path(paths.output_dir)
    triples_dir = output_dir / "triples"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    triples_csv_in = triples_dir / "triples.csv"
    triples_csv_out = triples_dir / "triples.filtered.csv"
    report_json = reports_dir / "filter_report.json"

    if not force and triples_csv_out.exists() and report_json.exists():
        logger.info("Filtered outputs already exist. Skipping (use --force to re-run).")
        return

    if not triples_csv_in.exists():
        raise FileNotFoundError("Missing triples.csv. Run 04_build_triples first.")

    triples = read_csv(triples_csv_in)

    config_dir = Path(__file__).parent.parent.parent / "configs"
    blacklist_path = config_dir / "blacklist.txt"
    blacklist = load_blacklist(blacklist_path)
    
    remove_obsolete = bool(cfg.get("filters", {}).get("remove_obsolete_doid_labels", True))
    remove_ingredients = bool(cfg.get("filters", {}).get("remove_has_ingredient", True))

    skip_reasons = Counter()
    kept_rows: List[List[str]] = []

    for row in triples:
        src = (row.get("src") or "").strip()
        rel = (row.get("rel") or "").strip()
        dst = (row.get("dst") or "").strip()

        if not src or not rel or not dst:
            skip_reasons["empty_fields"] += 1
            continue

        if src == dst:
            skip_reasons["self_loops"] += 1
            continue

        # Remove HAS_INGREDIENT relations (focus on predictive relations only)
        if remove_ingredients and rel == "HAS_INGREDIENT":
            skip_reasons["has_ingredient_removed"] += 1
            continue

        if dst in blacklist:
            skip_reasons["blacklisted_disease"] += 1
            continue

        if remove_obsolete and dst.upper().endswith("OBSOLETE"):
            skip_reasons["obsolete_doid"] += 1
            continue

        kept_rows.append([src, rel, dst])

    write_csv(triples_csv_out, ["src", "rel", "dst"], kept_rows)
    logger.info(f"Saved: {triples_csv_out}")
    logger.info(f"Kept triples: {len(kept_rows)} | Removed: {sum(skip_reasons.values())}")

    report = {
        "input_triples": len(triples),
        "output_triples": len(kept_rows),
        "removed": dict(skip_reasons),
        "blacklist_size": len(blacklist),
    }
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {report_json}")


run = filter_graph


if __name__ == "__main__":
    from kg_constructor.src.utils.config import load_config

    class Paths:
        def __init__(self, root: Path):
            self.root = root
            self.output_dir = root / "kg_constructor" / "data" / "output"

    root = Path(__file__).parent.parent.parent.parent
    config = load_config(root / "kg_constructor" / "configs" / "config.yaml")
    paths = Paths(root)
    
    filter_graph(config, paths, force=True)

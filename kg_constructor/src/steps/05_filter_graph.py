from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, header: List[str], rows: List[List[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 05_filter_graph: apply basic cleaning filters to canonical triples.

    Inputs:
      - data/output/triples/triples.csv

    Outputs:
      - data/output/triples/triples.filtered.csv
      - data/output/reports/filter_report.json
    """
    output_dir = Path(paths.output_dir)
    triples_dir = output_dir / "triples"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    triples_in = triples_dir / "triples.csv"
    triples_out = triples_dir / "triples.filtered.csv"
    report_path = reports_dir / "filter_report.json"

    if not force and triples_out.exists() and report_path.exists():
        logger.info("Filtered outputs already exist. Skipping (use --force to re-run).")
        return

    if not triples_in.exists():
        raise FileNotFoundError("Missing triples.csv. Run 04_build_triples first.")

    triples = _read_csv(triples_in)

    # Blacklist of generic disease IDs (config-driven)
    generic_blacklist = set(cfg.get("filters", {}).get("generic_doid_blacklist", ["DOID:4"]))
    remove_obsolete = bool(cfg.get("filters", {}).get("remove_obsolete_doid_labels", True))

    removed = Counter()
    kept_rows: List[List[str]] = []

    for r in triples:
        src = (r.get("src") or "").strip()
        rel = (r.get("rel") or "").strip()
        dst = (r.get("dst") or "").strip()

        if not src or not rel or not dst:
            removed["empty_fields"] += 1
            continue

        if src == dst:
            removed["self_loops"] += 1
            continue

        # Remove generic diseases (only for disease nodes)
        if dst in generic_blacklist:
            removed["generic_doid_blacklist"] += 1
            continue

        # Optional: remove obsolete DOID labels if your pipeline ever includes such flags
        # (Kept here for completeness; actual obsolete detection usually needs a DOID label map.)
        if remove_obsolete and dst.upper().endswith("OBSOLETE"):
            removed["obsolete_doid"] += 1
            continue

        kept_rows.append([src, rel, dst])

    _write_csv(triples_out, ["src", "rel", "dst"], kept_rows)
    logger.info(f"Saved: {triples_out}")
    logger.info(f"Kept triples: {len(kept_rows)} | Removed: {sum(removed.values())}")

    report = {
        "input_triples": len(triples),
        "output_triples": len(kept_rows),
        "removed": dict(removed),
        "generic_doid_blacklist": sorted(list(generic_blacklist)),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved: {report_path}")

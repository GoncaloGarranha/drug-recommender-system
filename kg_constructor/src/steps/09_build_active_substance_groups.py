from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from kg_constructor.src.utils.io import read_csv_dicts, write_csv_dicts
from kg_constructor.src.utils.paths import build_output_paths, resolve_relative_path


def build_groups(input_csv: Path, output_csv: Path, summary_json: Path) -> None:
    raw_rows = read_csv_dicts(input_csv)

    output_rows: List[Dict[str, str]] = []
    seen = set()
    summary = Counter()

    for row in raw_rows:
        drug_reference_id = row.get("drug_reference_id", "").strip()
        drug_name = row.get("drug_name", "").strip()
        group_id = row.get("active_ingredient_id", "").strip()
        group_name = row.get("active_ingredient_name", "").strip()
        confidence = row.get("confidence", "").strip()
        mapping_method = row.get("mapping_method", "").strip()
        source_file = row.get("file", "").strip()

        if not drug_name or not group_name:
            summary["skipped_missing_name"] += 1
            continue

        if not drug_reference_id:
            drug_reference_id = f"DRUG_NAME:{drug_name.lower()}"

        if not group_id:
            group_id = f"ACTIVE_SUBSTANCE_NAME:{group_name.lower()}"

        key = (source_file, drug_name, group_id)

        if key in seen:
            summary["duplicates_skipped"] += 1
            continue

        seen.add(key)

        output_rows.append(
            {
                "drug_reference_id": drug_reference_id,
                "drug_name": drug_name,
                "active_substance_group_id": group_id,
                "active_substance_group_name": group_name,
                "confidence": confidence,
                "mapping_method": mapping_method,
                "source_file": source_file,
            }
        )

        summary[f"confidence:{confidence}"] += 1
        summary[f"method:{mapping_method}"] += 1

    fieldnames = [
        "drug_reference_id",
        "drug_name",
        "active_substance_group_id",
        "active_substance_group_name",
        "confidence",
        "mapping_method",
        "source_file",
    ]

    write_csv_dicts(output_csv, output_rows, fieldnames)

    drug_count = len({(row["source_file"], row["drug_name"]) for row in output_rows})
    group_count = len({row["active_substance_group_id"] for row in output_rows})

    summary_payload = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_input": len(raw_rows),
        "rows_output": len(output_rows),
        "unique_drugs": drug_count,
        "unique_active_substance_groups": group_count,
        "counts": dict(summary),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved active substance groups: {output_csv}")
    print(f"Saved summary: {summary_json}")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


def run(cfg: Dict[str, object], paths, force: bool = False) -> None:
    step_cfg = cfg.get("active_substance_groups", {}) if isinstance(cfg, dict) else {}
    out = build_output_paths(paths.output_dir)

    input_csv = resolve_relative_path(out.root, step_cfg.get("input_csv", "interim/active_ingredients_raw.csv"))
    output_csv = resolve_relative_path(out.root, step_cfg.get("output_csv", "interim/drug_active_substance_groups.csv"))
    summary_json = resolve_relative_path(out.root, step_cfg.get("summary_json", "interim/drug_active_substance_groups_summary.json"))

    if not force and output_csv.exists() and summary_json.exists():
        return

    build_groups(input_csv=input_csv, output_csv=output_csv, summary_json=summary_json)

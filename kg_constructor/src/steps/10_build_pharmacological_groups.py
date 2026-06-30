from __future__ import annotations

import argparse
import csv
import json
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set


DEFAULT_INPUT_CSV = Path("kg_constructor/data/output/interim/drug_active_substance_groups.csv")
DEFAULT_OUTPUT_CSV = Path("kg_constructor/data/output/interim/drug_pharmacological_groups.csv")
DEFAULT_SUMMARY_JSON = Path("kg_constructor/data/output/interim/drug_pharmacological_groups_summary.json")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_drug_key(row: Dict[str, str]) -> str:
    source_file = row.get("source_file", "").strip()
    drug_name = row.get("drug_name", "").strip()
    drug_reference_id = row.get("drug_reference_id", "").strip()

    if source_file and drug_name:
        return f"{source_file}::{drug_name}"

    if drug_reference_id and drug_name:
        return f"{drug_reference_id}::{drug_name}"

    return drug_name or drug_reference_id


def make_group_id(active_substance_ids: Set[str]) -> str:
    joined = "|".join(sorted(active_substance_ids))
    digest = hashlib.md5(joined.encode("utf-8")).hexdigest()[:10]
    return f"PHARM_GROUP_{digest}"


def build_pharmacological_groups(
    input_csv: Path,
    output_csv: Path,
    summary_json: Path,
) -> None:
    rows = read_csv(input_csv)

    drug_meta: Dict[str, Dict[str, str]] = {}
    drug_to_substances: Dict[str, Set[str]] = defaultdict(set)
    drug_to_substance_names: Dict[str, Dict[str, str]] = defaultdict(dict)
    drug_to_confidences: Dict[str, Set[str]] = defaultdict(set)

    summary = Counter()

    for row in rows:
        drug_key = build_drug_key(row)
        group_id = row.get("active_substance_group_id", "").strip()
        group_name = row.get("active_substance_group_name", "").strip()
        confidence = row.get("confidence", "").strip()

        if not drug_key or not group_id:
            summary["skipped_missing_drug_or_substance"] += 1
            continue

        drug_meta[drug_key] = {
            "drug_reference_id": row.get("drug_reference_id", "").strip(),
            "drug_name": row.get("drug_name", "").strip(),
            "source_file": row.get("source_file", "").strip(),
        }

        drug_to_substances[drug_key].add(group_id)
        drug_to_substance_names[drug_key][group_id] = group_name
        drug_to_confidences[drug_key].add(confidence)

    output_rows: List[Dict[str, str]] = []

    for drug_key, substance_ids in sorted(drug_to_substances.items()):
        pharmacological_group_id = make_group_id(substance_ids)

        substance_names = [
            drug_to_substance_names[drug_key].get(substance_id, substance_id)
            for substance_id in sorted(substance_ids)
        ]

        confidences = drug_to_confidences[drug_key]

        if confidences == {"high"}:
            group_confidence = "high"
        elif "medium" in confidences:
            group_confidence = "medium"
        else:
            group_confidence = "unknown"

        meta = drug_meta[drug_key]

        output_rows.append(
            {
                "drug_key": drug_key,
                "drug_reference_id": meta["drug_reference_id"],
                "drug_name": meta["drug_name"],
                "source_file": meta["source_file"],
                "pharmacological_group_id": pharmacological_group_id,
                "active_substance_ids": "|".join(sorted(substance_ids)),
                "active_substance_names": "|".join(substance_names),
                "num_active_substances": str(len(substance_ids)),
                "group_confidence": group_confidence,
            }
        )

        summary[f"group_confidence:{group_confidence}"] += 1
        summary[f"num_active_substances:{len(substance_ids)}"] += 1

    fieldnames = [
        "drug_key",
        "drug_reference_id",
        "drug_name",
        "source_file",
        "pharmacological_group_id",
        "active_substance_ids",
        "active_substance_names",
        "num_active_substances",
        "group_confidence",
    ]

    write_csv(output_csv, output_rows, fieldnames)

    unique_groups = len({row["pharmacological_group_id"] for row in output_rows})

    summary_payload = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows_input": len(rows),
        "rows_output": len(output_rows),
        "unique_drugs": len(output_rows),
        "unique_pharmacological_groups": unique_groups,
        "counts": dict(summary),
    }

    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved pharmacological groups: {output_csv}")
    print(f"Saved summary: {summary_json}")
    print(json.dumps(summary_payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build pharmacological groups from active substance memberships."
    )

    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary_json", type=Path, default=DEFAULT_SUMMARY_JSON)

    args = parser.parse_args()

    build_pharmacological_groups(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        summary_json=args.summary_json,
    )


if __name__ == "__main__":
    main()
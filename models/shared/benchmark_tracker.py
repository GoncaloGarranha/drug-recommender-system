from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


def log_experiment(
    model_name: str,
    hits1: float,
    hits3: float,
    hits10: float,
    mrr: float,
    notes: str = "",
    dataset: str = "",
) -> None:
    """Append a benchmark row to `benchmarks.csv` at the repository root.

    This is intentionally lightweight: it creates the CSV if missing and appends
    a single row describing the run.
    """
    repo_root = Path(__file__).resolve().parents[2]
    out_path = repo_root / "benchmarks.csv"

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "dataset": dataset,
        "model_name": model_name,
        "hits@1": f"{hits1:.6f}",
        "hits@3": f"{hits3:.6f}",
        "hits@10": f"{hits10:.6f}",
        "mrr": f"{mrr:.6f}",
        "notes": notes,
    }

    write_header = not out_path.exists()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"Logged benchmark to {out_path}")

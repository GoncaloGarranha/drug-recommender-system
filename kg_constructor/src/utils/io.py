from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_csv_dicts(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as file_obj:
        return list(csv.DictReader(file_obj))


def write_csv_dicts(path: str | Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    return read_csv_dicts(path)


def write_csv(path: str | Path, header: List[str], rows: List[List[Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(header)
        writer.writerows(rows)


def write_csv_rows(path: str | Path, header: List[str], rows: List[List[Any]]) -> None:
    write_csv(path, header, rows)


def iter_json_files(folder: str | Path) -> Iterable[Path]:
    folder = Path(folder)

    if not folder.exists():
        return

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".json":
            yield file_path


def load_json_file(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON file: {path} | Error: {exc}") from exc
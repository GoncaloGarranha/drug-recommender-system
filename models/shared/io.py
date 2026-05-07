from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


def ensure_dir(path: str | Path) -> Path:
	"""Create a directory if it does not exist and return it as a Path."""
	directory = Path(path)
	directory.mkdir(parents=True, exist_ok=True)
	return directory


def read_json(path: str | Path) -> Any:
	"""Read a JSON file."""
	with Path(path).open("r", encoding="utf-8") as file_obj:
		return json.load(file_obj)


def save_json(path: str | Path, data: Any) -> None:
	"""Write data to a JSON file."""
	json_path = Path(path)
	json_path.parent.mkdir(parents=True, exist_ok=True)

	with json_path.open("w", encoding="utf-8") as file_obj:
		json.dump(data, file_obj, indent=2, ensure_ascii=False)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
	"""Read a CSV file into a list of dictionaries."""
	with Path(path).open("r", encoding="utf-8") as file_obj:
		return list(csv.DictReader(file_obj))


def write_csv(path: str | Path, header: List[str], rows: List[List[Any]]) -> None:
	"""Write rows to a CSV file."""
	csv_path = Path(path)
	csv_path.parent.mkdir(parents=True, exist_ok=True)

	with csv_path.open("w", newline="", encoding="utf-8") as file_obj:
		writer = csv.writer(file_obj)
		writer.writerow(header)
		writer.writerows(rows)


def read_triples_indexed_csv(path: str | Path) -> List[Dict[str, str]]:
	"""Read the indexed triples CSV used by the R-GCN pipeline."""
	return read_csv(path)

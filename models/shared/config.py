from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
	"""Load a YAML configuration file."""
	with Path(path).open("r", encoding="utf-8") as file_obj:
		return yaml.safe_load(file_obj) or {}


def save_config(path: str | Path, data: Dict[str, Any]) -> None:
	"""Save a YAML configuration file."""
	config_path = Path(path)
	config_path.parent.mkdir(parents=True, exist_ok=True)

	with config_path.open("w", encoding="utf-8") as file_obj:
		yaml.safe_dump(data, file_obj, sort_keys=False, allow_unicode=True)

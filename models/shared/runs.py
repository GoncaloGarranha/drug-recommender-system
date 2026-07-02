from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict

from .config import load_config, save_config
from .io import ensure_dir, save_json


def timestamp() -> str:
	"""Return the current timestamp in YYYY-MM-DD_HH-MM-SS format."""
	return time.strftime("%Y-%m-%d_%H-%M-%S")


def slugify(value: str) -> str:
	"""Convert a string into a filesystem-friendly slug."""
	cleaned = [character.lower() if character.isalnum() else "_" for character in value.strip()]
	slug = "".join(cleaned)
	while "__" in slug:
		slug = slug.replace("__", "_")
	return slug.strip("_") or "run"


def infer_run_domain(cfg: Dict[str, Any]) -> str:
	"""Infer the high-level experiment domain from the config."""
	processed_dir = str(cfg.get("data", {}).get("processed_dir", "")).lower()
	task_name = str(cfg.get("task", {}).get("name", "")).lower()

	if any(marker in processed_dir for marker in ("wn18", "fb15k")) or task_name == "tail_prediction":
		return "benchmarks"
	return "biomedical"


def resolve_run_dir(
	base_out_dir: str | Path,
	cfg: Dict[str, Any],
	model_name: str,
	seed: int | None = None,
) -> Path:
	"""Build the canonical run directory for a model/config pair.

	The shared run root is expected to be model-specific, e.g.
	`results/runs/rgcn` or `results/runs/content-based`.
	"""
	domain = infer_run_domain(cfg)
	task_name = str(cfg.get("task", {}).get("name") or Path(str(cfg.get("data", {}).get("processed_dir", "run"))).name)
	experiment_name = str(cfg.get("runs", {}).get("experiment_name") or "run")
	seed_value = seed if seed is not None else int(cfg.get("seed", 42))
	base_path = Path(base_out_dir)

	return base_path / domain / slugify(task_name) / slugify(experiment_name) / f"seed_{seed_value}"


def organize_legacy_runs(
	base_out_dir: str | Path,
	default_model_name: str,
	seed: int | None = None,
) -> list[tuple[Path, Path]]:
	"""Move flat legacy run folders into the canonical nested layout.

	This is intentionally conservative: it only moves direct children of the
	base runs directory that contain a `config.yaml` and are not already in a
	canonical domain folder.
	"""
	base_path = Path(base_out_dir)
	if not base_path.exists():
		return []

	moves: list[tuple[Path, Path]] = []
	for run_dir in base_path.iterdir():
		if not run_dir.is_dir():
			continue
		if run_dir.name in {"biomedical", "benchmarks"}:
			continue
		config_path = run_dir / "config.yaml"
		if not config_path.exists():
			continue

		run_cfg = load_config(config_path)
		model_name = str(run_cfg.get("model", {}).get("name") or default_model_name)
		target_root = resolve_run_dir(base_path, run_cfg, model_name=model_name, seed=seed)
		target_dir = target_root.parent / "legacy" / run_dir.name

		if target_dir.exists():
			continue

		target_dir.parent.mkdir(parents=True, exist_ok=True)
		shutil.move(str(run_dir), str(target_dir))
		moves.append((run_dir, target_dir))

	return moves


def capture_environment() -> Dict[str, Any]:
	"""Collect a compact environment snapshot for a run."""
	packages: Dict[str, str] = {}
	for package_name in ("numpy", "pandas", "torch", "dgl", "sklearn", "yaml"):
		try:
			module = __import__(package_name)
			packages[package_name] = getattr(module, "__version__", "unknown")
		except Exception:
			packages[package_name] = "not_available"

	return {
		"python": sys.version.replace("\n", " "),
		"executable": sys.executable,
		"platform": platform.platform(),
		"packages": packages,
	}


def capture_git_commit(repo_root: str | Path | None = None) -> str:
	"""Return the current git commit hash, if available."""
	repo_path = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
	try:
		result = subprocess.run(
			["git", "rev-parse", "HEAD"],
			cwd=repo_path,
			check=True,
			capture_output=True,
			text=True,
		)
		return result.stdout.strip()
	except Exception:
		return "unknown"


def write_run_artifacts(
	run_dir: str | Path,
	cfg: Dict[str, Any] | None = None,
	split_summary: Dict[str, Any] | None = None,
	model_summary: Dict[str, Any] | None = None,
	train_log: Any | None = None,
	environment: Dict[str, Any] | None = None,
	git_commit: str | None = None,
) -> Path:
	"""Write the standard metadata files shared by all runs."""
	run_path = ensure_dir(run_dir)

	if cfg is not None:
		save_config(run_path / "config.yaml", cfg)
	if split_summary is not None:
		save_json(run_path / "split_summary.json", split_summary)
	if model_summary is not None:
		save_json(run_path / "model_summary.json", model_summary)
	if train_log is not None:
		save_json(run_path / "train_log.json", train_log)
	if environment is not None:
		save_json(run_path / "environment.json", environment)
	if git_commit is not None:
		(run_path / "git_commit.txt").write_text(f"{git_commit}\n", encoding="utf-8")

	return run_path


def save_metrics(run_dir: str | Path, split: str, metrics: Dict[str, Any]) -> Path:
	"""Persist metrics for a given split using the standard filename."""
	run_path = ensure_dir(run_dir)
	metrics_path = run_path / f"metrics_{split}.json"
	save_json(metrics_path, metrics)
	return metrics_path


def find_latest_run(runs_dir: str | Path) -> str:
	"""
	Find the most recently modified run directory.
	
	Args:
		runs_dir: Directory containing run subdirectories.
	
	Returns:
		The full path to the most recently modified run directory.
	
	Raises:
		FileNotFoundError: If no run directories are found.
	"""
	runs_path = Path(runs_dir)
	if not runs_path.exists():
		raise FileNotFoundError(f"No run directories found in: {runs_path}")

	run_dirs = [path.parent for path in runs_path.rglob("config.yaml") if path.is_file()]
	if not run_dirs:
		run_dirs = [path.parent for path in runs_path.rglob("train_log.json") if path.is_file()]
	
	if not run_dirs:
		raise FileNotFoundError(f"No run directories found in: {runs_path}")

	run_dirs = list(dict.fromkeys(run_dirs))
	run_dirs.sort(key=lambda path: os.path.getmtime(path), reverse=True)
	return str(run_dirs[0])

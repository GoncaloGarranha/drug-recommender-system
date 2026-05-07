from __future__ import annotations

import glob
import os
import time
from pathlib import Path


def timestamp() -> str:
	"""Return the current timestamp in YYYY-MM-DD_HH-MM-SS format."""
	return time.strftime("%Y-%m-%d_%H-%M-%S")


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
	runs_dir = str(runs_dir)
	run_dirs = [
		path for path in glob.glob(os.path.join(runs_dir, "*"))
		if os.path.isdir(path)
	]
	
	if not run_dirs:
		raise FileNotFoundError(f"No run directories found in: {runs_dir}")

	run_dirs.sort(key=lambda path: os.path.getmtime(path), reverse=True)
	return run_dirs[0]

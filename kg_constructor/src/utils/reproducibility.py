from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def make_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def hash_input_directory(input_dir: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}

    if not input_dir.exists():
        return hashes

    for file_path in sorted(input_dir.rglob("*")):
        if file_path.is_file():
            relative_path = str(file_path.relative_to(input_dir)).replace("\\", "/")
            hashes[relative_path] = hash_file(file_path)

    return hashes


def write_run_summary(
    summary_path: Path,
    run_id: str,
    config_path: Path,
    cfg: Dict[str, Any],
    input_dir: Path,
    selected_steps: list[str],
    force: bool,
) -> None:
    summary = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "config_path": str(config_path),
        "config": cfg,
        "git_commit": get_git_commit(),
        "input_hashes": hash_input_directory(input_dir),
        "selected_steps": selected_steps,
        "force": force,
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
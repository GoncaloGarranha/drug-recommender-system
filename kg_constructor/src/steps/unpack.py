from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Dict, Any

from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)


def _find_required_dirs(extract_root: Path) -> Dict[str, Path]:
    """
    Locate required source directories (xml_files, txt_files, csv_files) after extraction.

    Some zips contain a single top-level folder; others extract the folders directly.
    This function searches recursively and returns the first match for each required dir.
    """
    required = ["xml_files", "txt_files", "csv_files"]
    found: Dict[str, Path] = {}

    for name in required:
        matches = list(extract_root.rglob(name))
        matches = [m for m in matches if m.is_dir()]
        if matches:
            # Prefer the shallowest directory (closest to extract_root)
            matches.sort(key=lambda p: len(p.relative_to(extract_root).parts))
            found[name] = matches[0]

    return found


def run(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 00: Unpack input zip into input_dir.

    Expected result: input_dir contains xml_files/, txt_files/, csv_files/.
    """
    input_dir = Path(paths.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    zip_path = Path(cfg.get("inputs", {}).get("zip_path", "")).expanduser()
    if not zip_path.is_absolute():
        zip_path = (Path(".").resolve() / zip_path).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"Input zip not found: {zip_path}")

    extract_subdir = cfg.get("inputs", {}).get("extract_subdir", "")
    extract_root = input_dir / extract_subdir if extract_subdir else input_dir

    # If not forcing, and required dirs already exist, skip.
    required_dirs = ["xml_files", "txt_files", "csv_files"]
    if not force and all((input_dir / d).exists() for d in required_dirs):
        logger.info("Input directories already exist. Skipping unpack (use --force to re-extract).")
        return

    # If forcing, wipe only the known dirs (safe cleanup).
    if force:
        for d in required_dirs:
            target = input_dir / d
            if target.exists():
                logger.info(f"Removing existing directory: {target}")
                shutil.rmtree(target, ignore_errors=True)

    extract_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting zip: {zip_path}")
    logger.info(f"Destination: {extract_root}")

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_root)

    # Validate and normalize folder placement:
    # If folders exist nested, move them to input_dir root.
    found = _find_required_dirs(extract_root)
    missing = [d for d in required_dirs if d not in found]

    if missing:
        raise RuntimeError(
            "After extraction, required folders were not found: "
            f"{missing}. Check the zip structure and inputs.zip_path."
        )

    for dname, src in found.items():
        dst = input_dir / dname
        if src.resolve() == dst.resolve():
            continue

        # Move the directory to input_dir root if it's nested.
        if dst.exists():
            logger.info(f"Removing existing destination before move: {dst}")
            shutil.rmtree(dst, ignore_errors=True)

        logger.info(f"Moving {src} -> {dst}")
        shutil.move(str(src), str(dst))

    logger.info("Unpack completed successfully.")

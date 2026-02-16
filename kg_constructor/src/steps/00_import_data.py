from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Any
from kg_constructor.src.utils.log import get_logger

logger = get_logger(__name__)

# Supported entity types
ENTITY_FOLDER_TYPES = ["xml_files", "csv_files", "txt_files"]

def import_entity_files_from_medjsonify(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """
    Step 00: Import entity files (xml/csv/txt) from MedJsonify-main/NER/data/entities/
    to kg_constructor/data/input/ (xml_files/, csv_files/, txt_files/).
    Creates destination folders if they do not exist and can overwrite if force=True.
    """
    source_entities_root = Path("MedJsonify-main/NER/data/entities")
    destination_input_root = Path(paths.input_dir)
    destination_input_root.mkdir(parents=True, exist_ok=True)

    for entity_type in ENTITY_FOLDER_TYPES:
        source_folder = source_entities_root / entity_type
        destination_folder = destination_input_root / entity_type

        if not source_folder.exists():
            logger.warning(f"Source folder not found: {source_folder}")
            continue

        if destination_folder.exists() and force:
            logger.info(f"Removing existing destination folder: {destination_folder}")
            shutil.rmtree(destination_folder)

        if not destination_folder.exists():
            logger.info(f"Copying {source_folder} -> {destination_folder}")
            shutil.copytree(source_folder, destination_folder)
        else:
            logger.info(f"Destination folder already exists: {destination_folder} (use --force to overwrite)")

    logger.info("Import data step completed.")

# Alias for pipeline compatibility
run = import_entity_files_from_medjsonify


if __name__ == "__main__":
    # Minimal config and paths mock for direct execution
    class Paths:
        input_dir = "kg_constructor/data/input"

    # No config needed for this step
    import_entity_files_from_medjsonify(cfg={}, paths=Paths(), force=True)

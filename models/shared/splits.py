from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .io import read_json, save_json


def get_split_ratios(split_cfg: Dict[str, Any]) -> Tuple[float, float, float]:
    """Normalize the train/validation/test split ratios from a config dict."""
    if "train_ratio" in split_cfg:
        train_ratio = float(split_cfg["train_ratio"])
        valid_ratio = float(split_cfg.get("val_ratio", split_cfg.get("valid_ratio", 0.1)))
        test_ratio = float(split_cfg["test_ratio"])
    else:
        train_ratio = float(split_cfg["train"])
        valid_ratio = float(split_cfg["valid"])
        test_ratio = float(split_cfg["test"])
    return train_ratio, valid_ratio, test_ratio


def split_target_triples(
    triples: List[Dict[str, str]],
    target_relation: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """Split target-relation triples into train/validation/test sets."""
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0.")

    target_triples = [row for row in triples if (row.get("rel") or "").strip() == target_relation]
    non_target_triples = [row for row in triples if (row.get("rel") or "").strip() != target_relation]

    if not target_triples:
        raise ValueError(f"No triples found for target relation: {target_relation}")

    rng = random.Random(seed)
    rng.shuffle(target_triples)

    n_total = len(target_triples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_target = target_triples[:n_train]
    valid_target = target_triples[n_train:n_train + n_val]
    test_target = target_triples[n_train + n_val:]

    return train_target, valid_target, test_target, non_target_triples


def build_known_positives(
    triples: List[Dict[str, str]],
    task_name: str | None = None,
    query_key: str | None = None,
    relation_key: str | None = None,
    candidate_key: str | None = None,
) -> Dict[Any, set]:
    """Build filtered-ranking positives indexed by the configured query key."""
    known = defaultdict(set)

    for row in triples:
        if query_key is not None and candidate_key is not None:
            # Accept either numeric ids or string entity identifiers.
            q_raw = row[query_key]
            try:
                q_val = int(q_raw)
            except Exception:
                q_val = q_raw

            if relation_key is not None:
                r_raw = row[relation_key]
                try:
                    r_val = int(r_raw)
                except Exception:
                    r_val = r_raw
                query_id = (q_val, r_val)
            else:
                query_id = q_val

            c_raw = row[candidate_key]
            try:
                candidate_id = int(c_raw)
            except Exception:
                candidate_id = c_raw
        else:
            src_id = int(row["src_id"])
            dst_id = int(row["dst_id"])

            if task_name == "disease_to_drug":
                query_id = dst_id
                candidate_id = src_id
            elif task_name == "drug_to_disease":
                query_id = src_id
                candidate_id = dst_id
            else:
                raise ValueError(f"Unsupported task name: {task_name}")

        known[query_id].add(candidate_id)

    return known


def save_splits(
    output_dir: str | Path,
    train_triples: List[Dict[str, str]],
    valid_triples: List[Dict[str, str]],
    test_triples: List[Dict[str, str]],
    split_metadata: Dict[str, Any] | None = None,
) -> None:
    """
    Save train/validation/test splits to JSON files for reproducibility.
    
    This ensures that all models use exactly the same splits.
    
    Args:
        output_dir: Directory to save split files.
        train_triples: Training split triples.
        valid_triples: Validation split triples.
        test_triples: Test split triples.
        split_metadata: Optional metadata about the split (ratios, seed, etc.).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_json(output_dir / "train_triples.json", train_triples)
    save_json(output_dir / "valid_triples.json", valid_triples)
    save_json(output_dir / "test_triples.json", test_triples)
    
    if split_metadata:
        save_json(output_dir / "split_metadata.json", split_metadata)


def load_splits(splits_dir: str | Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load train/validation/test splits from JSON files.
    
    Args:
        splits_dir: Directory containing split files.
    
    Returns:
        Tuple of (train_triples, valid_triples, test_triples).
    
    Raises:
        FileNotFoundError: If split files are missing.
    """
    splits_dir = Path(splits_dir)
    
    train_file = splits_dir / "train_triples.json"
    valid_file = splits_dir / "valid_triples.json"
    test_file = splits_dir / "test_triples.json"
    
    missing = [f for f in [train_file, valid_file, test_file] if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing split files in {splits_dir}:\n" + "\n".join(str(f) for f in missing)
        )
    
    return (
        read_json(train_file),
        read_json(valid_file),
        read_json(test_file),
    )


def get_or_create_splits(
    output_dir: str | Path,
    triples: List[Dict[str, str]],
    target_relation: str | None,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    force: bool = False,
    split_mode: str | None = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Load existing splits from disk, or create new ones if they don't exist.
    
    This is the recommended approach for consistent cross-model evaluation.
    
    Args:
        output_dir: Directory where splits are (or will be) saved.
        triples: All triples (used only if creating new splits).
        target_relation: Target relation for split (used only if creating new splits).
        seed: Random seed for split (used only if creating new splits).
        train_ratio: Training ratio (used only if creating new splits).
        val_ratio: Validation ratio (used only if creating new splits).
        test_ratio: Test ratio (used only if creating new splits).
        force: If True, recreate splits even if they exist.
    
    Returns:
        Tuple of (train_triples, valid_triples, test_triples, non_target_triples).
    """
    output_dir = Path(output_dir)
    splits_exist = (
        (output_dir / "train_triples.json").exists()
        and (output_dir / "valid_triples.json").exists()
        and (output_dir / "test_triples.json").exists()
    )
    
    if splits_exist and not force:
        # Validate stored metadata to ensure splits match current config
        metadata_file = output_dir / "split_metadata.json"
        if metadata_file.exists():
            stored_meta = read_json(metadata_file)
            mismatches = []
            if stored_meta.get("target_relation") != target_relation:
                mismatches.append("target_relation")
            if int(stored_meta.get("seed", -1)) != int(seed):
                mismatches.append("seed")
            if float(stored_meta.get("train_ratio", -1)) != float(train_ratio):
                mismatches.append("train_ratio")
            if float(stored_meta.get("val_ratio", stored_meta.get("valid_ratio", -1))) != float(val_ratio):
                mismatches.append("val_ratio")
            if float(stored_meta.get("test_ratio", -1)) != float(test_ratio):
                mismatches.append("test_ratio")
            # Mode check (e.g., 'official' vs random splits)
            stored_mode = stored_meta.get("mode")
            if (stored_mode or None) != (split_mode or None):
                mismatches.append("mode")
            if mismatches:
                raise ValueError(
                    "Existing splits found but metadata does not match current config. "
                    "Mismatch keys: %s. Remove the splits folder or run with force=True to recreate." % ",".join(mismatches)
                )

        train_triples, valid_triples, test_triples = load_splits(output_dir)

        if split_mode == "official" and target_relation is None:
            non_target_triples = []
        else:
            # For consistency with split_target_triples return signature
            non_target_triples = [
                row for row in triples
                if (row.get("rel") or "").strip() != (target_relation or "")
            ]
        return train_triples, valid_triples, test_triples, non_target_triples

    # Create new splits
    if split_mode == "official" and target_relation is None:
        train_triples = [row for row in triples if str(row.get("split", "")).strip() == "train"]
        valid_triples = [row for row in triples if str(row.get("split", "")).strip() == "valid"]
        test_triples = [row for row in triples if str(row.get("split", "")).strip() == "test"]
        non_target_triples = []
    elif split_mode == "official":
        # Expect the input triples to contain a 'split' column with values 'train','valid','test'
        train_triples = [
            row for row in triples
            if (row.get("rel") or "").strip() == target_relation and str(row.get("split", "")).strip() == "train"
        ]
        valid_triples = [
            row for row in triples
            if (row.get("rel") or "").strip() == target_relation and str(row.get("split", "")).strip() == "valid"
        ]
        test_triples = [
            row for row in triples
            if (row.get("rel") or "").strip() == target_relation and str(row.get("split", "")).strip() == "test"
        ]
        non_target_triples = [row for row in triples if (row.get("rel") or "").strip() != target_relation]

        # Basic validation
        if not (train_triples or valid_triples or test_triples):
            raise ValueError("No official split triples found for target relation: %s" % target_relation)
    else:
        if target_relation is None:
            # Generic link prediction mode: random split over all triples.
            if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
                raise ValueError("Split ratios must sum to 1.0.")

            all_triples = list(triples)
            rng = random.Random(seed)
            rng.shuffle(all_triples)

            n_total = len(all_triples)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)

            train_triples = all_triples[:n_train]
            valid_triples = all_triples[n_train:n_train + n_val]
            test_triples = all_triples[n_train + n_val:]
            non_target_triples = []
        else:
            train_triples, valid_triples, test_triples, non_target_triples = split_target_triples(
                triples=triples,
                target_relation=target_relation,
                seed=seed,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

    # Save for future use
    metadata = {
        "target_relation": target_relation,
        "seed": seed,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "num_train": len(train_triples),
        "num_valid": len(valid_triples),
        "num_test": len(test_triples),
        "mode": split_mode or "random",
    }
    save_splits(output_dir, train_triples, valid_triples, test_triples, metadata)

    return train_triples, valid_triples, test_triples, non_target_triples


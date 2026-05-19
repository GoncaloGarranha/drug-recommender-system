from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .config import load_config
from .graph_artifacts import load_prepared_graph, resolve_processed_dir
from .io import ensure_dir, save_json
from .ranking import build_candidate_pool
from .splits import build_known_positives, get_or_create_splits

TRIPLE_FIELDS = ("src", "rel", "dst")


def _triple_key(row: Mapping[str, object]) -> Tuple[str, str, str]:
	return tuple(str(row.get(field, "")).strip() for field in TRIPLE_FIELDS)


def _split_stats(rows: Sequence[Mapping[str, object]]) -> Dict[str, Any]:
	keys = [_triple_key(row) for row in rows]
	counts = Counter(keys)
	duplicate_keys = [key for key, count in counts.items() if count > 1]

	return {
		"rows": len(rows),
		"unique_rows": len(counts),
		"duplicate_rows": len(rows) - len(counts),
		"duplicate_examples": [
			{"src": src, "rel": rel, "dst": dst, "count": counts[(src, rel, dst)]}
			for src, rel, dst in duplicate_keys[:10]
		],
	}


def _row_to_candidate_id(task_name: str, row: Mapping[str, object]) -> int:
	if task_name == "disease_to_drug":
		return int(row["src_id"])
	if task_name == "drug_to_disease":
		return int(row["dst_id"])
	if task_name == "tail_prediction":
		return int(row["dst_id"])
	raise ValueError(f"Unsupported task name: {task_name}")


def _row_to_query_id(task_name: str, row: Mapping[str, object]) -> int | Tuple[int, int]:
	if task_name == "disease_to_drug":
		return int(row["dst_id"])
	if task_name == "drug_to_disease":
		return int(row["src_id"])
	if task_name == "tail_prediction":
		return int(row["src_id"]), int(row["rel_id"])
	raise ValueError(f"Unsupported task name: {task_name}")


def _build_known_positive_pairs(
	rows: Sequence[Mapping[str, object]],
	task_name: str,
) -> set[Tuple[object, int]]:
	pairs: set[Tuple[object, int]] = set()
	for row in rows:
		pairs.add((_row_to_query_id(task_name, row), _row_to_candidate_id(task_name, row)))
	return pairs


def _build_training_graph_rows(
	train_target: Sequence[Mapping[str, object]],
	non_target_triples: Sequence[Mapping[str, object]],
) -> List[Mapping[str, object]]:
	return list(train_target) + list(non_target_triples)


def _validate_condition(checks: List[Dict[str, Any]], name: str, condition: bool, details: Dict[str, Any]) -> None:
	checks.append({"name": name, "passed": bool(condition), "details": details})


def validate_experiment_integrity(
	cfg: Dict[str, Any],
	*,
	config_path: str | Path | None = None,
	output_path: str | Path | None = None,
	force: bool = False,
) -> Dict[str, Any]:
	"""Validate split integrity and persist a machine-readable report.
	
	Note: This function always recreates splits fresh to guarantee consistency
	with the current config, which helps catch data leakage issues.
	"""
	# Validation always uses force=True to ensure splits match current config
	force_val = True
	processed_dir = resolve_processed_dir(cfg["data"]["processed_dir"], task_name=cfg.get("task", {}).get("name"))
	prepared = load_prepared_graph(processed_dir)
	triples = prepared["triples"]
	entity2id = prepared["entity2id"]
	entity_types = prepared["entity_types"]

	task_cfg = cfg.get("task", {})
	task_name = str(task_cfg.get("name", "tail_prediction"))
	target_relation = task_cfg.get("target_relation")
	split_cfg = cfg["split"]
	seed = int(cfg.get("seed", 42))
	candidate_entity_type = task_cfg.get("candidate_entity_type", "ALL")

	splits_dir = Path(processed_dir) / "splits"
	train_target, valid_target, test_target, non_target_triples = get_or_create_splits(
		output_dir=splits_dir,
		triples=triples,
		target_relation=target_relation,
		seed=seed,
		train_ratio=float(split_cfg["train_ratio"]),
		val_ratio=float(split_cfg["val_ratio"]),
		test_ratio=float(split_cfg["test_ratio"]),
		split_mode=split_cfg.get("mode") if isinstance(split_cfg, dict) else None,
		force=force_val,
	)

	train_graph_rows = _build_training_graph_rows(train_target, non_target_triples)
	all_target_rows = list(train_target) + list(valid_target) + list(test_target)

	report: Dict[str, Any] = {
		"config_path": str(config_path) if config_path is not None else None,
		"processed_dir": str(processed_dir),
		"splits_dir": str(splits_dir),
		"task_name": task_name,
		"target_relation": target_relation,
		"split_mode": split_cfg.get("mode") if isinstance(split_cfg, dict) else None,
		"seed": seed,
		"status": "passed",
		"split_stats": {
			"train": _split_stats(train_target),
			"valid": _split_stats(valid_target),
			"test": _split_stats(test_target),
			"train_graph": _split_stats(train_graph_rows),
			"all_target": _split_stats(all_target_rows),
		},
		"checks": [],
	}

	failures: List[str] = []

	# Check 1: no duplicate exact triples inside each logical split.
	for split_name, rows in {
		"train": train_graph_rows,
		"valid": valid_target,
		"test": test_target,
	}.items():
		stats = _split_stats(rows)
		passed = stats["duplicate_rows"] == 0
		_validate_condition(
			report["checks"],
			f"no_duplicates_within_{split_name}",
			passed,
			{"split": split_name, **stats},
		)
		if not passed:
			failures.append(f"duplicate triples inside {split_name}")

	# Check 2: exact triples do not overlap across train graph / valid / test.
	train_graph_keys = {_triple_key(row) for row in train_graph_rows}
	valid_keys = {_triple_key(row) for row in valid_target}
	test_keys = {_triple_key(row) for row in test_target}
	valid_overlap = sorted(train_graph_keys & valid_keys)
	test_overlap = sorted(train_graph_keys & test_keys)
	valid_test_overlap = sorted(valid_keys & test_keys)

	_validate_condition(
		report["checks"],
		"train_graph_vs_valid_disjoint",
		not valid_overlap,
		{"overlap_count": len(valid_overlap), "examples": valid_overlap[:10]},
	)
	_validate_condition(
		report["checks"],
		"train_graph_vs_test_disjoint",
		not test_overlap,
		{"overlap_count": len(test_overlap), "examples": test_overlap[:10]},
	)
	_validate_condition(
		report["checks"],
		"valid_vs_test_disjoint",
		not valid_test_overlap,
		{"overlap_count": len(valid_test_overlap), "examples": valid_test_overlap[:10]},
	)
	if valid_overlap:
		failures.append("validation target triples leaked into training graph")
	if test_overlap:
		failures.append("test target triples leaked into training graph")
	if valid_test_overlap:
		failures.append("validation and test splits overlap")

	# Check 3: known positives must include every target triple from train+valid+test.
	if task_name == "tail_prediction":
		known_positives = build_known_positives(
			triples=all_target_rows,
			task_name=None,
			query_key="src_id",
			relation_key="rel_id",
			candidate_key="dst_id",
		)
	else:
		known_positives = build_known_positives(all_target_rows, task_name=task_name)

	missing_known_positive_examples: List[Dict[str, Any]] = []
	for row in all_target_rows:
		query_id = _row_to_query_id(task_name, row)
		candidate_id = _row_to_candidate_id(task_name, row)
		if candidate_id not in known_positives.get(query_id, set()):
			missing_known_positive_examples.append(
				{"query": query_id, "candidate": candidate_id, "row": dict(row)}
			)

	_validate_condition(
		report["checks"],
		"known_positives_complete",
		not missing_known_positive_examples,
		{"missing_examples": missing_known_positive_examples[:10], "missing_count": len(missing_known_positive_examples)},
	)
	if missing_known_positive_examples:
		failures.append("known positives do not cover all target triples")

	# Check 4: candidate pool must contain every true positive candidate id.
	required_entity_ids = {
		_row_to_candidate_id(task_name, row)
		for row in all_target_rows
	}
	candidate_pool = build_candidate_pool(
		entity_types=entity_types,
		entity2id=entity2id,
		candidate_entity_type=str(candidate_entity_type),
		device=__import__("torch").device("cpu"),
		required_entity_ids=required_entity_ids,
	)
	candidate_pool_ids = set(int(value) for value in candidate_pool.detach().cpu().tolist())
	missing_candidates = sorted(required_entity_ids - candidate_pool_ids)
	_validate_condition(
		report["checks"],
		"candidate_pool_complete",
		not missing_candidates,
		{"missing_candidate_ids": missing_candidates[:50], "missing_count": len(missing_candidates), "candidate_pool_size": len(candidate_pool_ids)},
	)
	if missing_candidates:
		failures.append("candidate pool is missing true positive entities")

	# Check 5: official benchmark splits must remain intact.
	source_has_split = bool(triples) and ("split" in triples[0])
	official_mode = split_cfg.get("mode") == "official"
	official_check_applies = official_mode and source_has_split
	official_preserved = True
	official_details: Dict[str, Any] = {"applies": official_check_applies}
	if official_check_applies:
		source_train = [_triple_key(row) for row in triples if str(row.get("split", "")).strip() == "train"]
		source_valid = [_triple_key(row) for row in triples if str(row.get("split", "")).strip() == "valid"]
		source_test = [_triple_key(row) for row in triples if str(row.get("split", "")).strip() == "test"]

		if target_relation is None:
			official_preserved = (
				sorted(source_train) == sorted(_triple_key(row) for row in train_target)
				and sorted(source_valid) == sorted(_triple_key(row) for row in valid_target)
				and sorted(source_test) == sorted(_triple_key(row) for row in test_target)
			)
			official_details.update(
				{
					"source_counts": {"train": len(source_train), "valid": len(source_valid), "test": len(source_test)},
					"loaded_counts": {"train": len(train_target), "valid": len(valid_target), "test": len(test_target)},
				}
			)
		else:
			source_target_train = [
				_triple_key(row)
				for row in triples
				if str(row.get("split", "")).strip() == "train" and (row.get("rel") or "").strip() == str(target_relation)
			]
			source_target_valid = [
				_triple_key(row)
				for row in triples
				if str(row.get("split", "")).strip() == "valid" and (row.get("rel") or "").strip() == str(target_relation)
			]
			source_target_test = [
				_triple_key(row)
				for row in triples
				if str(row.get("split", "")).strip() == "test" and (row.get("rel") or "").strip() == str(target_relation)
			]
			source_non_target = [
				_triple_key(row)
				for row in triples
				if (row.get("rel") or "").strip() != str(target_relation)
			]
			official_preserved = (
				sorted(source_target_train) == sorted(_triple_key(row) for row in train_target)
				and sorted(source_target_valid) == sorted(_triple_key(row) for row in valid_target)
				and sorted(source_target_test) == sorted(_triple_key(row) for row in test_target)
				and sorted(source_non_target) == sorted(_triple_key(row) for row in non_target_triples)
			)
			official_details.update(
				{
					"source_counts": {
						"train_target": len(source_target_train),
						"valid_target": len(source_target_valid),
						"test_target": len(source_target_test),
						"non_target": len(source_non_target),
					},
					"loaded_counts": {
						"train_target": len(train_target),
						"valid_target": len(valid_target),
						"test_target": len(test_target),
						"non_target": len(non_target_triples),
					},
				}
			)

	_validate_condition(
		report["checks"],
		"official_benchmark_splits_preserved",
		official_preserved,
		official_details,
	)
	if official_check_applies and not official_preserved:
		failures.append("official benchmark split assignment changed")

	# Check 6: split statistics are recorded in the report.
	_validate_condition(
		report["checks"],
		"split_statistics_saved",
		True,
		{"saved_sections": list(report["split_stats"].keys())},
	)

	report["status"] = "failed" if failures else "passed"
	report["failures"] = failures
	if candidate_pool.numel() > 0:
		report["candidate_pool_size"] = int(candidate_pool.numel())
	report["known_positives_size"] = sum(len(values) for values in known_positives.values())

	if output_path is None:
		output_path = splits_dir / "split_integrity_report.json"
	output_path = Path(output_path)
	ensure_dir(output_path.parent)
	save_json(output_path, report)
	report["report_path"] = str(output_path)

	if failures:
		raise ValueError("Split integrity validation failed: " + "; ".join(failures))

	return report


def load_and_validate_experiment_integrity(
	config_path: str | Path,
	*,
	output_path: str | Path | None = None,
	force: bool = False,
) -> Dict[str, Any]:
	"""Load a config file and run the integrity validation.
	
	Note: Validation always uses force=True internally to ensure splits are
	always fresh and consistent with the current config, regardless of the
	force parameter.
	"""
	cfg = load_config(config_path)
	# Always force fresh splits for validation to ensure consistency with config
	return validate_experiment_integrity(
		cfg,
		config_path=config_path,
		output_path=output_path,
		force=True,
	)

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

DRUG_PREFIXES = ("DRUG:", "DRUGBANK:", "CHEBI:")


def _resolve_path(value, default):
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _load_blacklist(path: Path) -> set[str]:
    if not path.exists():
        return set()

    blacklist = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():
        cleaned = line.split("#", 1)[0].strip()

        if cleaned:
            blacklist.add(cleaned)

    return blacklist


def _detect_columns(rows):
    cols = set(rows[0].keys())

    if {"src", "rel", "dst"}.issubset(cols):
        return "src", "rel", "dst"

    if {"head", "relation", "tail"}.issubset(cols):
        return "head", "relation", "tail"

    raise ValueError("Expected columns src,rel,dst or head,relation,tail.")


def _is_drug(entity: str) -> bool:
    return entity.startswith(DRUG_PREFIXES)


def _schema_error(src: str, rel: str, dst: str) -> str | None:
    if rel in {"INDICATION", "CONTRAINDICATION"}:
        if not (_is_drug(src) and dst.startswith("DOID:")):
            return "drug_disease_schema_error"

    elif rel == "HAS_INGREDIENT":
        if not _is_drug(src):
            return "has_ingredient_schema_error"

    elif rel == "INTERACTS_WITH_GENE":
        if not (_is_drug(src) and dst.startswith("GENE:")):
            return "drug_gene_schema_error"

    elif rel == "ASSOCIATED_WITH_GENE":
        if not (src.startswith("DOID:") and dst.startswith("GENE:")):
            return "disease_gene_schema_error"

    return None


def run(cfg: dict[str, Any], paths, force: bool = False) -> None:
    gate_cfg = cfg.get("quality_gate", {})

    if not bool(gate_cfg.get("enabled", True)):
        print("Quality gate disabled by config. Skipping step.")
        return

    input_path = _resolve_path(
        gate_cfg.get("input_triples_path"),
        ROOT / "kg_constructor/data/output/triples/triples_with_dgidb.csv",
    )
    blacklist_path = _resolve_path(
        gate_cfg.get("blacklist_path"),
        ROOT / "kg_constructor/configs/blacklist.txt",
    )
    report_path = _resolve_path(
        gate_cfg.get("report_path"),
        ROOT / "kg_constructor/data/output/reports/quality_gate_report.json",
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Quality gate input file not found: {input_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError("Input triples file is empty.")

    src_col, rel_col, dst_col = _detect_columns(rows)
    blacklist = _load_blacklist(blacklist_path)

    triples = [
        (
            str(row.get(src_col, "")).strip(),
            str(row.get(rel_col, "")).strip(),
            str(row.get(dst_col, "")).strip(),
        )
        for row in rows
    ]

    counts = Counter(triples)

    duplicates = [t for t, n in counts.items() if n > 1]
    self_loops = [t for t in triples if t[0] == t[2]]
    blacklisted = [t for t in triples if t[0] in blacklist or t[2] in blacklist]
    empty_fields = [t for t in triples if not t[0] or not t[1] or not t[2]]

    indication_pairs = {
        (src, dst)
        for src, rel, dst in triples
        if rel == "INDICATION"
    }

    contraindication_pairs = {
        (src, dst)
        for src, rel, dst in triples
        if rel == "CONTRAINDICATION"
    }

    conflicts = sorted(indication_pairs & contraindication_pairs)

    bad_drugbank = sorted({
        entity
        for src, _, dst in triples
        for entity in (src, dst)
        if entity.startswith("DRUGBANK:http")
    })

    schema_errors = []
    schema_error_types = Counter()

    for src, rel, dst in triples:
        error = _schema_error(src, rel, dst)
        if error:
            schema_errors.append((src, rel, dst))
            schema_error_types[error] += 1

    checks = {
        "duplicate_triples": len(duplicates),
        "self_loops": len(self_loops),
        "blacklisted_rows": len(blacklisted),
        "empty_field_rows": len(empty_fields),
        "indication_contraindication_conflicts": len(conflicts),
        "bad_drugbank_entities": len(bad_drugbank),
        "schema_errors": len(schema_errors),
    }

    failure_options = {
        "duplicate_triples": bool(
            gate_cfg.get("fail_on_duplicate_triples", True)
        ),
        "self_loops": bool(
            gate_cfg.get("fail_on_self_loops", True)
        ),
        "blacklisted_rows": bool(
            gate_cfg.get("fail_on_blacklisted_entities", True)
        ),
        "empty_field_rows": bool(
            gate_cfg.get("fail_on_empty_fields", True)
        ),
        "indication_contraindication_conflicts": bool(
            gate_cfg.get("fail_on_contraindication_conflicts", True)
        ),
        "bad_drugbank_entities": bool(
            gate_cfg.get("fail_on_bad_drugbank_urls", True)
        ),
        "schema_errors": bool(
            gate_cfg.get("fail_on_schema_errors", True)
        ),
    }

    failures = [
        name
        for name, value in checks.items()
        if value > 0
        and failure_options.get(name, True)
    ]

    warnings = [
        name
        for name, value in checks.items()
        if value > 0
        and not failure_options.get(name, True)
    ]

    report = {
        "status": "failed" if failures else "passed",
        "input_triples_path": str(input_path.relative_to(ROOT)),
        "rows": len(rows),
        "relation_distribution": dict(Counter(rel for _, rel, _ in triples)),
        "checks": checks,
        "failure_options": failure_options,
        "schema_error_types": dict(schema_error_types),
        "warnings": warnings,
        "failures": failures,
        "examples": {
            "duplicate_triple": duplicates[:1],
            "self_loop": self_loops[:1],
            "blacklisted_row": blacklisted[:1],
            "empty_field_row": empty_fields[:1],
            "indication_contraindication_conflict": conflicts[:1],
            "bad_drugbank_entity": bad_drugbank[:1],
            "schema_error": schema_errors[:1],
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))

    if failures:
        raise RuntimeError(
            f"Quality gate failed. See report: {report_path}. Failures: {failures}"
        )

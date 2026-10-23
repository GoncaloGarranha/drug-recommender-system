"""Build and validate the Zenodo dataset package used by the dissertation.

The script copies the canonical graph without modifying it, creates a filtered
alias table for final-graph entities, adds the quality and provenance reports,
and writes checksums for the resulting package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ID = "biomedical_knowledge_graph_2025-10_to_2026-03"
VALID_REVIEW_STATUSES = {
    "coherent_name_group",
    "case_only_variation",
    "possible_resolution_error",
    "shared_drug_concept",
}
RELATION_SCHEMA = [
    (
        "HAS_INGREDIENT",
        "Drug",
        "Ingredient",
        "Connects a drug or product to an ingredient.",
    ),
    (
        "INDICATION",
        "Drug",
        "Disease",
        "Connects a drug or product to a therapeutic indication.",
    ),
    (
        "CONTRAINDICATION",
        "Drug",
        "Disease",
        "Connects a drug or product to a contraindicated disease or condition.",
    ),
    (
        "ASSOCIATED_WITH_GENE",
        "Disease",
        "Gene",
        "Connects a disease to an associated gene.",
    ),
    (
        "INTERACTS_WITH_GENE",
        "Drug",
        "Gene",
        "Connects a drug to a gene reported by the interaction source.",
    ),
]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return loaded


def read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    """Read a CSV file and verify its required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"Missing columns in {path}: {', '.join(sorted(missing))}"
            )
        return [dict(row) for row in reader]


def graph_statistics(
    triples: Iterable[dict[str, str]],
) -> tuple[Counter[str], set[str], Counter[str]]:
    """Return relation, entity-namespace and entity statistics."""
    relations: Counter[str] = Counter()
    entities: set[str] = set()
    for row in triples:
        source = row["src"].strip()
        relation = row["rel"].strip()
        target = row["dst"].strip()
        if not source or not relation or not target:
            raise ValueError("The canonical graph contains an empty field")
        relations[relation] += 1
        entities.update((source, target))
    namespaces = Counter(
        entity.split(":", 1)[0] if ":" in entity else "UNPREFIXED"
        for entity in entities
    )
    return relations, entities, namespaces


def validate_graph(
    graph_path: Path,
    triples: list[dict[str, str]],
    build_manifest: dict[str, Any],
) -> tuple[Counter[str], set[str], Counter[str]]:
    """Validate the graph against the recorded canonical build."""
    canonical = build_manifest.get("canonical_graph")
    if not isinstance(canonical, dict):
        raise ValueError("Build manifest has no canonical_graph object")

    actual_hash = sha256_file(graph_path)
    if actual_hash != canonical.get("sha256"):
        raise ValueError(
            "Canonical graph checksum differs from the build manifest: "
            f"{actual_hash}"
        )

    relations, entities, namespaces = graph_statistics(triples)
    expected_relations = canonical.get("relation_counts")
    if dict(sorted(relations.items())) != dict(sorted(expected_relations.items())):
        raise ValueError("Relation counts differ from the build manifest")
    if len(triples) != canonical.get("triple_count"):
        raise ValueError("Triple count differs from the build manifest")
    if len(entities) != canonical.get("entity_count"):
        raise ValueError("Entity count differs from the build manifest")
    if len(relations) != canonical.get("relation_count"):
        raise ValueError("Relation count differs from the build manifest")
    return relations, entities, namespaces


def validate_review(
    audit_rows: list[dict[str, str]],
    review_rows: list[dict[str, str]],
) -> Counter[str]:
    """Check that every audited entity has one completed review."""
    audit_ids = [row["entity_id"].strip() for row in audit_rows]
    review_ids = [row["entity_id"].strip() for row in review_rows]
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError("Drug-name audit contains duplicate entity identifiers")
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("Drug-name review contains duplicate entity identifiers")
    if set(audit_ids) != set(review_ids):
        missing = sorted(set(audit_ids) - set(review_ids))
        extra = sorted(set(review_ids) - set(audit_ids))
        raise ValueError(
            "Drug-name review does not match the audit; "
            f"missing={missing}, extra={extra}"
        )

    counts: Counter[str] = Counter()
    for row in review_rows:
        status = row["review_status"].strip()
        if status not in VALID_REVIEW_STATUSES:
            raise ValueError(
                f"Unknown or empty review status for {row['entity_id']}: {status!r}"
            )
        if not row["review_notes"].strip():
            raise ValueError(f"Missing review notes for {row['entity_id']}")
        counts[status] += 1
    return counts


def write_filtered_aliases(
    source_path: Path,
    output_path: Path,
    graph_entities: set[str],
) -> tuple[int, int]:
    """Write unique aliases belonging to entities in the final graph."""
    rows = read_csv(source_path, {"alias", "entity_id", "alias_type"})
    seen: set[tuple[str, str, str]] = set()
    selected: list[dict[str, str]] = []
    covered_entities: set[str] = set()
    for row in rows:
        key = (row["alias"], row["entity_id"], row["alias_type"])
        if row["entity_id"] not in graph_entities or key in seen:
            continue
        seen.add(key)
        selected.append(row)
        covered_entities.add(row["entity_id"])

    expected_alias_entities = {
        entity for entity in graph_entities if not entity.startswith("GENE:")
    }
    if covered_entities != expected_alias_entities:
        missing = sorted(expected_alias_entities - covered_entities)
        raise ValueError(f"Aliases are missing for final-graph entities: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["alias", "entity_id", "alias_type"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(selected)
    return len(selected), len(covered_entities)


def write_relation_schema(path: Path) -> None:
    """Write the graph relation schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["relation", "source_type", "target_type", "description"])
        writer.writerows(RELATION_SCHEMA)


def inventory_files(package_dir: Path) -> list[dict[str, Any]]:
    """Describe package payload files, excluding generated integrity files."""
    inventory: list[dict[str, Any]] = []
    for path in sorted(package_dir.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "SHA256SUMS"}:
            continue
        inventory.append(
            {
                "path": path.relative_to(package_dir).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return inventory


def write_checksums(package_dir: Path) -> None:
    """Write checksums for all package files except SHA256SUMS itself."""
    paths = sorted(
        path
        for path in package_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    lines = [
        f"{sha256_file(path)}  {path.relative_to(package_dir).as_posix()}"
        for path in paths
    ]
    (package_dir / "SHA256SUMS").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def write_deterministic_zip(package_dir: Path) -> Path:
    """Create a ZIP archive with stable paths, timestamps and permissions."""
    archive_path = package_dir.with_suffix(".zip")
    if archive_path.exists():
        raise FileExistsError(f"Archive already exists: {archive_path}")
    with zipfile.ZipFile(
        archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in package_dir.rglob("*") if item.is_file()):
            relative = Path(package_dir.name) / path.relative_to(package_dir)
            info = zipfile.ZipInfo(relative.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    return archive_path


def copy_required(source: Path, destination: Path) -> None:
    """Copy one required file, preserving its contents."""
    if not source.is_file():
        raise FileNotFoundError(f"Required file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def build_package(build_dir: Path, publication_dir: Path, output_dir: Path) -> Path:
    """Validate inputs and create one complete publication package."""
    build_dir = build_dir.resolve()
    publication_dir = publication_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.with_suffix(".zip").exists():
        raise FileExistsError(
            "Output already exists. Choose an empty output path so an existing "
            f"publication package is never overwritten: {output_dir}"
        )

    graph_path = build_dir / "triples" / "triples_with_dgidb.csv"
    aliases_path = build_dir / "mappings" / "aliases.csv"
    quality_path = build_dir / "reports" / "quality_gate_report.json"
    resolution_path = build_dir / "mappings" / "resolution_report.json"
    build_manifest_path = build_dir / "manifest.json"
    audit_path = publication_dir / "drug_name_audit.csv"
    review_path = publication_dir / "drug_name_review.csv"
    readme_path = publication_dir / "README.md"

    build_manifest = load_json(build_manifest_path)
    triples = read_csv(graph_path, {"src", "rel", "dst"})
    relations, entities, namespaces = validate_graph(
        graph_path, triples, build_manifest
    )
    audit_rows = read_csv(audit_path, {"entity_id"})
    review_rows = read_csv(
        review_path,
        {"entity_id", "review_status", "preferred_label", "review_notes"},
    )
    review_counts = validate_review(audit_rows, review_rows)
    quality_report = load_json(quality_path)
    if quality_report.get("status") != "passed":
        raise ValueError("The canonical graph has not passed its quality gate")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.", dir=output_dir.parent
    ) as temporary:
        temporary_dir = Path(temporary) / output_dir.name
        temporary_dir.mkdir()

        copy_required(readme_path, temporary_dir / "README.md")
        copy_required(graph_path, temporary_dir / "data" / "triples.csv")
        alias_rows, alias_entities = write_filtered_aliases(
            aliases_path,
            temporary_dir / "mappings" / "entity_aliases.csv",
            entities,
        )
        write_relation_schema(temporary_dir / "schema" / "relations.csv")
        copy_required(
            quality_path, temporary_dir / "quality" / "quality_gate_report.json"
        )
        copy_required(
            resolution_path, temporary_dir / "quality" / "resolution_report.json"
        )
        copy_required(
            audit_path, temporary_dir / "quality" / "drug_name_audit.csv"
        )
        copy_required(
            review_path, temporary_dir / "quality" / "drug_name_review.csv"
        )
        copy_required(
            build_manifest_path,
            temporary_dir / "provenance" / "build_manifest.json",
        )

        manifest = {
            "manifest_version": 1,
            "package_id": PACKAGE_ID,
            "source_build_id": build_manifest.get("build_id"),
            "canonical_graph": {
                "path": "data/triples.csv",
                "sha256": sha256_file(temporary_dir / "data" / "triples.csv"),
                "triple_count": len(triples),
                "entity_count": len(entities),
                "entity_counts": {
                    "drug": namespaces["DRUG"]
                    + namespaces["CHEBI"]
                    + namespaces["DRUGBANK"],
                    "disease": namespaces["DOID"],
                    "ingredient": namespaces["UNII"],
                    "gene": namespaces["GENE"],
                },
                "relation_count": len(relations),
                "relation_counts": dict(sorted(relations.items())),
            },
            "aliases": {
                "path": "mappings/entity_aliases.csv",
                "row_count": alias_rows,
                "entity_count": alias_entities,
                "gene_labels_embedded_in_identifiers": True,
            },
            "drug_name_review": {
                "audit_entity_count": len(audit_rows),
                "reviewed_entity_count": len(review_rows),
                "status_counts": dict(sorted(review_counts.items())),
                "modifies_canonical_graph": False,
            },
            "quality_gate": {
                "status": quality_report.get("status"),
                "checks": quality_report.get("checks", {}),
            },
            "files": inventory_files(temporary_dir),
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        write_checksums(temporary_dir)
        temporary_dir.rename(output_dir)

    return write_deterministic_zip(output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate the Zenodo publication package."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        required=True,
        help="Final KG build containing triples, mappings and reports",
    )
    parser.add_argument(
        "--publication-dir",
        type=Path,
        required=True,
        help="Directory containing README.md and the drug-name review files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to create; existing outputs are never overwritten",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive_path = build_package(
        args.build_dir, args.publication_dir, args.output_dir
    )
    package_dir = args.output_dir.resolve()
    print(f"Package created: {package_dir}")
    print(f"Archive created: {archive_path}")
    print(f"Archive SHA-256: {sha256_file(archive_path)}")


if __name__ == "__main__":
    main()

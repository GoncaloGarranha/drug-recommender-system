from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_TRIPLES = ROOT / "kg_constructor/data/output/triples/triples_with_genes.csv"
DEFAULT_DGIDB_INTERACTIONS = ROOT / "kg_constructor/data/external/dgidb/interactions.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "kg_constructor/data/output/dgidb"
DEFAULT_OUTPUT_TRIPLES = ROOT / "kg_constructor/data/output/triples/triples_with_dgidb.csv"

RELATION = "INTERACTS_WITH_GENE"


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _map_drug(row: pd.Series, kg_entities: set[str]) -> Tuple[str | None, str | None]:
    concept_id = str(row.get("drug_concept_id", "")).strip()
    drug_name = str(row.get("drug_name", "")).strip().upper()

    if concept_id.startswith("drugbank:"):
        dbid = concept_id.split(":", 1)[1].upper()
        candidate = f"DRUGBANK:https://go.drugbank.com/drugs/{dbid}"
        if candidate in kg_entities:
            return candidate, "drugbank"

    if drug_name:
        candidate = f"DRUG:{drug_name}"
        if candidate in kg_entities:
            return candidate, "name"

    return None, None


def _map_gene(row: pd.Series, kg_entities: set[str]) -> Tuple[str | None, str | None]:
    gene_name = str(row.get("gene_name", "")).strip().upper()

    if gene_name:
        candidate = f"GENE:{gene_name}"
        if candidate in kg_entities:
            return candidate, "symbol"

    return None, None


def add_dgidb_interactions(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    dgidb_cfg = cfg.get("dgidb_interactions", {})

    if not bool(dgidb_cfg.get("enabled", False)):
        print("DGIdb interaction step disabled by config. Skipping step.")
        return

    input_triples = _resolve_path(dgidb_cfg.get("input_triples_path"), DEFAULT_INPUT_TRIPLES)
    dgidb_interactions = _resolve_path(
        dgidb_cfg.get("interactions_path"),
        DEFAULT_DGIDB_INTERACTIONS,
    )
    output_dir = _resolve_path(dgidb_cfg.get("output_dir"), DEFAULT_OUTPUT_DIR)
    output_triples = _resolve_path(
        dgidb_cfg.get("output_triples_path"),
        DEFAULT_OUTPUT_TRIPLES,
    )

    mapped_path = output_dir / "drug_gene_interactions_mapped.csv"
    unmapped_drugs_path = output_dir / "unmapped_drugs.csv"
    unmapped_genes_path = output_dir / "unmapped_genes.csv"
    report_path = output_dir / "dgidb_import_report.json"

    output_files = [mapped_path, unmapped_drugs_path, unmapped_genes_path, report_path, output_triples]

    if not force and all(path.exists() for path in output_files):
        print("DGIdb outputs already exist. Skipping step (use --force to re-run).")
        return

    if not input_triples.exists():
        raise FileNotFoundError(f"Missing input triples: {input_triples}")

    if not dgidb_interactions.exists():
        raise FileNotFoundError(f"Missing DGIdb interactions TSV: {dgidb_interactions}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_triples.parent.mkdir(parents=True, exist_ok=True)

    triples = pd.read_csv(input_triples, dtype=str)
    dgidb = pd.read_csv(dgidb_interactions, sep="\t", dtype=str, comment="#")

    kg_entities = set(triples["src"]) | set(triples["dst"])

    mapped_rows = []
    unmapped_drugs = []
    unmapped_genes = []

    for _, row in dgidb.iterrows():
        drug, drug_mapping = _map_drug(row, kg_entities)
        gene, gene_mapping = _map_gene(row, kg_entities)

        if drug is None:
            unmapped_drugs.append(
                {
                    "drug_name": row.get("drug_name"),
                    "drug_concept_id": row.get("drug_concept_id"),
                }
            )

        if gene is None:
            unmapped_genes.append(
                {
                    "gene_name": row.get("gene_name"),
                    "gene_concept_id": row.get("gene_concept_id"),
                }
            )

        if drug and gene:
            mapped_rows.append(
                {
                    "src": drug,
                    "rel": RELATION,
                    "dst": gene,
                    "source": "DGIdb",
                    "dgidb_source": row.get("interaction_source_db_name"),
                    "interaction_types": row.get("interaction_types"),
                    "interaction_score": row.get("interaction_score"),
                    "drug_mapping": drug_mapping,
                    "gene_mapping": gene_mapping,
                }
            )

    mapped_df = pd.DataFrame(
        mapped_rows,
        columns=[
            "src",
            "rel",
            "dst",
            "source",
            "dgidb_source",
            "interaction_types",
            "interaction_score",
            "drug_mapping",
            "gene_mapping",
        ],
    ).drop_duplicates(subset=["src", "rel", "dst"])

    unmapped_drugs_df = pd.DataFrame(
        unmapped_drugs,
        columns=["drug_name", "drug_concept_id"],
    ).drop_duplicates()

    unmapped_genes_df = pd.DataFrame(
        unmapped_genes,
        columns=["gene_name", "gene_concept_id"],
    ).drop_duplicates()

    mapped_df.to_csv(mapped_path, index=False)
    unmapped_drugs_df.to_csv(unmapped_drugs_path, index=False)
    unmapped_genes_df.to_csv(unmapped_genes_path, index=False)

    dgidb_triples = mapped_df[["src", "rel", "dst"]].copy()

    triples_with_dgidb = pd.concat(
        [triples[["src", "rel", "dst"]], dgidb_triples],
        ignore_index=True,
    ).drop_duplicates(subset=["src", "rel", "dst"])

    triples_with_dgidb.to_csv(output_triples, index=False)

    report = {
        "input_triples": str(input_triples.relative_to(ROOT)),
        "dgidb_interactions": str(dgidb_interactions.relative_to(ROOT)),
        "output_triples": str(output_triples.relative_to(ROOT)),
        "relation": RELATION,
        "dgidb_rows": int(len(dgidb)),
        "mapped_unique_interactions": int(len(mapped_df)),
        "unmapped_unique_drugs": int(len(unmapped_drugs_df)),
        "unmapped_unique_genes": int(len(unmapped_genes_df)),
        "original_kg_triples": int(len(triples)),
        "new_kg_triples": int(len(triples_with_dgidb)),
        "drug_mapping_methods": mapped_df["drug_mapping"].value_counts().to_dict(),
        "gene_mapping_methods": mapped_df["gene_mapping"].value_counts().to_dict(),
        "relation_distribution": triples_with_dgidb["rel"].value_counts().to_dict(),
        "top_dgidb_sources": mapped_df["dgidb_source"].value_counts().head(20).to_dict(),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


run = add_dgidb_interactions

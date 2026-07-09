from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_TRIPLES = ROOT / "kg_constructor/data/output/triples/triples.filtered.csv"
DEFAULT_DGIDB_INTERACTIONS = ROOT / "kg_constructor/data/external/dgidb/interactions.tsv"
DEFAULT_OUTPUT_DIR = ROOT / "kg_constructor/data/output/dgidb"
DEFAULT_OUTPUT_TRIPLES = ROOT / "kg_constructor/data/output/triples/triples_with_dgidb.csv"
DEFAULT_ALIASES = ROOT / "kg_constructor/data/output/mappings/aliases.csv"
DEFAULT_DRUG_NAME_MAP = ROOT / "kg_constructor/data/output/mappings/drug_name_to_entity.json"

RELATION = "INTERACTS_WITH_GENE"
DRUGBANK_RE = re.compile(r"DB\d{5}", re.IGNORECASE)


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _safe_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _norm_key(value: Any) -> str:
    return " ".join(_safe_text(value).split()).upper()


def _drugbank_keys(value: Any) -> set[str]:
    text = _safe_text(value)
    if not text:
        return set()

    keys = {_norm_key(text)}

    if text.lower().startswith("http"):
        last = text.rstrip("/").rsplit("/", 1)[-1]
        keys.add(_norm_key(last))
        keys.add(_norm_key(f"DRUGBANK:{last}"))
        keys.add(_norm_key(f"https://go.drugbank.com/drugs/{last}"))

    if text.lower().startswith("drugbank:"):
        dbid = text.split(":", 1)[1].strip()
        if dbid.lower().startswith("http"):
            dbid = dbid.rstrip("/").rsplit("/", 1)[-1]
        keys.add(_norm_key(dbid))
        keys.add(_norm_key(f"DRUGBANK:{dbid}"))
        keys.add(_norm_key(f"https://go.drugbank.com/drugs/{dbid}"))

    for match in DRUGBANK_RE.findall(text):
        dbid = match.upper()
        keys.add(dbid)
        keys.add(f"DRUGBANK:{dbid}")
        keys.add(f"HTTPS://GO.DRUGBANK.COM/DRUGS/{dbid}")

    return {key for key in keys if key}


def _build_drug_alias_index(
    kg_entities: set[str],
    aliases_path: Path,
    drug_name_map_path: Path,
) -> dict[str, str]:
    """
    Build a robust index from drug aliases / external IDs to canonical KG drug entities.

    This fixes the old DGIdb behaviour where drugs were only matched as:
        DRUG:<NAME>
        DRUGBANK:https://go.drugbank.com/drugs/<DBID>

    The new resolver can map DGIdb names or DrugBank IDs to canonical entities such as:
        DRUGBANK:DB00002
        CHEBI:...
        DRUG:...
    """
    kg_drug_entities = {
        entity
        for entity in kg_entities
        if entity.startswith(("DRUG:", "DRUGBANK:", "CHEBI:"))
    }

    index: dict[str, str] = {}

    def add(key: Any, entity_id: str) -> None:
        norm = _norm_key(key)
        if norm and entity_id in kg_drug_entities:
            index.setdefault(norm, entity_id)

    for entity_id in kg_drug_entities:
        add(entity_id, entity_id)

        if entity_id.startswith("DRUG:"):
            add(entity_id.replace("DRUG:", "", 1), entity_id)

        if entity_id.startswith("DRUGBANK:"):
            dbid = entity_id.replace("DRUGBANK:", "", 1)
            for key in _drugbank_keys(dbid) | _drugbank_keys(entity_id):
                add(key, entity_id)

        if entity_id.startswith("CHEBI:"):
            add(entity_id.replace("CHEBI:", "", 1), entity_id)

    if drug_name_map_path.exists():
        name_map = json.loads(drug_name_map_path.read_text(encoding="utf-8"))
        for name_norm, entity_id in name_map.items():
            if entity_id in kg_drug_entities:
                add(name_norm, entity_id)
                add(f"DRUG:{name_norm}", entity_id)

    if aliases_path.exists():
        aliases = pd.read_csv(aliases_path, dtype=str).fillna("")

        for _, row in aliases.iterrows():
            alias = _safe_text(row.get("alias"))
            entity_id = _safe_text(row.get("entity_id"))
            alias_type = _safe_text(row.get("alias_type")).upper()

            if entity_id not in kg_drug_entities:
                continue

            if not (
                alias_type.startswith("DRUG")
                or alias_type.startswith("CHEBI")
                or alias_type.startswith("DRUGBANK")
            ):
                continue

            add(alias, entity_id)

            if "DRUGBANK" in alias_type or "DRUGBANK" in alias.upper() or "DB" in alias.upper():
                for key in _drugbank_keys(alias):
                    add(key, entity_id)

    return index


def _map_drug(
    row: pd.Series,
    kg_entities: set[str],
    drug_alias_index: dict[str, str],
) -> Tuple[str | None, str | None]:
    concept_id = _safe_text(row.get("drug_concept_id"))
    drug_name = _safe_text(row.get("drug_name"))

    if concept_id:
        for key in _drugbank_keys(concept_id):
            mapped = drug_alias_index.get(_norm_key(key))
            if mapped:
                return mapped, "drugbank"

        mapped = drug_alias_index.get(_norm_key(concept_id))
        if mapped:
            return mapped, "concept_id"

    if drug_name:
        mapped = drug_alias_index.get(_norm_key(drug_name))
        if mapped:
            return mapped, "alias"

        direct_name_candidate = f"DRUG:{_norm_key(drug_name)}"
        if direct_name_candidate in kg_entities:
            return direct_name_candidate, "name"

    return None, None


def _map_gene(
    row: pd.Series,
    kg_entities: set[str],
    require_existing_gene: bool = False,
) -> Tuple[str | None, str | None]:
    gene_name = _safe_text(row.get("gene_name")).upper()

    if gene_name:
        candidate = f"GENE:{gene_name}"

        if not require_existing_gene or candidate in kg_entities:
            return candidate, "symbol"

    return None, None


def add_dgidb_interactions(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    dgidb_cfg = cfg.get("dgidb_interactions", {})

    if not bool(dgidb_cfg.get("enabled", False)):
        print("DGIdb interaction step disabled by config. Skipping step.")
        return

    require_existing_gene = bool(dgidb_cfg.get("require_existing_gene", False))

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
    aliases_path = _resolve_path(dgidb_cfg.get("aliases_path"), DEFAULT_ALIASES)
    drug_name_map_path = _resolve_path(
        dgidb_cfg.get("drug_name_map_path"),
        DEFAULT_DRUG_NAME_MAP,
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
    dgidb = pd.read_csv(dgidb_interactions, sep="\t", dtype=str, comment="#").fillna("")

    kg_entities = set(triples["src"]) | set(triples["dst"])
    drug_alias_index = _build_drug_alias_index(
        kg_entities=kg_entities,
        aliases_path=aliases_path,
        drug_name_map_path=drug_name_map_path,
    )

    mapped_rows = []
    unmapped_drugs = []
    unmapped_genes = []

    for _, row in dgidb.iterrows():
        drug, drug_mapping = _map_drug(row, kg_entities, drug_alias_index)
        gene, gene_mapping = _map_gene(
            row,
            kg_entities,
            require_existing_gene=require_existing_gene,
        )

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
        "drug_alias_index_size": int(len(drug_alias_index)),
        "require_existing_gene": require_existing_gene,
        "drug_mapping_methods": mapped_df["drug_mapping"].value_counts().to_dict(),
        "gene_mapping_methods": mapped_df["gene_mapping"].value_counts().to_dict(),
        "relation_distribution": triples_with_dgidb["rel"].value_counts().to_dict(),
        "top_dgidb_sources": mapped_df["dgidb_source"].value_counts().head(20).to_dict(),
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


run = add_dgidb_interactions

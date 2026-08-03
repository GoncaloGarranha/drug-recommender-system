from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[3]

DISEASES_FOR_GENES = (
    ROOT / "kg_constructor/data/external/diseases_for_genes.csv"
)

OT_DISEASE_PATH = ROOT / "kg_constructor/data/external/open_targets/disease/disease.parquet"
OT_ASSOC_PATH = ROOT / "kg_constructor/data/external/open_targets/association_overall_direct"
OT_TARGET_PATH = ROOT / "kg_constructor/data/external/open_targets/target"


def _resolve_path(value: str | Path | None, default: Path) -> Path:
    if value is None:
        return default

    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def add_gene_associations(cfg: Dict[str, Any], paths, force: bool = False) -> None:
    """Step 06: add disease-gene associations from Open Targets to KG triples."""
    gene_cfg = cfg.get("gene_associations", {})

    output_dir = Path(paths.output_dir)
    default_triples_path = output_dir / "triples" / "triples.csv"
    default_output_dir = output_dir / "gene_associations"
    default_output_triples = (
        output_dir / "triples" / "triples_with_genes.csv"
    )

    mode = str(
        gene_cfg.get(
            "mode",
            "current_deduplicated",
        )
    )

    allowed_modes = {
        "current_deduplicated",
        "historical_exact",
    }

    if mode not in allowed_modes:
        raise ValueError(
            "Invalid gene association mode: "
            f"{mode}. Allowed values: "
            f"{sorted(allowed_modes)}"
        )

    print("Gene association mode:", mode)

    triples_path = _resolve_path(
        gene_cfg.get("input_triples_path"),
        default_triples_path,
    )
    out_triples = _resolve_path(
        gene_cfg.get("output_triples_path"),
        default_output_triples,
    )

    if not bool(gene_cfg.get("enabled", False)):
        import shutil

        if not triples_path.exists():
            raise FileNotFoundError(
                f"Gene association pass-through input not found: {triples_path}"
            )

        out_triples.parent.mkdir(parents=True, exist_ok=True)

        if force or not out_triples.exists():
            shutil.copyfile(triples_path, out_triples)
            print(
                "Gene association step disabled by config. "
                f"Pass-through copied {triples_path} -> {out_triples}"
            )
        else:
            print(
                "Gene association step disabled by config. "
                f"Pass-through output already exists: {out_triples}"
            )

        return

    import pandas as pd

    diseases_for_genes = _resolve_path(gene_cfg.get("diseases_for_genes_path"), DISEASES_FOR_GENES)
    triples_path = _resolve_path(gene_cfg.get("input_triples_path"), default_triples_path)
    ot_disease_path = _resolve_path(gene_cfg.get("open_targets_disease_path"), OT_DISEASE_PATH)
    ot_assoc_path = _resolve_path(gene_cfg.get("open_targets_association_path"), OT_ASSOC_PATH)
    ot_target_path = _resolve_path(gene_cfg.get("open_targets_target_path"), OT_TARGET_PATH)
    out_dir = _resolve_path(gene_cfg.get("output_dir"), default_output_dir)
    out_triples = _resolve_path(gene_cfg.get("output_triples_path"), default_output_triples)

    if not force and out_triples.exists():
        print(f"Gene association output already exists. Skipping step (use --force to re-run): {out_triples}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    out_triples.parent.mkdir(parents=True, exist_ok=True)

    # Load the input KG once. Disease-gene enrichment must not introduce
    # diseases that are absent from the graph being enriched.
    triples = pd.read_csv(triples_path)

    required_columns = {"src", "rel", "dst"}
    missing_columns = required_columns - set(triples.columns)
    if missing_columns:
        raise ValueError(
            "Input triples file is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    graph_entities = (
        set(triples["src"].dropna().astype(str))
        | set(triples["dst"].dropna().astype(str))
    )
    existing_diseases = {
        entity
        for entity in graph_entities
        if entity.startswith("DOID:")
    }

    diseases = pd.read_csv(diseases_for_genes)
    diseases = diseases.rename(
        columns={
            "id": "doid",
            "name": "kg_disease_name",
        }
    )

    if mode == "historical_exact":
        print(
            "Historical disease candidates retained: "
            f"{len(diseases)}"
        )
    else:
        diseases["doid"] = diseases["doid"].astype(str)

        diseases_before_filter = len(diseases)
        diseases = (
            diseases[
                diseases["doid"].isin(existing_diseases)
            ]
            .drop_duplicates(subset=["doid"])
            .copy()
        )

        print(
            "Disease candidates restricted to input KG: "
            f"{diseases_before_filter} -> {len(diseases)}"
        )

    ot_diseases = pd.read_parquet(ot_disease_path)

    rows = []

    for _, row in ot_diseases.iterrows():
        refs = row.get("dbXRefs")

        if refs is None:
            continue

        for ref in list(refs):
            ref = str(ref).replace("_", ":")

            if ref.startswith("DOID:"):
                rows.append(
                    {
                        "doid": ref,
                        "ot_disease_id": row["id"],
                        "ot_disease_name": row["name"],
                    }
                )

    xrefs = pd.DataFrame(rows).drop_duplicates()

    disease_map = diseases.merge(xrefs, on="doid", how="left")
    disease_map.to_csv(out_dir / "disease_ot_mapping_xrefs.csv", index=False)

    matched_diseases = disease_map.dropna(subset=["ot_disease_id"]).copy()

    print(f"KG diseases: {len(diseases)}")
    print(f"Matched diseases: {matched_diseases['doid'].nunique()}")

    associations = pd.read_parquet(ot_assoc_path)
    associations = associations[
        ["diseaseId", "targetId", "associationScore", "evidenceCount"]
    ]

    associations = associations.merge(
        matched_diseases[["doid", "kg_disease_name", "ot_disease_id", "ot_disease_name"]],
        left_on="diseaseId",
        right_on="ot_disease_id",
        how="inner",
    )

    print(f"Raw matched disease-gene associations: {len(associations)}")

    targets = pd.read_parquet(ot_target_path)
    targets = targets[["id", "approvedSymbol", "approvedName"]].rename(
        columns={
            "id": "targetId",
            "approvedSymbol": "gene_symbol",
            "approvedName": "gene_name",
        }
    )

    associations = associations.merge(targets, on="targetId", how="left")
    associations = associations.dropna(subset=["gene_symbol"])

    associations["gene_id"] = "GENE:" + associations["gene_symbol"].astype(str)

    associations = associations.sort_values(
        ["doid", "associationScore"], ascending=[True, False]
    )

    associations.to_csv(
        out_dir / "disease_gene_associations.csv",
        index=False,
    )

    top_k_genes_per_disease = int(gene_cfg.get("top_k_genes_per_disease", 25))
    min_score = float(gene_cfg.get("min_score", 0.1))

    if mode == "historical_exact":
        # Historical behaviour:
        # - no Disease-Gene deduplication before top-k;
        # - associations were already sorted above;
        # - top-k was applied directly per DOID.
        filtered = associations[
            associations["associationScore"] >= min_score
        ].copy()

        filtered = (
            filtered
            .groupby("doid")
            .head(top_k_genes_per_disease)
        )
    else:
        # A single DOID can map to more than one Open Targets disease.
        # Keep only the strongest Disease-Gene association before top-k.
        unique_associations = (
            associations
            .sort_values(
                ["doid", "associationScore"],
                ascending=[True, False],
            )
            .drop_duplicates(
                subset=["doid", "gene_id"],
                keep="first",
            )
        )

        filtered = unique_associations[
            unique_associations["associationScore"] >= min_score
        ].copy()

        filtered = (
            filtered
            .sort_values(
                ["doid", "associationScore"],
                ascending=[True, False],
            )
            .groupby("doid", group_keys=False)
            .head(top_k_genes_per_disease)
        )

    gene_edges = filtered[["doid", "gene_id", "associationScore"]].copy()
    gene_edges = gene_edges.rename(
        columns={
            "doid": "src",
            "gene_id": "dst",
            "associationScore": "score",
        }
    )
    gene_edges["rel"] = "ASSOCIATED_WITH_GENE"
    gene_edges = gene_edges[["src", "rel", "dst", "score"]]

    gene_edges.to_csv(
        out_dir / "disease_gene_edges.csv",
        index=False,
    )

    print(f"Filtered gene edges: {len(gene_edges)}")
    print(f"Diseases with gene edges: {gene_edges['src'].nunique()}")
    print(f"Unique genes: {gene_edges['dst'].nunique()}")

    triples_with_genes = pd.concat(
        [
            triples,
            gene_edges[["src", "rel", "dst"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    triples_with_genes.to_csv(out_triples, index=False)

    print(f"Original triples: {len(triples)}")
    print(f"Triples with genes: {len(triples_with_genes)}")
    print(f"Saved: {out_triples}")


run = add_gene_associations

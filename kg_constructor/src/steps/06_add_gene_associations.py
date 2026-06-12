from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]

DISEASES_FOR_GENES = ROOT / "diseases_for_genes.csv"
TRIPLES_PATH = ROOT / "kg_constructor/data/output/triples/triples.csv"

OT_DISEASE_PATH = ROOT / "kg_constructor/data/external/open_targets/disease/disease.parquet"
OT_ASSOC_PATH = ROOT / "kg_constructor/data/external/open_targets/association_overall_direct"
OT_TARGET_PATH = ROOT / "kg_constructor/data/external/open_targets/target"

OUT_DIR = ROOT / "kg_constructor/data/output/gene_associations"
OUT_TRIPLES = ROOT / "kg_constructor/data/output/triples/triples_with_genes.csv"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TRIPLES.parent.mkdir(parents=True, exist_ok=True)

    diseases = pd.read_csv(DISEASES_FOR_GENES)
    diseases = diseases.rename(columns={"id": "doid", "name": "kg_disease_name"})

    ot_diseases = pd.read_parquet(OT_DISEASE_PATH)

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
    disease_map.to_csv(OUT_DIR / "disease_ot_mapping_xrefs.csv", index=False)

    matched_diseases = disease_map.dropna(subset=["ot_disease_id"]).copy()

    print(f"KG diseases: {len(diseases)}")
    print(f"Matched diseases: {matched_diseases['doid'].nunique()}")

    associations = pd.read_parquet(OT_ASSOC_PATH)
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

    targets = pd.read_parquet(OT_TARGET_PATH)
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
        OUT_DIR / "disease_gene_associations.csv",
        index=False,
    )

    # Keep top genes per disease to avoid exploding the KG.
    TOP_K_GENES_PER_DISEASE = 25
    MIN_SCORE = 0.1

    filtered = associations[associations["associationScore"] >= MIN_SCORE].copy()
    filtered = filtered.groupby("doid").head(TOP_K_GENES_PER_DISEASE)

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
        OUT_DIR / "disease_gene_edges.csv",
        index=False,
    )

    print(f"Filtered gene edges: {len(gene_edges)}")
    print(f"Diseases with gene edges: {gene_edges['src'].nunique()}")
    print(f"Unique genes: {gene_edges['dst'].nunique()}")

    triples = pd.read_csv(TRIPLES_PATH)

    triples_with_genes = pd.concat(
        [
            triples,
            gene_edges[["src", "rel", "dst"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    triples_with_genes.to_csv(OUT_TRIPLES, index=False)

    print(f"Original triples: {len(triples)}")
    print(f"Triples with genes: {len(triples_with_genes)}")
    print(f"Saved: {OUT_TRIPLES}")


if __name__ == "__main__":
    main()
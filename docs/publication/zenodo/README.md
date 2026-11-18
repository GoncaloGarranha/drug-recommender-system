# Biomedical Knowledge Graph for Disease-to-Drug Recommendation

This dataset contains the final six-month biomedical Knowledge Graph used in
the disease-to-drug recommendation experiments described in the associated
dissertation. The graph combines regulatory drug information processed through
MedJsonify with disease-gene associations from Open Targets Platform and
drug-gene interactions from DGIdb.

## Dataset scope

The graph contains 51,759 triples, 10,380 entities and five relation types. Its
entities comprise 2,347 drugs, 793 diseases, 2,278 ingredients and 4,962 genes.
The relation counts are:

| Relation | Source entity | Target entity | Triples |
| --- | --- | --- | ---: |
| `HAS_INGREDIENT` | Drug | Ingredient | 24,030 |
| `INDICATION` | Drug | Disease | 6,143 |
| `CONTRAINDICATION` | Drug | Disease | 2,221 |
| `ASSOCIATED_WITH_GENE` | Disease | Gene | 9,951 |
| `INTERACTS_WITH_GENE` | Drug | Gene | 9,414 |

The directions shown above describe the triples stored in `data/triples.csv`.
In the dissertation's recommendation task, `INDICATION` is reversed to query a
disease and rank candidate drugs. This modelling transformation does not change
the published graph.

## Package contents

- `data/triples.csv`: canonical graph with the columns `src`, `rel` and `dst`.
- `mappings/entity_aliases.csv`: aliases for every drug, disease and ingredient
  in the final graph. Exact duplicate rows and aliases for entities outside the
  final graph were removed when this package was generated. Gene symbols are
  contained directly in identifiers of the form `GENE:<symbol>`.
- `schema/relations.csv`: entity domains, ranges and short descriptions for the
  five relation types.
- `quality/quality_gate_report.json`: automatic checks applied to the canonical
  graph.
- `quality/resolution_report.json`: summary produced by the entity-resolution
  stage.
- `quality/drug_name_audit.csv`: reproducible audit of drug identifiers linked
  to more than one original source name.
- `quality/drug_name_review.csv`: manual classification of every row in the
  drug-name audit.
- `provenance/build_manifest.json`: source versions, retrieval dates,
  construction settings and the expected graph checksum.
- `manifest.json`: package statistics and SHA-256 checksums for the data files.
- `SHA256SUMS`: checksums that can be verified with
  `shasum -a 256 -c SHA256SUMS` from the package directory.

## Drug-name audit

The automatic audit identified 188 drug entities associated with more than one
exact source name. All 188 rows were reviewed manually: 115 were classified as
coherent name groups, 46 as case-only variations, 10 as shared drug concepts and
17 as possible entity-resolution errors. The review records an assessment; it
does not modify the identifiers or triples in the canonical graph. This keeps
the published data identical to the graph used in the experiments while making
the known cases available for inspection.

## Quality checks and retained relation pairs

The quality gate found no duplicate triples, self-loops, empty fields,
blacklisted rows, invalid DrugBank entities or schema errors. It identified
1,427 drug-disease pairs represented by both `INDICATION` and
`CONTRAINDICATION`. These pairs were retained intentionally because the source
relations can refer to different clinical contexts. They are reported as a
warning rather than treated as invalid rows.

## Provenance

The regulatory data were obtained from DailyMed, the Orange Book and the Purple
Book and were initially processed by MedJsonify. The final graph was enriched
with Open Targets Platform version 26.03, retrieved on 12 June 2026, and DGIdb
version 5.0.7 (data version December 2024), retrieved on 6 July 2026. Exact
source information and checksums available at build time are recorded in
`provenance/build_manifest.json`.

The DailyMed component covers July, August, November and December 2025,
and January and February 2026. The Orange Book and Purple Book inputs
correspond to the February 2026 versions.

The internal build identifier `2025-10_to_2026-03` is retained to preserve
the repository paths and experiment configurations. It does not represent
the exact temporal coverage of the regulatory data.

The construction workflow depends on the JSON structure and extracted content
produced by MedJsonify. Omissions, extraction errors or structural changes in
that preprocessing stage may affect the entities and relations in this dataset.

## Intended use and limitations

The dataset supports research on Knowledge Graph construction, link prediction
and computational prioritisation of drug-disease associations. A model score or
high ranking does not establish a new therapeutic indication, clinical efficacy
or safety. Results derived from this graph require independent biomedical and,
where appropriate, experimental or clinical validation.

The package preserves the graph used in the dissertation experiments, including
the possible entity-resolution errors documented in the manual review. Users
whose analysis is sensitive to these cases should inspect
`quality/drug_name_review.csv` and define an explicit filtering or correction
policy.

## Reproducible package generation

From the repository root, the package can be regenerated with:

```bash
python kg_constructor/scripts/publication/build_zenodo_package.py \
  --build-dir kg_constructor/data/builds/2025-10_to_2026-03 \
  --publication-dir docs/publication/zenodo \
  --output-dir docs/publication/zenodo/dist/biomedical_knowledge_graph_2025-10_to_2026-03
```

The script verifies the canonical graph against its build manifest, validates
the correspondence between the automatic audit and the manual review, and
creates the package manifest, checksums and ZIP archive.

## Citation and licence

Use the citation and licence recorded on the Zenodo record associated with the
published version. The package does not replace or modify the terms of the
upstream data sources.

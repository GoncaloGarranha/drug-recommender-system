# Hybrid Content-Based Baseline

This directory contains a single, interpretable baseline for the biomedical
`disease_to_drug` task. It is designed for a fair comparison with the R-GCN:

- same canonical graph;
- same train, validation and test splits;
- same drug candidate pool;
- same filtered-ranking protocol;
- same MRR and Hits@K metrics;
- no target information from validation or test is used to build profiles.

## Score components

The final score keeps each source of evidence separate:

1. `ingredient_knn`: similarity to training drugs for the query disease using
   `HAS_INGREDIENT` profiles;
2. `direct_gene_knn`: similarity to training drugs using
   `INTERACTS_WITH_GENE` profiles;
3. `train_indication_gene_knn`: similarity using genes reached through
   `Drug -> INDICATION(train only) -> Disease -> ASSOCIATED_WITH_GENE -> Gene`;
4. `disease_gene_overlap`: direct compatibility between
   `Disease -> ASSOCIATED_WITH_GENE -> Gene` and
   `Drug -> INTERACTS_WITH_GENE -> Gene`;
5. `contraindication_knn`: similarity to training drugs using their broader
   contraindication profiles;
6. `direct_contraindication`: explicit penalty when a candidate drug is
   contraindicated for the query disease;
7. `popularity`: a small train-only fallback signal.

TF-IDF spaces are fitted independently for each profile type. This prevents a
relation with many attributes from dominating the others merely because of its
scale. Similarity to known training drugs is calculated against their normalized
centroid.

## Leakage policy

`HAS_INGREDIENT`, `INTERACTS_WITH_GENE`, `ASSOCIATED_WITH_GENE` and
`CONTRAINDICATION` are treated as contextual graph relations. The only feature
that traverses `INDICATION` is `train_indication_gene_knn`, and that traversal
uses the training split exclusively.

Validation may be used to select component weights. Test must only be evaluated
after the selected validation weights have been frozen.

## Preparation

Prepare the graph using the same processed directory used by the R-GCN:

```bash
python -m models.rgcn.src.prepare_graph \
  --config models/content-based/config/disease_to_drug_with_dgidb.yaml
```

The required processed artefacts are:

```text
models/rgcn/data/processed/disease_to_drug/six_months_main_random_split/
├── entity2id.json
├── entity_types.json
├── triples_indexed.csv
└── splits/
```

## 1. Select weights on validation

```bash
python -m models.content-based.src.baseline \
  --config models/content-based/config/disease_to_drug_with_dgidb.yaml \
  --split valid \
  --tune
```

The run directory will contain:

```text
selected_weights.json
weight_search_results.json
coverage_report.json
diagnostics_valid.json
metrics_valid.json
```

## 2. Evaluate test with frozen weights

Use the `selected_weights.json` created by the validation run:

```bash
python -m models.content-based.src.baseline \
  --config models/content-based/config/disease_to_drug_with_dgidb.yaml \
  --split test \
  --weights-file results/runs/content-based/<run>/selected_weights.json
```

Do not use `--tune` with the test split.

## Fixed-weight smoke test

Without `--tune` or `--weights-file`, the script uses the fixed weights in the
YAML. This mode is useful to check that the pipeline runs, but it should not be
reported as the final baseline unless those weights were fixed independently of
test performance.

## Reports

Each run records:

- profile coverage per relation;
- candidates with no usable profile;
- diseases without training drugs;
- diseases without associated genes;
- zero-signal evaluation cases;
- selected weights and validation search results;
- standard ranking metrics and reproducibility artefacts.

## Removed implementation

The previous `metapath_baseline.py` was removed because it overlapped with the
TF-IDF baseline while omitting direct drug-gene compatibility. The final model
now exposes all evidence sources within one configurable implementation.

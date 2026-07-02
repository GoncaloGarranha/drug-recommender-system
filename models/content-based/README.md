# Content-Based Baseline (TF-IDF)

A lightweight content-based baseline for the biomedical `disease_to_drug_full` recommendation task.

The baseline ranks candidate drugs for a given disease using TF-IDF representations of drug profiles built from selected Knowledge Graph relations, such as `HAS_INGREDIENT` and `CONTRAINDICATION`.

This baseline is intentionally aligned with the R-GCN evaluation protocol for the biomedical dataset, so results can be compared under the same splits, candidate pool and filtered ranking setting.

---

## Supported Task

This baseline currently supports only the biomedical `disease_to_drug` task, used here in the `disease_to_drug_full` graph setting:

```text
disease_to_drug
Disease → Drug recommendation
```

It is not intended for:

- `drug_to_disease`;
- `tail_prediction`;
- WN18RR;
- FB15k-237.

For generic benchmark datasets such as WN18RR and FB15k-237, use Knowledge Graph Embedding baselines such as DistMult, ComplEx or TransE.

---

## Overview

The implementation lives in:

```text
models/content-based/src/baseline.py
```

The baseline follows these steps:

1. Load canonical triples from the path defined in the R-GCN config (`data.triples_path`).
2. Load processed graph artifacts from `data.processed_dir`.
3. Select candidate entities of type `DRUG`.
4. Build one textual profile per candidate drug using the configured feature relations.
5. Vectorize drug profiles with TF-IDF.
6. Compute cosine similarity between candidate drugs.
7. Load or create the same train/valid/test splits used by R-GCN.
8. Compute filtered ranking metrics on the selected split (`valid` or `test`).
9. Log results to `results/benchmarks.csv` and save a full structured run under `results/runs/content-based/`.

---

## How It Works

Each candidate drug is represented as a bag of relation-level tokens.

For the main biomedical full-graph setting, the profile is built from:

```text
CONTRAINDICATION
HAS_INGREDIENT
```

The target relation is:

```text
INDICATION
```

For each disease in the evaluation split, the baseline:

1. identifies drugs associated with that disease in the training set;
2. scores all candidate drugs by maximum cosine similarity to the training drugs;
3. removes other known positives from the ranking using the filtered protocol;
4. computes tie-aware ranks;
5. aggregates metrics: `MRR`, `Hits@1`, `Hits@3`, `Hits@10`.

This provides an interpretable reference point for assessing how much predictive signal is captured by immediate drug-profile similarity compared with learned relational embeddings.

---

## Configuration

The baseline uses the same YAML config as the R-GCN model.

Example:

```text
models/rgcn/configs/disease_to_drug.yaml
```

Important fields:

```yaml
data:
  triples_path: "kg_constructor/data/output/triples/triples.csv"
  processed_dir: "models/rgcn/data/processed/disease_to_drug_full"

task:
  name: "disease_to_drug"
  target_relation: "INDICATION"
  query_entity_type: "DISEASE"
  candidate_entity_type: "DRUG"

baseline:
  use_ingredients: true
  feature_relations:
    - "CONTRAINDICATION"
    - "HAS_INGREDIENT"
```

### Notes

- `baseline.use_ingredients: true` ensures that `HAS_INGREDIENT` is used.
- `baseline.feature_relations` defines the additional relations used to build drug profiles.
- The target relation (`INDICATION`) is removed from the feature set to avoid direct leakage.
- If `baseline.use_ingredients: false`, then `baseline.feature_relations` must still define the relations used as features.

---

## Required Preparation

Before running the baseline, prepare and validate the graph artifacts using the same config:

```bash
python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/disease_to_drug.yaml --force
```

```bash
python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/disease_to_drug_full
```

The baseline expects the processed directory to contain:

```text
triples_indexed.csv
entity2id.json
entity_types.json
splits/
```

---

## Run

From the repository root:

```bash
python -m models.content-based.src.baseline --config models/rgcn/configs/disease_to_drug.yaml --split test
```

Evaluate on validation split:

```bash
python -m models.content-based.src.baseline --config models/rgcn/configs/disease_to_drug.yaml --split valid
```

Force split regeneration if metadata mismatches or if you intentionally want to recreate splits:

```bash
python -m models.content-based.src.baseline --config models/rgcn/configs/disease_to_drug.yaml --split test --force
```

---

## Output

The script prints:

```text
Hits@1
Hits@3
Hits@10
MRR
```

The results are appended to:

```text
results/benchmarks.csv
```

at the repository root.

The structured run directory contains:

```text
config.yaml
metrics_valid.json or metrics_test.json
train_log.json
split_summary.json
model_summary.json
environment.json
git_commit.txt
```

---

## Fair Comparison with R-GCN

For a valid comparison with R-GCN, both models must use:

- the same `data.triples_path`;
- the same `data.processed_dir`;
- the same train/valid/test splits;
- the same target relation;
- the same candidate entity type (`DRUG`);
- the same filtered ranking protocol;
- the same evaluation metrics.

### `disease_to_drug_full`

```text
R-GCN context:
INDICATION + CONTRAINDICATION + HAS_INGREDIENT

Content-Based features:
CONTRAINDICATION + HAS_INGREDIENT
```

The baseline does not use `INDICATION` as a feature because it is the target relation.

### `disease_to_drug_filtered`

```text
R-GCN context:
INDICATION + CONTRAINDICATION

Content-Based features:
CONTRAINDICATION
```

The full graph and filtered graph should be reported as separate experimental settings.

---

## Dependencies

Project environment plus:

- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`

---

## Limitations

- Only supports the biomedical `disease_to_drug_full` setting.
- Does not support WN18RR or FB15k-237.
- Does not support `drug_to_disease` or generic `tail_prediction`.
- Does not learn graph embeddings.
- Relies on the quality and coverage of the selected feature relations.
- It is intended as a reproducible baseline, not as a production recommender.

---

## Example Output

Example output format:

```text
--- [3/3] Final metrics (aligned with R-GCN protocol) ---
Hits@1:  0.4620
Hits@3:  0.5258
Hits@10: 0.5805
MRR:     0.5066
Logged benchmark to results/benchmarks.csv
```

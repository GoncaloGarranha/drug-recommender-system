# R-GCN With Genes — Best Thesis Run

This directory contains the final selected R-GCN experiment for the primary task of this dissertation.

---

# Experiment Overview

| Item | Value |
|------|-------|
| Experiment | `with_genes_best` |
| Status | Final thesis model |
| Model | R-GCN + DistMult |
| Task | Disease-to-Drug Recommendation |
| Dataset | `triples_with_genes.csv` |
| Selection criterion | Highest validation MRR |

---

# Task

Disease-to-drug recommendation.

Target relation:

```text
Disease ── INDICATION ──► Drug
```

The model is trained on the biomedical Knowledge Graph enriched with gene associations.

---

# Dataset

Knowledge Graph:

```text
kg_constructor/data/output/triples/triples_with_genes.csv
```

Processed graph:

```text
models/rgcn/data/processed/disease_to_drug_with_genes/
```

---

# Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Split strategy | Random (80% / 10% / 10%) |
| Target relation | `INDICATION` |
| Query entity type | `DISEASE` |
| Candidate entity type | `DRUG` |
| Inverse edges | Enabled |
| Encoder | R-GCN |
| Decoder | DistMult |
| Embedding dimension | 128 |
| Number of R-GCN layers | 2 |
| Number of basis functions | 4 |
| Dropout | 0.2 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Negative samples | 50 |
| Evaluation frequency | Every 10 epochs |
| Early stopping | Monitor `valid_mrr` (patience = 6000) |
| Device | CPU |

Configuration file:

```text
kg_constructor\configs\config.yaml
```

---

# Context Relations

The graph used during training includes the following relation types:

- `INDICATION`
- `CONTRAINDICATION`
- `HAS_INGREDIENT`
- `ASSOCIATED_WITH_GENE`

---

# Test Results

| Metric | Value |
|--------|------:|
| MRR | **0.6993** |
| Hits@1 | **0.6591** |
| Hits@3 | **0.7252** |
| Hits@10 | **0.7739** |

---

# Semantic Evaluation

This experiment also includes a semantic evaluation based on pharmacological similarity.

Metrics are available in:

```text
semantic_evaluation/
```

The semantic evaluation reports:

- Raw MRR
- Semantic MRR
- Semantic MRR (excluding exact matches)
- Mean Best Jaccard
- Semantic Hits@1
- Semantic Hits@3
- Semantic Hits@10

---

# Directory Contents

```text
checkpoints/
semantic_evaluation/
splits/

best_metrics.json
config.yaml
environment.json
final_metrics_disease_to_drug_test.json
git_commit.txt
metrics_test.json
metrics_valid.json
model_summary.json
split_summary.json
train_log.json
```

---

# Reproducing the Evaluation

Evaluate this run on the test split:

```powershell
python -m models.rgcn.src.evaluate `
  --run_dir results/final/rgcn/with_genes_best `
  --split test
```

Run the semantic evaluation:

```powershell
python -m models.rgcn.src.semantic_evaluation `
  --run_dir results/final/rgcn/with_genes_best `
  --split test
```

---

# Notes

- This is the primary experiment reported in the dissertation.
- It represents the best-performing R-GCN model obtained during the experimental evaluation.
- The model was selected according to the highest validation MRR.
- The final reported performance corresponds to evaluation on the held-out test split.
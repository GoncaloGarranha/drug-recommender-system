# R-GCN With Genes — Best Thesis Run

This directory contains the final selected R-GCN experiment for the primary task of this thesis.

## Task

**Disease-to-Drug Recommendation**

```text
Disease → INDICATION → Drug
```

This experiment uses the biomedical Knowledge Graph enriched with gene-related information.

---

## Dataset

```text
kg_constructor/data/output/triples/triples_with_genes.csv
```

---

## Model

- **Encoder:** Relational Graph Convolutional Network (R-GCN)
- **Decoder:** DistMult
- **Task:** Link Prediction
- **Prediction type:** Tail Prediction

---

## Configuration

Configuration file:

```text
models/rgcn/configs/biomedical/disease_to_drug_with_genes.yaml
```

Main settings:

| Parameter | Value |
|-----------|-------|
| Random seed | 42 |
| Dataset split | Random (80% / 10% / 10%) |
| Target relation | `INDICATION` |
| Query entity type | `DISEASE` |
| Candidate entity type | `DRUG` |

### Context relations

- `INDICATION`
- `CONTRAINDICATION`
- `HAS_INGREDIENT`
- `ASSOCIATED_WITH_GENE`

---

## Test Results

| Metric | Value |
|--------|------:|
| MRR | **0.6993** |
| Hits@1 | **0.6591** |
| Hits@3 | **0.7252** |
| Hits@10 | **0.7739** |

---

## Reproducing the Evaluation

```powershell
python -m models.rgcn.src.evaluate `
  --run_dir results/final/rgcn/with_genes_best `
  --split test
```

---

## Run Contents

```text
checkpoints/
semantic_evaluation/
splits/
best_metrics.json
config.yaml
environment.json
final_metrics_disease_to_drug_test.json
metrics_test.json
metrics_valid.json
model_summary.json
split_summary.json
train_log.json
```

---

## Notes

- This is the **primary experiment** reported in the dissertation.
- The model was trained on the biomedical Knowledge Graph enriched with gene associations.
- Evaluation was performed on the held-out test split using filtered ranking metrics.
- A complementary semantic evaluation based on pharmacological similarity is available in the `semantic_evaluation/` directory.
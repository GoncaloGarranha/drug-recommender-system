# R-GCN configuration guide

This directory contains the YAML configuration files used to prepare, train, and evaluate R-GCN models.

The configurations are grouped by purpose:

```text
models/rgcn/configs/
├── benchmarks/
├── biomedical/
│   ├── archive/
│   ├── main/
│   └── relation_studies/
```

## 1. Main biomedical configurations

These are the main configurations for the biomedical drug recommendation task.

| Config | Experiment | Dataset | Decoder | Purpose |
|---|---|---|---|---|
| `biomedical/main/disease_to_drug_with_targets.yaml` | `with_dgidb` | Base KG | DistMult | Main baseline on the base KG with DGIdb drug-gene relations |
| `biomedical/main/disease_to_drug_with_targets_plus_3_months.yaml` | `with_dgidb_plus_3_months` | KG + 3 months | DistMult | Main experiment on the extended KG |
| `biomedical/main/disease_to_drug_with_targets_plus_3_months_complex.yaml` | `with_dgidb_plus_3_months_complex` | KG + 3 months | ComplEx | Decoder experiment using ComplEx |

The main prediction task is:

```text
Disease -> Drug recommendation
Target relation: INDICATION
```

Internally, the KG stores target triples as:

```text
Drug --INDICATION--> Disease
```

The model evaluates the inverse recommendation task by ranking candidate drugs for each disease.

## 2. Relation studies / ablations

These configurations are used to test the contribution of specific relation types.

| Config | Experiment | Removed relation(s) | Purpose |
|---|---|---|---|
| `biomedical/relation_studies/no_contraindications.yaml` | `with_dgidb_no_contraindications` | `CONTRAINDICATION` | Tests whether contraindication-like relations add signal or noise |
| `biomedical/relation_studies/no_ingredients.yaml` | `with_dgidb_no_ingredients` | `HAS_INGREDIENT` | Tests the importance of ingredient/substance information |
| `biomedical/relation_studies/no_gene_relations.yaml` | `with_dgidb_no_gene_relations` | `INTERACTS_WITH_GENE`, `ASSOCIATED_WITH_GENE` | Tests the contribution of gene-related information |

Some relation-study configurations currently point to the base KG output path:

```text
kg_constructor/data/output/triples/triples_with_dgidb.csv
```

For experiments on the `+3 months` KG, create explicit `plus_3_months` versions instead of reusing these files directly.

## 3. Benchmark configurations

The `benchmarks/` directory contains configurations for standard link prediction datasets and external biomedical benchmarks.

| Config group | Task | Purpose |
|---|---|---|
| `fb15k*.yaml` | Tail prediction | Standard KG benchmark comparison |
| `wn18*.yaml`, `wn18rr*.yaml` | Tail prediction | Standard KG benchmark comparison |

These are not the main thesis drug recommendation experiments. They are mainly useful for sanity checks and external comparison.

## 4. Archived biomedical configurations

The `biomedical/archive/` directory contains older configurations used during development.

These files are kept for reproducibility and historical comparison, but should not be used as the primary experimental setup.

Examples include:

```text
disease_to_drug.yaml
disease_to_drug_freq200.yaml
disease_to_drug_freq200_3layers.yaml
disease_to_drug_with_genes.yaml
disease_to_drug_with_genes_fair.yaml
```

The current main configurations should be preferred over archived files.

## 5. Common configuration fields

Most biomedical configurations contain the following sections:

```yaml
seed: 42

data:
  triples_path: ...
  processed_dir: ...

task:
  name: disease_to_drug
  target_relation: INDICATION
  query_entity_type: DISEASE
  candidate_entity_type: DRUG
  context_relations:
    - INDICATION
    - CONTRAINDICATION
    - HAS_INGREDIENT
    - ASSOCIATED_WITH_GENE
    - INTERACTS_WITH_GENE

training:
  emb_dim: 128
  num_layers: 2
  num_bases: 5
  neg_k: 50
  negative_sampling: mixed
  hard_negative_ratio: 0.5

runs:
  out_dir: results/runs/rgcn
  experiment_name: ...
```

## 6. Typical workflow

Prepare graph artifacts:

```bash
python -m models.rgcn.src.prepare_graph \
  --config models/rgcn/configs/biomedical/main/disease_to_drug_with_targets_plus_3_months.yaml \
  --force
```

Validate the prepared graph:

```bash
python -m models.rgcn.src.load_data \
  --processed_dir models/rgcn/data/processed/disease_to_drug_with_dgidb_plus_3_months
```

Train:

```bash
python -m models.rgcn.src.train \
  --config models/rgcn/configs/biomedical/main/disease_to_drug_with_targets_plus_3_months.yaml
```

Evaluate full-ranking performance:

```bash
python -m models.rgcn.src.evaluate \
  --run_dir results/runs/rgcn/biomedical/disease_to_drug/with_dgidb_plus_3_months/seed_42 \
  --split test
```

Evaluate sampled negative-set performance:

```bash
python -m models.rgcn.src.negative_set_evaluation \
  --run_dir results/runs/rgcn/biomedical/disease_to_drug/with_dgidb_plus_3_months/seed_42 \
  --split test \
  --num_negatives 50 \
  --negative_mode mixed \
  --hard_negative_ratio 0.5
```

## 7. Recommended main experiments

For the current project state, the most relevant configurations are:

```text
biomedical/main/disease_to_drug_with_targets.yaml
biomedical/main/disease_to_drug_with_targets_plus_3_months.yaml
biomedical/main/disease_to_drug_with_targets_plus_3_months_complex.yaml
```

The main fair comparison should be:

```text
base KG corrected
vs
+3 months KG corrected
```

Both should use the same preprocessing assumptions, entity typing, split strategy, and training hyperparameters.

## 8. Notes on external indications

External indication sources are not currently included in the main experiments.

If future work adds new indication evidence, it should be added as separate relation types, for example:

```text
INDICATION_EXTERNAL
INDICATION_LABEL
INDICATION_CURATED
```

These should not be merged directly into `INDICATION` unless the target definition and train/validation/test split protocol are explicitly redesigned to avoid leakage.

## 9. Notes on contraindications

The `CONTRAINDICATION` relation may contain heterogeneous clinical mentions, including contraindications, adverse effects, warnings, precautions, or disease mentions in drug labels.

For this reason, configurations excluding `CONTRAINDICATION` should be used as ablations to test whether this relation improves or harms performance.

## 10. Notes on gene relations

The current KG includes drug-gene information through:

```text
Drug --INTERACTS_WITH_GENE--> Gene
```

A future biologically motivated extension is to add disease-gene associations:

```text
Disease --ASSOCIATED_WITH_GENE--> Gene
```

This would allow the model to exploit paths of the form:

```text
Drug -> Gene <- Disease
```

These paths are more directly aligned with biomedical mechanisms and explainable drug recommendation.

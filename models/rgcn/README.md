# R-GCN for Knowledge Graph Link Prediction

This module provides an **R-GCN + DistMult benchmark reimplementation** for link prediction over multi-relational knowledge graphs.

The main use case is biomedical drug recommendation, where the model predicts drug–disease relationships from a biomedical Knowledge Graph. The same benchmark reimplementation also supports external datasets such as **WN18RR** and **FB15k-237** through a generic `tail_prediction` task.

The R-GCN learns vector representations of entities by propagating information through typed relations in the graph and uses a DistMult decoder to score candidate triples.

---

## Supported Tasks

The module currently supports three tasks:

### 1. `disease_to_drug_full`

Biomedical disease-to-drug recommendation over the full graph.

Task name:

```text
disease_to_drug
```

Given a disease, rank candidate drugs.

```text
Disease → Drug
```

Target relation:

```text
INDICATION
```

This is the main task for the biomedical recommender system.

---

### 2. `disease_to_drug_filtered`

Biomedical disease-to-drug recommendation over the filtered graph.

Task name:

```text
disease_to_drug
```

Given a disease, rank candidate drugs.

```text
Disease → Drug
```

Target relation:

```text
INDICATION
```

This variant excludes `HAS_INGREDIENT` and is useful for ablation studies.

---

### 3. `drug_to_disease`

Given a drug, rank candidate diseases.

```text
Drug → Disease
```

Target relation:

```text
INDICATION
```

This task uses the same biomedical triple schema as the `disease_to_drug_full` / `disease_to_drug_filtered` variants, but reverses the ranking direction.

---

### 4. `tail_prediction`

Generic Knowledge Graph link prediction task.

```text
(head, relation, ?) → tail
```

This is used for external benchmark datasets such as:

- WN18RR;
- FB15k-237.

Note: the current benchmark implementation evaluates **tail prediction only**. For direct comparison with most published WN18RR/FB15k-237 results, head prediction should also be implemented and combined with tail prediction.

---

## Project Structure

```text
models/rgcn/
│
├── configs/
│   ├── disease_to_drug.yaml
│   ├── drug_to_disease.yaml
│   ├── wn18rr.yaml
│   └── fb15k237.yaml
│
├── data/
│   └── processed/
│       ├── disease_to_drug_full/
│       ├── disease_to_drug_filtered/
│       ├── wn18rr/
│       └── fb15k237/
│
├── runs/
│   ├── wn18rr/
│   ├── fb15k237/
│   └── ...
│
└── src/
    ├── 00_prepare_graph.py
    ├── 01_load_data.py
    ├── 02_train.py
    ├── 03_evaluate.py
    └── 04_infer.py
```

Shared utilities used by this module are located in:

```text
models/shared/
```

---

## Pipeline Overview

The R-GCN pipeline consists of five main stages:

### 1. Prepare Graph

```text
00_prepare_graph.py
```

Converts canonical triples into indexed graph artefacts.

Input:

```csv
src,rel,dst
```

or, for benchmarks with official splits:

```csv
src,rel,dst,split
```

Output examples:

```text
triples_indexed.csv
entity2id.json
rel2id.json
entity_types.json
graph_metadata.json
```

---

### 2. Validate Graph Data

```text
01_load_data.py
```

Loads and validates processed graph artefacts and prints graph statistics.

---

### 3. Train Model

```text
02_train.py
```

Trains an R-GCN encoder with a DistMult decoder for link prediction.

The best checkpoint is selected using validation MRR.

---

### 4. Evaluate Model

```text
03_evaluate.py
```

Evaluates a trained model using filtered ranking metrics.

Metrics are saved inside the run directory and logged to:

```text
benchmarks.csv
```

---

### 5. Run Inference

```text
04_infer.py
```

Generates ranked predictions for a given query entity.

This is mainly intended for biomedical inference.

---

## How the Model Works

The architecture combines three main components:

1. R-GCN encoder;
2. DistMult decoder;
3. Negative sampling and filtered ranking evaluation.

---

## R-GCN Encoder

The encoder learns node representations by aggregating information from neighbouring nodes while considering relation types.

For each relation type, the model learns a relation-specific transformation.

```text
h_i^(l+1) = σ( Σ_r Σ_{j∈N_i^r} (1 / c_{i,r}) W_r^(l) h_j^(l) + W_0^(l) h_i^(l) )
```

Where:

- `h_i^(l)` is the embedding of node `i` at layer `l`;
- `N_i^r` is the set of neighbours connected to node `i` through relation `r`;
- `W_r` is a relation-specific transformation matrix;
- `W_0` is the self-loop transformation;
- `c_{i,r}` is a normalization constant.

Basis decomposition may be used to reduce the number of relation-specific parameters.

---

## DistMult Decoder

The decoder scores triples using a bilinear scoring function:

```text
score(h, r, t) = hᵀ R_r t
```

Where:

- `h` is the source entity embedding;
- `t` is the target entity embedding;
- `R_r` is the relation embedding.

Higher scores indicate more likely relationships.

---

## Negative Sampling

During training, artificial negative examples are created by corrupting triples.

Example for the biomedical full-graph variant `disease_to_drug_full`:

```text
Positive:
(DRUG:ALBUTEROL, INDICATION, DOID:2841)

Negative:
(DRUG:IBUPROFEN, INDICATION, DOID:2841)
```

For benchmark `tail_prediction`, the negative candidate is generated by corrupting the tail entity:

```text
Positive:
(head, relation, tail)

Negative:
(head, relation, corrupted_tail)
```

Known positives are excluded from negative sampling where applicable.

---

## Filtered Ranking

During evaluation, known true triples are removed from candidate rankings so the model is not penalized for ranking another valid triple highly.

For the biomedical `disease_to_drug_full` and `disease_to_drug_filtered` variants, filtered ranking is applied per disease.

For benchmark `tail_prediction`, filtered ranking is applied per `(head, relation)` pair.

---

## Biomedical Dataset

The biomedical Knowledge Graph has two main triple files.

### `disease_to_drug_full`

```text
kg_constructor/data/output/triples/triples.csv
```

Contains:

- `INDICATION`
- `CONTRAINDICATION`
- `HAS_INGREDIENT`

This is the main setting when ingredient information is used.

### `disease_to_drug_filtered`

```text
kg_constructor/data/output/triples/triples.filtered.csv
```

Contains:

- `INDICATION`
- `CONTRAINDICATION`

This setting excludes `HAS_INGREDIENT` and is useful for ablation studies.

---

## Benchmark Datasets

The module supports benchmark datasets converted by the `kg_constructor`.

### WN18RR

Input to R-GCN:

```text
kg_constructor/data/output/benchmarks/wn18rr/triples.csv
```

### FB15k-237

Input to R-GCN:

```text
kg_constructor/data/output/benchmarks/fb15k237/triples.csv
```

Both files are expected to contain:

```csv
src,rel,dst,split
```

The `split` column is used to preserve official benchmark splits.

---

## Configuration

Configuration files are located in:

```text
models/rgcn/configs/
```

---

## Example Configuration — Biomedical Full Graph (`disease_to_drug_full`)

```yaml
seed: 42

data:
  triples_path: "kg_constructor/data/output/triples/triples.csv"
  processed_dir: "models/rgcn/data/processed/disease_to_drug_full"

task:
  name: "disease_to_drug"
  target_relation: "INDICATION"
  query_entity_type: "DISEASE"
  candidate_entity_type: "DRUG"
  context_relations:
    - "INDICATION"
    - "CONTRAINDICATION"
    - "HAS_INGREDIENT"

split:
  mode: "random"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  max_epochs: 4000
  lr: 0.001
  weight_decay: 0.0001
  dropout: 0.2
  emb_dim: 128
  num_bases: 30
  num_layers: 2
  neg_k: 50
  eval_every: 10
  hits_k: [1, 3, 10]
  device: "cpu"

early_stopping:
  monitor: "valid_mrr"
  patience: 30
  min_delta: 0.0005
  mode: "max"

runs:
  out_dir: "models/rgcn/runs"
```

---

## Example Configuration — WN18RR / FB15k-237

```yaml
seed: 42

data:
  triples_path: "kg_constructor/data/output/benchmarks/wn18rr/triples.csv"
  processed_dir: "models/rgcn/data/processed/wn18rr"

task:
  name: "tail_prediction"
  target_relation: null
  query_entity_type: "ALL"
  candidate_entity_type: "ALL"
  context_relations: "ALL"

split:
  mode: "official"

training:
  max_epochs: 1000
  lr: 0.001
  weight_decay: 0.0001
  dropout: 0.2
  emb_dim: 128
  num_bases: 30
  num_layers: 2
  neg_k: 20
  eval_every: 10
  hits_k: [1, 3, 10]
  device: "cpu"

early_stopping:
  monitor: "valid_mrr"
  patience: 30
  min_delta: 0.0005
  mode: "max"

runs:
  out_dir: "models/rgcn/runs/wn18rr"
```

For FB15k-237, change:

```yaml
data:
  triples_path: "kg_constructor/data/output/benchmarks/fb15k237/triples.csv"
  processed_dir: "models/rgcn/data/processed/fb15k237"

runs:
  out_dir: "models/rgcn/runs/fb15k237"
```

---

## Commands

All commands should be executed from the repository root.

---

## Biomedical Full Graph (`disease_to_drug_full`)

### 1. Prepare graph artefacts

```bash
python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/disease_to_drug.yaml --force
```

### 2. Validate graph artefacts

```bash
python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/disease_to_drug_full
```

### 3. Train

```bash
python -m models.rgcn.src.02_train --config models/rgcn/configs/disease_to_drug.yaml
```

### 4. Evaluate

```bash
python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs --split test
```

### 5. Inference

```bash
python -m models.rgcn.src.04_infer --query "Albuterol" --topk 10
```

or:

```bash
python -m models.rgcn.src.04_infer --entity_id "DRUG:ALBUTEROL SULFATE" --topk 10
```

---

## Biomedical Filtered Graph (`disease_to_drug_filtered`)

The command sequence is the same as the full graph, but the config should
point `data.processed_dir` to `models/rgcn/data/processed/disease_to_drug_filtered`.

---

## WN18RR

Before running these commands, the benchmark must be converted by `kg_constructor`.

### 1. Prepare graph artefacts

```bash
python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/wn18rr.yaml --force
```

### 2. Validate graph artefacts

```bash
python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/wn18rr
```

### 3. Train

```bash
python -m models.rgcn.src.02_train --config models/rgcn/configs/wn18rr.yaml
```

### 4. Evaluate

```bash
python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs/wn18rr --split test
```

---

## FB15k-237

Before running these commands, the benchmark must be converted by `kg_constructor`.

### 1. Prepare graph artefacts

```bash
python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/fb15k237.yaml --force
```

### 2. Validate graph artefacts

```bash
python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/fb15k237
```

### 3. Train

```bash
python -m models.rgcn.src.02_train --config models/rgcn/configs/fb15k237.yaml
```

### 4. Evaluate

```bash
python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs/fb15k237 --split test
```

---

## Smoke Tests

For quick technical validation, reduce the training configuration:

```yaml
training:
  max_epochs: 3
  eval_every: 1
  emb_dim: 32
  num_layers: 1
  num_bases: 10
  neg_k: 2
  device: "cpu"
```

Smoke tests are only intended to verify that the pipeline runs end-to-end. They should not be reported as final results.

---

## Evaluation Metrics

The model reports:

- `MRR`
- `Hits@1`
- `Hits@3`
- `Hits@10`

Evaluation uses filtered ranking.

Results are logged to:

```text
benchmarks.csv
```

and saved in the corresponding run directory.

---

## Results Tracking

Each evaluation run appends metrics to:

```text
benchmarks.csv
```

at the repository root.

This replaces the previous:

```text
models/performance_leaderboard.csv
```

---

## Fair Comparison Requirements

For valid comparison between models, ensure that all models use:

- the same triples file;
- the same train/validation/test split;
- the same target relation or task;
- the same candidate pool;
- the same filtered ranking protocol;
- the same metrics.

For the biomedical full graph setting:

```text
R-GCN context:
INDICATION + CONTRAINDICATION + HAS_INGREDIENT
```

A Content-Based baseline should use:

```text
CONTRAINDICATION + HAS_INGREDIENT
```

because `INDICATION` is the target relation and should not be used as a direct feature.

---

## Experimental Protocol

For final experiments:

- use fixed splits;
- select the best checkpoint using validation MRR;
- evaluate only the best checkpoint on the test set;
- run multiple seeds when possible;
- report mean and standard deviation;
- use early stopping where appropriate;
- log all results to `benchmarks.csv`.

Where datasets differ significantly in size and structure, prefer a common protocol with validation-based model selection rather than forcing the same number of epochs.

---

## Limitations

- Benchmark evaluation currently supports tail prediction only.
- Direct comparison with published WN18RR/FB15k-237 results requires head + tail prediction.
- Training with `candidate_entity_type: ALL` can be slow on CPU.
- Hyperparameters may need dataset-specific tuning based on validation performance.

---

## Troubleshooting

### Very slow benchmark training

WN18RR and FB15k-237 use:

```yaml
candidate_entity_type: "ALL"
```

This makes ranking more expensive than biomedical disease-to-drug recommendation.

For quick tests, reduce:

```yaml
training:
  max_epochs: 3
  emb_dim: 32
  neg_k: 2
```

### Module import errors

Run commands from the repository root.

If needed, set `PYTHONPATH`:

PowerShell:

```powershell
$env:PYTHONPATH = "D:\path\to\drug-recommender-system"
```

macOS/Linux:

```bash
export PYTHONPATH=/path/to/drug-recommender-system
```

---

## Output Files

Each training run creates a structured directory under `runs.out_dir`, for example:

```text
runs/biomedical/rgcn/disease_to_drug/seed_42/
```

The standard run files are:

```text
checkpoints/best.pt
config.yaml
metrics_valid.json
metrics_test.json
train_log.json
split_summary.json
model_summary.json
environment.json
git_commit.txt
```

The best checkpoint is selected according to validation MRR.

---

## Citation

If you use this implementation, please cite:

```bibtex
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N. and Bloem, Peter and Van Den Berg, Rianne and Titov, Ivan and Welling, Max},
  booktitle={European Semantic Web Conference},
  pages={593--607},
  year={2018}
}
```
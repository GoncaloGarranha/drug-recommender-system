# Drug Recommender System

This project develops a knowledge graph-based recommendation system for biomedical drug discovery, with a focus on drug-disease relationship prediction and drug repositioning.

The system builds and processes biomedical knowledge graphs, trains relational learning models such as R-GCN (Relational Graph Convolutional Networks), and evaluates link prediction performance using standard ranking metrics.

The project currently supports:

- A custom biomedical knowledge graph;
- WN18RR benchmark dataset;
- FB15k-237 benchmark dataset;
- R-GCN model;
- Content-Based baseline;
- Shared evaluation utilities;
- Centralized summary logging in `results/benchmarks.csv`, with full per-run artifacts stored under `results/runs/`.

---

## Project Structure

```text
.
├── kg_constructor/              # Knowledge graph construction and benchmark conversion
│   ├── configs/                 # KG and benchmark conversion configs
│   ├── data/                    # Input and output triples
│   └── src/                     # KG construction and conversion source code
│
├── models/
│   ├── rgcn/                    # R-GCN model for link prediction
│   │   ├── configs/             # Model configurations
│   │   ├── data/processed/      # Processed graph artefacts
│   │   └── src/                 # Training, evaluation and inference scripts
│   │
│   ├── content-based/           # Content-based baseline
│   │   └── src/
│   │
│   └── shared/                  # Shared utilities across models
│       ├── config.py
│       ├── io.py
│       ├── metrics.py
│       ├── ranking.py
│       ├── splits.py
│       └── benchmark_tracker.py
│
├── datasets/                    # Raw benchmark datasets, when used locally
│   ├── wn18rr/
│   └── fb15k237/
│
├── results/
│   ├── benchmarks.csv           # Summary table for consolidated results
│   ├── final/                   # Final reports and summaries
│   ├── runs/                    # Structured per-run artifacts
│   └── figures/                 # Generated figures and plots
├── requirements.txt
└── README.md
```

---

## System Overview

The project is organized into two main blocks:

```text
Raw / Structured Data
        ↓
kg_constructor
        ↓
Canonical triples
        ↓
models
        ↓
Evaluation and recommendation
```

### `kg_constructor`

Responsible for producing canonical triples.

For the biomedical dataset, it builds drug-disease and drug-ingredient relations from structured biomedical data.

For benchmark datasets, it converts official splits (`train.txt`, `valid.txt`, `test.txt`) into the canonical CSV format expected by the models.

### `models`

Responsible for training, evaluating and comparing models.

Current models:

- R-GCN;
- Content-Based baseline.

Shared code for metrics, splits, ranking, I/O and benchmark tracking is located in:

```text
models/shared/
```

---

## Datasets

### Biomedical Knowledge Graph

The biomedical dataset is represented as triples in the following format:

```csv
src,rel,dst
```

The project preserves two canonical graph builds with different temporal
coverage.

| Build | Months included | Coverage | Purpose | Triples |
|---|---|---:|---|---:|
| `2025-10_to_2025-12` | October, November and December 2025 | 3 months | Historical reference build | 40746 |
| `2025-10_to_2026-03` | October 2025 to March 2026 | 6 months | Main build used in the final experiments | 49903 |

The six-month build is cumulative. It contains the data from October,
November and December 2025, extended with January, February and March 2026.

#### Historical reference build

```text
kg_constructor/data/builds/2025-10_to_2025-12/triples/triples_with_dgidb.csv
```

#### Main biomedical build

```text
kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv
```

The main graph contains the following relations:

| Relation | Triples |
|---|---:|
| `INDICATION` | 7216 |
| `CONTRAINDICATION` | 2752 |
| `HAS_INGREDIENT` | 24978 |
| `ASSOCIATED_WITH_GENE` | 10423 |
| `INTERACTS_WITH_GENE` | 4534 |

`INDICATION` is the target relation for disease-to-drug recommendation.
The remaining relations provide contextual information during message
passing.

The main graph is also used in the relation studies. These experiments do not
create separate filtered triple files. Instead, they reuse the same canonical
graph and data splits while excluding one relation at a time through the
`context_relations` field of the corresponding configuration.

The active biomedical configurations are located in:

```text
models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/
├── main/
│   └── all_supported_relations.yaml
└── relation_studies/
    ├── without_associated_with_gene.yaml
    ├── without_contraindication.yaml
    ├── without_has_ingredient.yaml
    └── without_interacts_with_gene.yaml
```

Further details about the datasets, manifests and graph provenance are
available in:

```text
kg_constructor/data/README.md
```

---

### WN18RR

WN18RR is supported as a standard link-prediction benchmark.

Expected raw structure:

```text
kg_constructor/data/benchmarks/wn18rr/raw/
├── train.txt
├── valid.txt
└── test.txt
```

Converted triples are stored in:

```text
kg_constructor/data/benchmarks/wn18rr/triples.csv
```

with the following format:

```csv
src,rel,dst,split
```

The `split` column preserves the original training, validation and test
partitions.

---

### FB15k-237

FB15k-237 is also supported as a standard link-prediction benchmark.

Expected raw structure:

```text
kg_constructor/data/benchmarks/fb15k237/raw/
├── train.txt
├── valid.txt
└── test.txt
```

Converted triples are stored in:

```text
kg_constructor/data/benchmarks/fb15k237/triples.csv
```

with the following format:

```csv
src,rel,dst,split
```

The `split` column preserves the original training, validation and test
partitions.

---

## Tasks Supported

### `disease_to_drug`

Given a disease, rank candidate drugs according to their predicted
`INDICATION` score.

```text
Disease → Drug
```

Task definition:

```text
Target relation:       INDICATION
Query entity type:     DISEASE
Candidate entity type: DRUG
```

The main configuration is:

```text
models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml
```

The processed graph artefacts and data splits are stored in:

```text
models/rgcn/data/processed/disease_to_drug/six_months_main_random_split/
```

The main experiment and all relation studies reuse these same processed
artefacts and splits, ensuring that their results are directly comparable.

---

### Relation studies

The relation studies evaluate the contribution of individual contextual
relations by removing one relation at a time:

- `without_contraindication`;
- `without_has_ingredient`;
- `without_associated_with_gene`;
- `without_interacts_with_gene`.

Only the relation used during message passing changes. The target relation,
candidate set and training, validation and test splits remain unchanged.

---

### `tail_prediction`

Generic tail-prediction task used for benchmark datasets:

```text
(head, relation, ?) → tail
```

This task is used for datasets such as WN18RR and FB15k-237.

The current benchmark evaluation focuses on tail prediction. A direct
comparison with benchmark results that report the average of head and tail
prediction would also require implementing the corresponding head-prediction
evaluation.

---

## Quick Start

### 1. Environment Setup

#### Create virtual environment

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
python -m venv venv
venv\Scripts\activate.bat
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

#### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Biomedical Dataset Workflow

### 1. Build the biomedical knowledge graph

```bash
python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml
```

This generates the biomedical triple files, including:

```text
kg_constructor/data/output/triples/triples.csv
kg_constructor/data/output/triples/triples.filtered.csv
```

---

### 2. Prepare graph artefacts for R-GCN

Example using the full biomedical graph (`disease_to_drug_full`):

```bash
python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/disease_to_drug.yaml --force
```

This creates indexed artefacts in the configured `processed_dir`, for example:

```text
models/rgcn/data/processed/disease_to_drug_full/
```

---

### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/disease_to_drug_full
```

---

### 4. Train R-GCN

```bash
python -m models.rgcn.src.train --config models/rgcn/configs/disease_to_drug.yaml
```

---

### 5. Evaluate R-GCN

```bash
python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```

Evaluation metrics are saved in the run directory and logged to:

```text
results/benchmarks.csv
```

---

### 6. Run Content-Based baseline

```bash
python -m models.content-based.src.baseline --config models/rgcn/configs/disease_to_drug.yaml --split test
```

The baseline uses the same config, splits and evaluation protocol as the R-GCN.

---

### Biomedical filtered graph (`disease_to_drug_filtered`)

For the filtered biomedical setting, use the same task name (`disease_to_drug`)
but point the processed artefacts to:

```text
models/rgcn/data/processed/disease_to_drug_filtered/
```

The corresponding triple file is:

```text
kg_constructor/data/output/triples/triples.filtered.csv
```

This setting excludes `HAS_INGREDIENT` and is mainly used for ablation studies.

---

## Benchmark Dataset Workflow

### WN18RR

#### 1. Convert benchmark splits

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/wn18rr.yaml --force
```

Expected output:

```text
kg_constructor/data/benchmarks/wn18rr/triples.csv
```

#### 2. Prepare graph artefacts

```bash
python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/wn18rr.yaml --force
```

#### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/wn18rr
```

#### 4. Train

```bash
python -m models.rgcn.src.train --config models/rgcn/configs/wn18rr.yaml
```

#### 5. Evaluate

```bash
python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```

---

### FB15k-237

#### 1. Convert benchmark splits

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/fb15k237.yaml --force
```

Expected output:

```text
kg_constructor/data/benchmarks/fb15k237/triples.csv
```

#### 2. Prepare graph artefacts

```bash
python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/fb15k237.yaml --force
```

#### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/fb15k237
```

#### 4. Train

```bash
python -m models.rgcn.src.train --config models/rgcn/configs/fb15k237.yaml
```

#### 5. Evaluate

```bash
python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```


### Schlichtkrull-style R-GCN link prediction

A standalone article-style R-GCN + DistMult implementation lives in:

```text
models/rgcn_benchmark/
```

This module is intentionally **link prediction only** and is separate from the biomedical `models/rgcn` pipeline. It supports WN18, WN18RR, FB15k and FB15k-237 with filtered head + tail evaluation. See `models/rgcn_benchmark/README.md` for commands and configuration details.

---

## Inference

For the main biomedical model, a disease name is sufficient. By default, known
`INDICATION` associations are excluded and only unobserved pairs are ranked:

```bash
python -m models.rgcn.src.infer --query "asthma" --topk 10
```

The designated `disease_to_drug` run is selected automatically. Use
`--run_dir` only to select a different trained run.

When an exact disease name is not found, an interactive terminal shows up to
five close matches. Select one by number, enter `R` to refine the query, or
enter `0` to cancel. For example, `--query "astma"` suggests `asthma` and its
closest related labels. Non-interactive executions return the suggestions as
an error instead of waiting for input. Exact canonical IDs remain available
through `--entity_id`. If no sufficiently close disease exists, the command
returns a short explanation that queries are limited to entities represented
in the current graph.

Audit the ranking and distinguish associations from each experimental split:

```bash
python -m models.rgcn.src.infer \
  --run_dir results/runs/rgcn/biomedical/disease_to_drug/hyperparam_combined_best/seed_42 \
  --query "type 2 diabetes" \
  --mode audit \
  --topk 10
```

Use the canonical ID when a name is ambiguous:

```bash
python -m models.rgcn.src.infer \
  --run_dir results/runs/rgcn/biomedical/disease_to_drug/hyperparam_combined_best/seed_42 \
  --entity_id "DOID:9352" \
  --mode audit \
  --topk 10
```

Exclude every `INDICATION` association present in the train, validation, or
test split, remove pairs with a recorded `CONTRAINDICATION`, and save the
remaining ranked candidates to JSON:

```bash
python -m models.rgcn.src.infer \
  --run_dir results/runs/rgcn/biomedical/disease_to_drug/hyperparam_combined_best/seed_42 \
  --entity_id "DOID:9352" \
  --exclude_contraindications \
  --topk 10 \
  --json_out predictions.json
```

`audit` reports `TRAIN`, `VALIDATION`, `TEST`, or `UNOBSERVED` for every
candidate and includes the evaluation-compatible filtered rank for validation
and test associations. `exploration` is the default and only returns
`UNOBSERVED` pairs. Use `--details` to expose raw scores and internal fields.
Scores are raw
DistMult ranking values; they are not calibrated probabilities or evidence of
clinical efficacy. `UNOBSERVED` only means that the pair has no `INDICATION`
triple in the current graph; it does not establish that the association is
novel or clinically valid. Contraindications are not shown as a terminal output
column. The JSON field records whether the pair also has a `CONTRAINDICATION`
relation in the graph, and `--exclude_contraindications` can be used as an
explicit structural filter. The JSON output also records every original name associated
with each canonical entity so that synonyms, brands, formulations, and possible
mapping problems can be inspected separately.

---

## Results Tracking

All evaluated experiments are logged to:

```text
results/benchmarks.csv
```

This file is the central results table for model comparison.

It replaces the previous:

```text
models/performance_leaderboard.csv
```

Typical logged metrics:

- MRR
- Hits@1
- Hits@3
- Hits@10
- model name
- dataset/task
- split
- run path

---

## Metrics

The current evaluation metrics are:

- `MRR`
- `Hits@1`
- `Hits@3`
- `Hits@10`

The evaluation uses filtered ranking.

For benchmark datasets, the current implementation supports tail prediction:

```text
(head, relation, ?) → tail
```

For direct comparison with standard literature results, head prediction should also be implemented:

```text
(?, relation, tail) → head
```

and combined with tail prediction.

---

## Fair Model Comparison

For model comparisons to be valid, all models must use:

- the same triples file;
- the same train/validation/test splits;
- the same target relation/task;
- the same candidate pool;
- the same filtered ranking protocol;
- the same evaluation metrics;
- the same available relational information.

Example: full biomedical graph comparison

```text
R-GCN:
INDICATION + CONTRAINDICATION + HAS_INGREDIENT

Content-Based:
CONTRAINDICATION + HAS_INGREDIENT
```

Example: filtered biomedical graph comparison

```text
R-GCN:
INDICATION + CONTRAINDICATION

Content-Based:
CONTRAINDICATION
```

The full graph and the filtered graph should be treated as different experimental settings.

---

## Smoke Tests

For quick technical validation, use small configurations such as:

```yaml
training:
  epochs: 3
  eval_every: 1
  emb_dim: 32
  num_layers: 1
  num_bases: 10
  neg_k: 2
  device: "cpu"
```

Smoke tests only validate that the pipeline runs correctly. They should not be reported as final results.

---

## Recommended Experimental Protocol

For final experiments:

- use fixed splits;
- use validation MRR for model selection;
- keep the best validation checkpoint;
- evaluate the best checkpoint on the test set;
- run multiple seeds when possible;
- report mean and standard deviation;
- log all results to `results/benchmarks.csv`.

Where datasets differ significantly in size, prefer a common training protocol with early stopping rather than forcing the same number of epochs.

---

## Example End-to-End Commands

### Biomedical full graph

```bash
python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml

python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/disease_to_drug.yaml --force

python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/disease_to_drug_full

python -m models.rgcn.src.train --config models/rgcn/configs/disease_to_drug.yaml

python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test

python -m models.content-based.src.baseline --config models/rgcn/configs/disease_to_drug.yaml --split test
```

### WN18RR

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/wn18rr.yaml --force

python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/wn18rr.yaml --force

python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/wn18rr

python -m models.rgcn.src.train --config models/rgcn/configs/wn18rr.yaml

python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```

### FB15k-237

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/fb15k237.yaml --force

python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/fb15k237.yaml --force

python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/fb15k237

python -m models.rgcn.src.train --config models/rgcn/configs/fb15k237.yaml

python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```

---

## Troubleshooting

### Virtual environment issues

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Module import errors

Run commands from the project root.

If needed, set `PYTHONPATH`.

PowerShell:

```powershell
$env:PYTHONPATH = "D:\path\to\drug-recommender-system"
```

macOS/Linux:

```bash
export PYTHONPATH=/path/to/drug-recommender-system
```

### Very slow benchmark training

Benchmark datasets use `candidate_entity_type: ALL`, which can make ranking expensive, especially on CPU.

For quick testing, reduce:

```yaml
training:
  epochs: 3
  emb_dim: 32
  neg_k: 2
```

For final experiments, increase the configuration gradually.

---

## Requirements

- Python 3.8+
- PyTorch
- DGL
- NumPy
- Pandas
- scikit-learn

See `requirements.txt` for the complete list.

---

## Documentation

Additional documentation:

```text
kg_constructor/README.md
models/rgcn/README.md
models/content-based/README.md
docs/
```

---

## License

This project is developed for academic research purposes.

---

## Citation

If you use the R-GCN implementation, please cite:

```bibtex
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N. and Bloem, Peter and Van Den Berg, Rianne and Titov, Ivan and Welling, Max},
  booktitle={European Semantic Web Conference},
  pages={593--607},
  year={2018}
}

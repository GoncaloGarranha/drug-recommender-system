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
- Centralized experiment logging in `benchmarks.csv`.

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
│   │   ├── runs/                # Training outputs and checkpoints
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
├── benchmarks.csv               # Central results table
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

The biomedical dataset is represented through triples in the format:

```csv
src,rel,dst
```

There are currently two main versions:

#### Full graph

```text
kg_constructor/data/output/triples/triples.csv
```

Contains:

- `INDICATION`
- `CONTRAINDICATION`
- `HAS_INGREDIENT`

This is the complete biomedical graph and is the main version when ingredient information is required.

#### Filtered graph

```text
kg_constructor/data/output/triples/triples.filtered.csv
```

Contains:

- `INDICATION`
- `CONTRAINDICATION`

This version excludes `HAS_INGREDIENT` and is useful for ablation studies or experiments focused only on therapeutic relations.

---

### WN18RR

WN18RR is supported as an external benchmark dataset.

Expected raw structure:

```text
kg_constructor/data/input/benchmarks/wn18rr/raw/
├── train.txt
├── valid.txt
└── test.txt
```

After conversion, the output is:

```text
kg_constructor/data/output/benchmarks/wn18rr/triples.csv
```

with format:

```csv
src,rel,dst,split
```

---

### FB15k-237

FB15k-237 is also supported as an external benchmark dataset.

Expected raw structure:

```text
kg_constructor/data/input/benchmarks/fb15k237/raw/
├── train.txt
├── valid.txt
└── test.txt
```

After conversion, the output is:

```text
kg_constructor/data/output/benchmarks/fb15k237/triples.csv
```

with format:

```csv
src,rel,dst,split
```

---

## Tasks Supported

### `disease_to_drug`

Given a disease, recommend candidate drugs.

```text
Disease → Drug
```

Used for the biomedical recommendation task.

Target relation:

```text
INDICATION
```

---

### `drug_to_disease`

Given a drug, predict associated diseases.

```text
Drug → Disease
```

Target relation:

```text
INDICATION
```

---

### `tail_prediction`

Generic benchmark link prediction task.

```text
(head, relation, ?) → tail
```

Used for WN18RR and FB15k-237.

Note: this currently evaluates tail prediction only. For direct comparison with most published benchmark results, head prediction should also be implemented and combined with tail prediction.

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

Example using the full biomedical graph:

```bash
python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/disease_to_drug.yaml --force
```

This creates indexed artefacts in the configured `processed_dir`, for example:

```text
models/rgcn/data/processed/disease_to_drug/
```

---

### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/disease_to_drug
```

---

### 4. Train R-GCN

```bash
python -m models.rgcn.src.02_train --config models/rgcn/configs/disease_to_drug.yaml
```

---

### 5. Evaluate R-GCN

```bash
python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs --split test
```

Evaluation metrics are saved in the run directory and logged to:

```text
benchmarks.csv
```

---

### 6. Run Content-Based baseline

```bash
python -m models.content-based.src.baseline --config models/rgcn/configs/disease_to_drug.yaml --split test
```

The baseline uses the same config, splits and evaluation protocol as the R-GCN.

---

## Benchmark Dataset Workflow

### WN18RR

#### 1. Convert benchmark splits

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/wn18rr.yaml --force
```

Expected output:

```text
kg_constructor/data/output/benchmarks/wn18rr/triples.csv
```

#### 2. Prepare graph artefacts

```bash
python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/wn18rr.yaml --force
```

#### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/wn18rr
```

#### 4. Train

```bash
python -m models.rgcn.src.02_train --config models/rgcn/configs/wn18rr.yaml
```

#### 5. Evaluate

```bash
python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs/wn18rr --split test
```

---

### FB15k-237

#### 1. Convert benchmark splits

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/fb15k237.yaml --force
```

Expected output:

```text
kg_constructor/data/output/benchmarks/fb15k237/triples.csv
```

#### 2. Prepare graph artefacts

```bash
python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/fb15k237.yaml --force
```

#### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/fb15k237
```

#### 4. Train

```bash
python -m models.rgcn.src.02_train --config models/rgcn/configs/fb15k237.yaml
```

#### 5. Evaluate

```bash
python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs/fb15k237 --split test
```

---

## Inference

Generate predictions for a specific drug or disease, depending on the configured task.

Example using a drug query:

```bash
python -m models.rgcn.src.04_infer --query "Albuterol" --topk 10
```

Example using an exact entity ID:

```bash
python -m models.rgcn.src.04_infer --entity_id "DRUG:ALBUTEROL SULFATE" --topk 10
```

Save results to JSON:

```bash
python -m models.rgcn.src.04_infer --query "Metformin" --topk 10 --json_out predictions.json
```

---

## Results Tracking

All evaluated experiments are logged to:

```text
benchmarks.csv
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
- log all results to `benchmarks.csv`.

Where datasets differ significantly in size, prefer a common training protocol with early stopping rather than forcing the same number of epochs.

---

## Example End-to-End Commands

### Biomedical full graph

```bash
python -m kg_constructor.src.run_kg_pipeline --config kg_constructor/configs/config.yaml

python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/disease_to_drug.yaml --force

python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/disease_to_drug

python -m models.rgcn.src.02_train --config models/rgcn/configs/disease_to_drug.yaml

python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs --split test

python -m models.content-based.src.baseline --config models/rgcn/configs/disease_to_drug.yaml --split test
```

### WN18RR

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/wn18rr.yaml --force

python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/wn18rr.yaml --force

python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/wn18rr

python -m models.rgcn.src.02_train --config models/rgcn/configs/wn18rr.yaml

python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs/wn18rr --split test
```

### FB15k-237

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/fb15k237.yaml --force

python -m models.rgcn.src.00_prepare_graph --config models/rgcn/configs/fb15k237.yaml --force

python -m models.rgcn.src.01_load_data --processed_dir models/rgcn/data/processed/fb15k237

python -m models.rgcn.src.02_train --config models/rgcn/configs/fb15k237.yaml

python -m models.rgcn.src.03_evaluate --runs_dir models/rgcn/runs/fb15k237 --split test
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
```
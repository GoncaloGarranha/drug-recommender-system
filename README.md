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

The project preserves two canonical graph builds with different regulatory
data coverage.

| Build identifier | Regulatory data used | Purpose | Triples |
|---|---|---|---:|
| `2025-10_to_2025-12` | DailyMed: July, August and November 2025; Orange Book and Purple Book: November 2025 | Initial three-month build | 44,088 |
| `2025-10_to_2026-03` | DailyMed: July, August, November and December 2025, January and February 2026; Orange Book and Purple Book: February 2026 | Final six-month build used in the experiments | 51,759 |

The directory names are legacy build identifiers retained to preserve the
existing experiment configurations. They do not represent the exact temporal
coverage of the regulatory data.

#### Historical reference build

```text
kg_constructor/data/builds/2025-10_to_2025-12/triples/triples_with_dgidb.csv
```

#### Main biomedical build

```text
kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv
```

The canonical final graph is distributed through
[Zenodo](https://doi.org/10.5281/zenodo.22687285), rather than inside the source
repository. Installation and checksum-verification instructions are provided
in the [Biomedical Dataset Workflow](#biomedical-dataset-workflow).

The main graph contains the following relations:

| Relation | Triples |
|---|---:|
| `HAS_INGREDIENT` | 24,030 |
| `ASSOCIATED_WITH_GENE` | 9,951 |
| `INTERACTS_WITH_GENE` | 9,414 |
| `INDICATION` | 6,143 |
| `CONTRAINDICATION` | 2,221 |

The final graph contains 10,380 entities: 2,347 drugs, 2,278 ingredients,
793 diseases and 4,962 genes.

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
models/rgcn/data/processed/disease_to_drug/six_months_random_split/
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

The repository has a large historical Git archive. For a new source-code
checkout, a shallow clone avoids downloading obsolete generated artefacts:

```bash
git clone --depth 1 \
  https://github.com/GoncaloGarranha/drug-recommender-system.git
cd drug-recommender-system
```

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

### Option A — Use the published biomedical knowledge graph (recommended)

The final six-month biomedical knowledge graph used in the dissertation experiments
is available on Zenodo:

**Dataset DOI:** <https://doi.org/10.5281/zenodo.22687285>

The published graph contains 51,759 triples and 10,380 entities. To reproduce the
model experiments without rebuilding the graph:

1. Download the dataset package from Zenodo.
2. Extract the package and locate `data/triples.csv`.
3. From the repository root, create the destination directory if necessary and
   copy `data/triples.csv` into it, naming the copy `triples_with_dgidb.csv`:

   ```text
   kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv
   ```

4. Verify the SHA-256 checksum of the copied file against the canonical graph
   checksum recorded in the build manifest:

   ```text
   3d5c7df7d021a077b3d5c7fdd986ee6ac1343fec57d585b66d2f622eb387ad72
   ```

   Linux:

   ```bash
   sha256sum kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv
   ```

   macOS:

   ```bash
   shasum -a 256 kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv
   ```

   Windows PowerShell:

   ```powershell
   Get-FileHash kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv -Algorithm SHA256
   ```

5. Continue with **Step 2 — Prepare graph artefacts for R-GCN** below.

This option uses the published canonical graph and does not require the original
MedJsonify inputs, Open Targets or DGIdb source datasets. The model experiments
still require the dependencies and artefacts generated by the subsequent steps.

---

### Option B — Rebuild the biomedical knowledge graph from source data

To reconstruct the graph, first obtain the original MedJsonify inputs and
external datasets, following their documented versions and directory structure:

- [`kg_constructor/data/README.md`](kg_constructor/data/README.md)
- [`kg_constructor/data/external/README.md`](kg_constructor/data/external/README.md)

Then run Step 1 below. This reconstruction is optional when working from the
published canonical graph.

---

### 1. Build the biomedical knowledge graph (Option B only)

```bash
python -m kg_constructor.src.run_kg_pipeline \
  --config kg_constructor/configs/biomedical/periods/2025-10_to_2026-03.yaml
```

This generates the biomedical triple files, including:

```text
kg_constructor/data/builds/2025-10_to_2026-03/triples/triples.filtered.csv
kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_genes.csv
kg_constructor/data/builds/2025-10_to_2026-03/triples/triples_with_dgidb.csv
```

---

### 2. Prepare graph artefacts for R-GCN

Example using the final six-month biomedical graph:

```bash
python -m models.rgcn.src.prepare_graph \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml \
  --force
```

This creates indexed artefacts in the configured `processed_dir`, for example:

```text
models/rgcn/data/processed/disease_to_drug/six_months_random_split/
```

---

### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.load_data \
  --processed_dir models/rgcn/data/processed/disease_to_drug/six_months_random_split
```

---

### 4. Train R-GCN

```bash
python -m models.rgcn.src.train \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml
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
python -m models.content-based.src.baseline \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml \
  --split test
```

The baseline uses the same config, splits and evaluation protocol as the R-GCN.

---

### Relation-study variants

The relation studies reuse the same canonical graph and processed split while
excluding one contextual relation at a time through their configuration. The
maintained configurations are located in:

```text
models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/relation_studies/
```

---

## Benchmark Dataset Workflow

### WN18RR

#### 1. Convert benchmark splits

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/benchmarks/wn18rr.yaml --force
```

Expected output:

```text
kg_constructor/data/benchmarks/wn18rr/triples.csv
```

#### 2. Prepare graph artefacts

```bash
python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/benchmarks/wn18rr.yaml --force
```

#### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/wn18rr
```

#### 4. Train

```bash
python -m models.rgcn.src.train --config models/rgcn/configs/benchmarks/wn18rr.yaml
```

#### 5. Evaluate

```bash
python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```

---

### FB15k-237

#### 1. Convert benchmark splits

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/benchmarks/fb15k237.yaml --force
```

Expected output:

```text
kg_constructor/data/benchmarks/fb15k237/triples.csv
```

#### 2. Prepare graph artefacts

```bash
python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/benchmarks/fb15k237.yaml --force
```

#### 3. Validate processed artefacts

```bash
python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/fb15k237
```

#### 4. Train

```bash
python -m models.rgcn.src.train --config models/rgcn/configs/benchmarks/fb15k237.yaml
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
python -m kg_constructor.src.run_kg_pipeline \
  --config kg_constructor/configs/biomedical/periods/2025-10_to_2026-03.yaml

python -m models.rgcn.src.prepare_graph \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml \
  --force

python -m models.rgcn.src.load_data \
  --processed_dir models/rgcn/data/processed/disease_to_drug/six_months_random_split

python -m models.rgcn.src.train \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml

python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test

python -m models.content-based.src.baseline \
  --config models/rgcn/configs/biomedical/disease_to_drug/2025-10_to_2026-03/main/all_supported_relations.yaml \
  --split test
```

### WN18RR

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/benchmarks/wn18rr.yaml --force

python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/benchmarks/wn18rr.yaml --force

python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/wn18rr

python -m models.rgcn.src.train --config models/rgcn/configs/benchmarks/wn18rr.yaml

python -m models.rgcn.src.evaluate --runs_dir results/runs/rgcn --split test
```

### FB15k-237

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/benchmarks/fb15k237.yaml --force

python -m models.rgcn.src.prepare_graph --config models/rgcn/configs/benchmarks/fb15k237.yaml --force

python -m models.rgcn.src.load_data --processed_dir models/rgcn/data/processed/fb15k237

python -m models.rgcn.src.train --config models/rgcn/configs/benchmarks/fb15k237.yaml

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

## Automated Tests

Activate the project environment and run the central regression suite from the
repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The suite covers filtered and tie-aware ranking, deterministic and disjoint
splits, known-positive direction, candidate typing, inverse-edge construction,
and inference-query behaviour.

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

No software license has been granted for this repository. All rights are
reserved by the author. Third-party datasets and dependencies remain subject
to their respective terms.

---

## Citation

GitHub exposes the software citation recorded in [`CITATION.cff`](CITATION.cff)
through the repository's **Cite this repository** option.

The published Knowledge Graph is available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22687285.svg)](https://doi.org/10.5281/zenodo.22687285)

When using the graph data, cite:

> dos Reis Garranha, G., Pós-de-Mina Pato, M., & Datia, N. (2026).
> *Biomedical Knowledge Graph for Disease-to-Drug Recommendation in Drug
> Repositioning* (Version 1.0.0) [Data set]. Zenodo.
> <https://doi.org/10.5281/zenodo.22687285>

If you use the R-GCN implementation, also cite the original architecture:

```bibtex
@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N. and Bloem, Peter and Van Den Berg, Rianne and Titov, Ivan and Welling, Max},
  booktitle={European Semantic Web Conference},
  pages={593--607},
  year={2018}
}

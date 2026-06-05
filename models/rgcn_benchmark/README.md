# Schlichtkrull-style R-GCN Link Prediction

This module contains a standalone, project-native implementation of the R-GCN + DistMult link prediction setup inspired by Schlichtkrull et al. It is intentionally separate from `models/rgcn` so it can be compared against the thesis model without mixing biomedical recommendation logic with generic Knowledge Graph Completion.

## Scope

This implementation is **link prediction only**:

```text
(h, r, ?) -> tail prediction
(?, r, t) -> head prediction
```

It does not implement `disease_to_drug`, `drug_to_disease`, inference over named biomedical entities, or the content-based baseline.

## What it implements

- R-GCN encoder with DGL `RelGraphConv`;
- DistMult decoder;
- inverse edges inside the encoder graph;
- official benchmark splits from the prepared `split` column;
- filtered ranking for both head and tail prediction;
- combined MRR/Hits@K plus separate `head_*` and `tail_*` metrics.

This is a PyTorch/DGL reimplementation inside the project pipeline, not a direct copy of the original TensorFlow/Keras code.

## Supported configs

| Dataset | Config | Expected processed dir | Runs dir |
|---|---|---|---|
| WN18 | `models/rgcn_benchmark/configs/wn18.yaml` | `models/rgcn/data/processed/wn18` | `models/rgcn_benchmark/runs/wn18` |
| WN18RR | `models/rgcn_benchmark/configs/wn18rr.yaml` | `models/rgcn/data/processed/wn18rr` | `models/rgcn_benchmark/runs/wn18rr` |
| FB15k | `models/rgcn_benchmark/configs/fb15k.yaml` | `models/rgcn/data/processed/fb15k` | `models/rgcn_benchmark/runs/fb15k` |
| FB15k-237 | `models/rgcn_benchmark/configs/fb15k237.yaml` | `models/rgcn/data/processed/fb15k237` | `models/rgcn_benchmark/runs/fb15k237` |

## Workflow

Convert the raw benchmark splits with `kg_constructor` when needed:

```bash
python -m kg_constructor.src.benchmarks.convert_benchmark --config kg_constructor/configs/wn18rr.yaml --force
```

Prepare the indexed graph artifacts with the existing generic R-GCN preprocessing script:

```bash
python -m models.rgcn.src.prepare_graph --config models/rgcn_schlichtkrull/configs/wn18rr.yaml --force
```

Train the standalone Schlichtkrull-style model:

```bash
python -m models.rgcn_benchmark.src.train --config models/rgcn_benchmark/configs/wn18rr.yaml
```

Evaluate the best checkpoint:

```bash
python -m models.rgcn_benchmark.src.evaluate --runs_dir models/rgcn_benchmark/runs/wn18rr --split test
```

## Colab notebook

If you want a GPU-friendly walkthrough, open [notebooks/rgcn_benchmark_colab.ipynb](../../notebooks/rgcn_benchmark_colab.ipynb). It includes Colab setup, Drive mount, dependency installation, train/evaluate commands, and quick smoke-test cells.

## Notes

- WN18 and FB15k are historical benchmarks and may produce inflated scores due to inverse-relation leakage.
- WN18RR and FB15k-237 should be preferred when discussing robust external benchmark behavior.
- For quick local checks, set `training.max_eval_triples` to a small number in the config.

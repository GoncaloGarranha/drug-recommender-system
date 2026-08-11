# PyKEEN Benchmarks

PyKEEN benchmarks used to compare standard knowledge graph models with the
task-specific R-GCN developed in this project.

## Task

The target relation is `INDICATION`.

Triples are represented as:

    (Drug, INDICATION, Disease)

The recommendation task is evaluated exclusively as head prediction:

    (?, INDICATION, Disease)

Only `Drug` entities are considered candidates.

The generic PyKEEN head-and-tail evaluation is therefore not used for the
reported results.

## Dataset

All benchmarks use the frozen publication dataset:

    kg_constructor/data/builds/2025-10_to_2026-03_publication/

Main statistics:

- 47,946 graph triples
- 9,277 entities
- 5 relations
- 6,239 INDICATION triples
- 4,991 target training triples
- 623 validation triples
- 625 test triples
- 46,698 triples available to PyKEEN during training
- 2,380 candidate drugs

## Model selection

Model selection was performed using validation MRR only.

First, two loss functions were compared for each architecture with
`embedding_dim=128` and `learning_rate=0.001`.

After selecting the loss, the following common parameter search was used:

    embedding_dim ∈ {64, 128}
    learning_rate ∈ {0.001, 0.01}

Common training settings include:

- seed 42
- Adam
- batch size 1024
- PseudoTyped negative sampling
- filtered negative sampling
- 1 negative per positive
- maximum 1000 epochs
- validation every 10 epochs
- early stopping patience of 20 validation evaluations

The complete selection protocol is stored in:

    configs/selection_protocol.yaml

Validation results from the model-selection experiments are stored in:

    results/publication/model_selection.csv

## Final models

| Model | Loss | Dim | LR | Valid MRR |
|---|---|---:|---:|---:|
| DistMult | BCEWithLogitsLoss | 128 | 0.001 | 0.4834 |
| TransE | SoftMarginRankingLoss | 128 | 0.001 | 0.4612 |
| RGCN | BCEWithLogitsLoss | 128 | 0.001 | 0.4422 |
| ComplEx | SoftplusLoss | 128 | 0.01 | 0.2894 |

Final configurations are stored in:

    configs/final/

## Test results

| Model | Full MRR | Hits@1 | Hits@3 | Hits@10 | 1-vs-50 MRR |
|---|---:|---:|---:|---:|---:|
| DistMult | 0.4192 | 0.2976 | 0.4800 | 0.6432 | 0.8059 |
| TransE | 0.4037 | 0.2928 | 0.4560 | 0.6176 | 0.7853 |
| RGCN | 0.3922 | 0.2912 | 0.4384 | 0.5920 | 0.7767 |
| ComplEx | 0.2130 | 0.1456 | 0.2368 | 0.3328 | 0.5408 |

Complete metrics are stored in:

    results/publication/final_results.csv

## Methodological note

The PyKEEN models and the custom R-GCN use the same publication dataset,
target splits, candidate drug set, and final evaluation protocol.

Their training objectives are not identical.

The custom R-GCN is optimized specifically for `INDICATION`, whereas the
PyKEEN benchmarks are trained as knowledge graph models using the available
training graph.

The PyKEEN RGCN should therefore be interpreted as a standard library
benchmark evaluated on the same recommendation task, rather than as an
exact reproduction of the custom R-GCN training procedure.

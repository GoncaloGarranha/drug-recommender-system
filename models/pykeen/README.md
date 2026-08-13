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

All final benchmarks use the publication dataset and the same processed graph
and target splits as the custom R-GCN.

Main statistics:

- 51,759 graph triples
- 10,380 entities
- 5 relations
- 6,143 INDICATION triples
- 4,914 target training triples
- 614 validation triples
- 615 test triples
- 50,530 triples available to PyKEEN during training
- 2,347 candidate drugs

The processed data used by the final configurations is specified explicitly
through `data.processed_dir`.

## Model selection

The final PyKEEN hyperparameters were selected in the earlier model-selection
stage using validation MRR only.

The selection protocol compared alternative losses and then evaluated common
embedding dimensions and learning rates. The complete protocol is stored in:

    configs/selection_protocol.yaml

After correction and reconstruction of the final publication dataset, the
selected hyperparameters were kept fixed. No additional hyperparameter search
was performed on the final dataset. The four models were retrained from scratch
with seed 42 using the same final configurations.

This preserves the previously defined model-selection decisions while avoiding
additional tuning after inspection of the final dataset.

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

Historical model-selection results are stored in:

    results/publication/model_selection.csv

## Final models

| Model | Loss | Dim | LR | Best epoch | Valid MRR |
|---|---|---:|---:|---:|---:|
| DistMult | BCEWithLogitsLoss | 128 | 0.001 | 310 | 0.4486 |
| TransE | SoftMarginRankingLoss | 128 | 0.001 | 350 | 0.4469 |
| RGCN | BCEWithLogitsLoss | 128 | 0.001 | 610 | 0.3901 |
| ComplEx | SoftplusLoss | 128 | 0.01 | 970 | 0.1363 |

Final configurations are stored in:

    configs/final/

Final runs are stored using a compact model/seed structure:

    results/runs/pykeen/distmult/seed_42/
    results/runs/pykeen/transe/seed_42/
    results/runs/pykeen/rgcn/seed_42/
    results/runs/pykeen/complex/seed_42/

## Test results

| Model | Full MRR | Hits@1 | Hits@3 | Hits@10 | 1-vs-50 MRR |
|---|---:|---:|---:|---:|---:|
| DistMult | 0.4109 | 0.2959 | 0.4650 | 0.6146 | 0.7951 |
| TransE | 0.4171 | 0.3057 | 0.4780 | 0.6146 | 0.7844 |
| RGCN | 0.3693 | 0.2634 | 0.4211 | 0.5772 | 0.7572 |
| ComplEx | 0.1376 | 0.0163 | 0.2130 | 0.4033 | 0.6013 |

Complete metrics are stored in:

    results/publication/final_results.csv

## Evaluation protocols

Two evaluation protocols are reported.

### Full filtered ranking

For each disease, all 2,347 candidate drugs are ranked. Other known positive
INDICATION associations for the same disease are filtered when calculating the
rank of the target drug.

### Fixed 1-vs-50

Each positive test association is evaluated against the same fixed set of 50
negative candidate drugs used by the other models in the project.

The fixed evaluation set ensures that the custom R-GCN, the content-based
baseline and all PyKEEN models are compared against identical sampled
candidates.

## Methodological note

The PyKEEN models and the custom R-GCN use the same final graph, target splits,
candidate drug set and evaluation protocols.

Their training objectives are not identical.

The custom R-GCN is optimized specifically for `INDICATION`, whereas the
PyKEEN benchmarks are trained as general knowledge graph models using the
available training graph.

The PyKEEN RGCN should therefore be interpreted as a standard library benchmark
evaluated on the same recommendation task, rather than as an exact reproduction
of the custom R-GCN training procedure.

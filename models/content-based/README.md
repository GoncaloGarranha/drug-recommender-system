# Content-Based Baseline

This module provides a simple **content-based baseline** for drug recommendation.
It is designed to complement the graph-based R-GCN pipeline by using only the
ingredient information available in the knowledge graph artifacts.

The baseline builds a text profile for each drug from its ingredient identifiers,
represents those profiles with **TF-IDF**, and ranks candidates with **cosine similarity**.

It currently evaluates the same drug-disease prediction setting used across the project:

- **Disease -> Drug recommendation**
- **Drug -> Disease analysis through ranking of known training drugs**

---

## Overview

The implementation lives in [src/baseline.py](src/baseline.py).

At a high level, the pipeline does the following:

1. Load the full triple set from `kg_constructor/data/output/triples/triples.csv`.
2. Extract all `HAS_INGREDIENT` relations.
3. Load R-GCN prepared artifacts from `models/rgcn/data/processed/`.
4. Build one ingredient-based text profile per drug candidate from the same DRUG pool used by R-GCN.
5. Vectorize the profiles with TF-IDF.
6. Recreate the target split using the same seed and split ratios from the R-GCN config.
7. Evaluate with filtered ranking metrics on the selected split (`valid` or `test`).

This makes the baseline fast, interpretable, and useful as a reference point for the learned model.

---

## How It Works

The model does not learn embeddings.
Instead, it treats each drug as a bag of ingredient tokens.

For every disease in the test set, the baseline looks at the drugs associated with that disease in the training set and scores each candidate drug by its best similarity to any of those known drugs.

In practice, this means:

- drugs with similar ingredient composition receive higher scores
- diseases with no training drugs fall back to a large default rank
- the final output is a ranking, not a binary prediction

This is intentionally simpler than R-GCN and is meant to provide a transparent comparison point.

---

## Dependencies

The script relies on the project-wide Python environment plus:

- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`

The required data artifacts are:

- `kg_constructor/data/output/triples/triples.csv`
- `models/rgcn/data/processed/triples_indexed.csv`
- `models/rgcn/data/processed/entity2id.json`
- `models/rgcn/data/processed/entity_types.json`

---

## Run

From the repository root:

```bash
python models/content-based/src/baseline.py --config models/rgcn/configs/disease_to_drug.yaml --split test
```

The script will:

- load ingredient triples and R-GCN processed artifacts
- use the same split policy from the provided R-GCN config
- build the TF-IDF similarity matrix over the same DRUG candidate pool
- compute filtered ranking metrics on `valid` or `test`

---

## Output

The script prints the following metrics:

- `Hits@1`
- `Hits@3`
- `Hits@10`
- `MRR`

These values are calculated over the generated rankings for the test split.

---

## Design Notes

This baseline follows the same project philosophy used in the other pipelines:

- keep the data source explicit
- make the ranking logic easy to inspect
- prefer reproducible preprocessing over hidden state
- keep the baseline lightweight so it can be compared directly with R-GCN

The implementation is intentionally minimal. It is not a production recommender,
but a clean reference system for testing whether ingredient-based similarity alone can recover useful drug-disease patterns under the same evaluation protocol used by R-GCN.

Each run also logs its metrics to `models/performance_leaderboard.csv`, so the
content-based baseline stays comparable with the R-GCN experiments.

---

## Limitations

- It only uses ingredient-level information.
- It depends on the availability of `HAS_INGREDIENT` relations in the full triples file.
- It depends on ingredient coverage for the candidate DRUG pool.
- It does not model graph structure, relation semantics, or learned latent features.

If you want, this README can be extended with a comparison section against R-GCN, or with a small example of the ranking output.

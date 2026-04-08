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
3. Load `models/rgcn/data/processed/entity2id.json` to keep only mapped drug entities.
4. Build one ingredient-based text profile per drug.
5. Vectorize the profiles with TF-IDF.
6. Rank candidate drugs by cosine similarity.
7. Report ranking metrics on a random test split.

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
- `kg_constructor/data/output/triples/triples.filtered.csv`
- `models/rgcn/data/processed/entity2id.json`

---

## Run

From the repository root:

```bash
python models/content-based/src/baseline.py
```

The script will:

- load the triples and entity mapping
- create a random 90/10 train-test split from the filtered triples
- build the TF-IDF similarity matrix
- compute ranking metrics for the held-out triples

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
but a clean reference system for testing whether ingredient-based similarity alone can recover useful drug-disease patterns.

Each run also logs its metrics to `models/performance_leaderboard.csv`, so the
content-based baseline stays comparable with the R-GCN experiments.

---

## Limitations

- It only uses ingredient-level information.
- It depends on the availability of `HAS_INGREDIENT` relations in the full triples file.
- The train/test split is created inside the script, so results may vary slightly across runs.
- It does not model graph structure, relation semantics, or learned latent features.

If you want, this README can be extended with a comparison section against R-GCN, or with a small example of the ranking output.

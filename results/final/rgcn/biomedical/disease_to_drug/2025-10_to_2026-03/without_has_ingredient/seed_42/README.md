# R-GCN without HAS_INGREDIENT

This directory contains the consolidated result of the relation study in
which `HAS_INGREDIENT` was removed from the graph context.

## Experiment

- Dataset period: `2025-10_to_2026-03`
- Task: `disease_to_drug`
- Target relation: `INDICATION`
- Seed: `42`
- Removed relation: `HAS_INGREDIENT`
- Best epoch: `8190`
- Test MRR: `0.723993`

## Source run

`results/runs/rgcn/biomedical/disease_to_drug/six_months_without_has_ingredient/seed_42`

The checkpoint and complete split files remain in the source run directory.
This final directory stores the configuration, evaluation metrics, split
metadata and provenance information required for reporting.
